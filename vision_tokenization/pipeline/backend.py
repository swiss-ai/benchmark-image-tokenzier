"""Output backends for the unified tokenization loop.

Two backends:
- ``DirectBackend``: assembles and writes final bin/idx immediately.
  Used for single-image image_only, image2text, text2image, single-image sft.
- ``SpillBackend``: writes keyed component payloads for offline rebuild.
  Used for multi-image and interleave.

The unified loop calls ``backend.write_batch(...)`` after GPU encode.
The backend handles assembly/writing differences.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .checkpoint import WorkerStats

logger = logging.getLogger(__name__)


class OutputBackend(ABC):
    """Abstract output backend for the tokenization loop."""

    @abstractmethod
    def open(self, output_dir: str, rank: int, resume_state: Optional[dict] = None) -> None:
        ...

    @abstractmethod
    def write_batch(
        self,
        image_tokens: List[torch.Tensor],
        texts: Optional[List[Any]],
        component_indices: np.ndarray,
        resize_height: int,
        resize_width: int,
        plan: Any,
        tokenizer: Any,
        stats: WorkerStats,
    ) -> dict:
        """Write one batch's results. Returns timing dict."""
        ...

    @abstractmethod
    def checkpoint(self) -> Any:
        """Flush and checkpoint. Returns checkpoint metadata."""
        ...

    @abstractmethod
    def finalize(self) -> None:
        ...


class DirectBackend(OutputBackend):
    """Assemble and write final bin/idx sequences immediately.

    Each batch produces complete sequences: BOS + image_struct + text + EOS.
    No intermediate spill, no offline rebuild.
    """

    def __init__(self, mode: str, seqlen_threshold: Optional[int] = None):
        self._mode = mode
        self._seqlen_threshold = seqlen_threshold
        self._handler = None
        self._chunk_id = 0

    def open(self, output_dir: str, rank: int, resume_state: Optional[dict] = None) -> None:
        from .direct.handler import TokenizationHandler
        from .direct.writer import MicroShardWriter, SplitMicroShardWriter

        if self._seqlen_threshold is not None:
            writer = SplitMicroShardWriter(seqlen_threshold=self._seqlen_threshold)
        else:
            writer = MicroShardWriter()

        needs_text = self._mode in ("sft", "image2text", "text2image")
        self._handler = TokenizationHandler(writer, needs_text)

        start_chunk = 0
        if resume_state:
            start_chunk = resume_state.get("chunk_id", 0) + 1
        self._chunk_id = start_chunk

        if self._seqlen_threshold is not None:
            self._handler.setup_writer(
                output_dir, rank,
                resume_state.get("stage2_chunk_id", 0) + 1 if resume_state else 0,
                resume_state.get("lct_chunk_id", 0) + 1 if resume_state else 0,
                None,  # tokenizer set later
            )
        else:
            self._handler.setup_writer(output_dir, rank, start_chunk, None)

    def write_batch(
        self,
        image_tokens: List[torch.Tensor],
        texts: Optional[List[Any]],
        component_indices: np.ndarray,
        resize_height: int,
        resize_width: int,
        plan: Any,
        tokenizer: Any,
        stats: WorkerStats,
    ) -> dict:
        import time

        resize_size = (resize_height, resize_width)
        # Reconstruct the images list as PIL images are already consumed by tokenizer.
        # The handler.process_batch expects images, but we already have tokens.
        # For direct backend, we call the handler directly.
        t0 = time.perf_counter()
        process_timing = self._handler.process_batch_from_tokens(
            image_tokens, resize_size, tokenizer, stats,
            texts=texts,
            timing_enabled=True,
        )
        write_ms = (time.perf_counter() - t0) * 1000
        return {"write_ms": write_ms, **(process_timing or {})}

    def checkpoint(self) -> Any:
        done = self._handler.checkpoint_writer()
        self._chunk_id = done + 1 if isinstance(done, int) else self._chunk_id + 1
        return {"chunk_id": done}

    def finalize(self) -> None:
        if self._handler:
            self._handler.finalize_writer()


class SpillBackend(OutputBackend):
    """Write keyed component payloads for offline rebuild."""

    def __init__(self):
        self._writer = None

    def open(self, output_dir: str, rank: int, resume_state: Optional[dict] = None) -> None:
        from .spill import ComponentSpillWriter, recover_worker_shards

        self._writer = ComponentSpillWriter(output_dir, rank, token_dtype=np.int32)
        start_shard = 0
        if resume_state:
            rank_dir = Path(output_dir) / f"rank_{rank:04d}"
            start_shard = recover_worker_shards(rank_dir)
        self._writer.open(start_shard_id=start_shard)

    def write_batch(
        self,
        image_tokens: List[torch.Tensor],
        texts: Optional[List[Any]],
        component_indices: np.ndarray,
        resize_height: int,
        resize_width: int,
        plan: Any,
        tokenizer: Any,
        stats: WorkerStats,
    ) -> dict:
        import time
        from .spill import ComponentSpillWriter

        IMAGE_KIND = 0
        TEXT_KIND = 1

        t0 = time.perf_counter()

        # Spill image components
        for seq, comp_idx in zip(image_tokens, component_indices):
            comp_idx = int(comp_idx)
            doc_id = int(plan.comp_document_id[comp_idx])
            ci = int(plan.comp_component_index[comp_idx])
            tokens_cpu = seq.cpu().numpy() if seq.is_cuda else seq.numpy()
            self._writer.add_component(
                document_id=doc_id,
                component_index=ci,
                kind=IMAGE_KIND,
                tokens=tokens_cpu,
                resize_height=resize_height,
                resize_width=resize_width,
            )
            stats.samples_processed += 1
            stats.image_tokens += len(tokens_cpu)
            stats.tokens_generated += len(tokens_cpu)

        # Spill text components (deduplicate per document)
        if texts is not None:
            seen_docs = set()
            for i, comp_idx in enumerate(component_indices):
                comp_idx = int(comp_idx)
                doc_id = int(plan.comp_document_id[comp_idx])
                if doc_id in seen_docs:
                    continue
                seen_docs.add(doc_id)
                text = texts[i] if i < len(texts) else None
                if text is not None and hasattr(tokenizer, 'tokenize_text'):
                    text_tokens = tokenizer.tokenize_text(text)
                    if text_tokens is not None:
                        text_np = text_tokens.cpu().numpy() if hasattr(text_tokens, 'cpu') else np.array(text_tokens, dtype=np.int32)
                        text_ci = int(plan.doc_num_components[doc_id]) - 1
                        self._writer.add_component(
                            document_id=doc_id,
                            component_index=text_ci,
                            kind=TEXT_KIND,
                            tokens=text_np,
                        )
                        stats.text_tokens += len(text_np)
                        stats.tokens_generated += len(text_np)

        write_ms = (time.perf_counter() - t0) * 1000
        return {"write_ms": write_ms}

    def checkpoint(self) -> Any:
        done = self._writer.checkpoint()
        return {"shard_id": done}

    def finalize(self) -> None:
        if self._writer:
            self._writer.finalize()


def select_backend(mode: str, multi_image: bool, seqlen_threshold: Optional[int] = None) -> OutputBackend:
    """Select the appropriate backend based on mode and multi_image flag."""
    if multi_image or mode == "interleave":
        return SpillBackend()
    return DirectBackend(mode=mode, seqlen_threshold=seqlen_threshold)
