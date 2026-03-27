"""Output backends for the unified tokenization loop.

- ``DirectBackend``: assembles and writes final bin/idx immediately.
- ``SpillBackend``: writes keyed component payloads for offline rebuild.

The executor instantiates one based on ``use_spill = multi_image or mode == "interleave"``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .checkpoint import WorkerStats
from ..indexing.planning.tokenization_plan import IMAGE, TEXT

logger = logging.getLogger(__name__)


class DirectBackend:
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
        group_slices: Optional[np.ndarray],
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


class SpillBackend:
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
        group_slices: Optional[np.ndarray],
        resize_height: int,
        resize_width: int,
        plan: Any,
        tokenizer: Any,
        stats: WorkerStats,
    ) -> dict:
        import time

        IMAGE_KIND = int(IMAGE)
        TEXT_KIND = int(TEXT)

        t0 = time.perf_counter()

        if plan.mode == "interleave":
            self._write_interleave_batch(
                image_tokens=image_tokens,
                texts=texts,
                component_indices=component_indices,
                group_slices=group_slices,
                resize_height=resize_height,
                resize_width=resize_width,
                plan=plan,
                tokenizer=tokenizer,
                stats=stats,
            )
            write_ms = (time.perf_counter() - t0) * 1000
            return {"write_ms": write_ms}

        # Spill image components without per-image BOS/EOS wrappers.
        for seq, comp_idx in zip(image_tokens, component_indices):
            comp_idx = int(comp_idx)
            doc_id = int(plan.components.document_id[comp_idx])
            ci = int(plan.components.component_index[comp_idx])
            tokens_cpu = self._strip_component_wrapper(seq, tokenizer)
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

        # Spill one text component per document for non-interleave modes.
        if texts is not None:
            if group_slices is not None:
                for g_idx, (start, end) in enumerate(group_slices):
                    start, end = int(start), int(end)
                    if start >= end:
                        continue
                    text = texts[g_idx] if g_idx < len(texts) else None
                    if text is None:
                        continue
                    doc_comp_indices = component_indices[start:end]
                    # Spill doc-level text only once, from the batch containing image_index=0.
                    if not np.any(plan.components.image_index[doc_comp_indices] == 0):
                        continue
                    doc_id = int(plan.components.document_id[int(doc_comp_indices[0])])
                    text_np = self._tokenize_doc_text(text, plan.mode, tokenizer)
                    text_ci = int(plan.documents.num_images[doc_id])
                    self._writer.add_component(
                        document_id=doc_id,
                        component_index=text_ci,
                        kind=TEXT_KIND,
                        tokens=text_np,
                    )
                    stats.text_tokens += len(text_np)
                    stats.tokens_generated += len(text_np)
            else:
                for i, comp_idx in enumerate(component_indices):
                    text = texts[i] if i < len(texts) else None
                    if text is None:
                        continue
                    comp_idx = int(comp_idx)
                    doc_id = int(plan.components.document_id[comp_idx])
                    text_np = self._tokenize_doc_text(text, plan.mode, tokenizer)
                    text_ci = int(plan.documents.num_images[doc_id])
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

    @staticmethod
    def _strip_component_wrapper(tokens: torch.Tensor, tokenizer: Any) -> np.ndarray:
        """Drop outer BOS/EOS from image components before spilling."""
        seq = tokens.cpu() if tokens.is_cuda else tokens
        if seq.ndim != 1:
            seq = seq.reshape(-1)
        if (
            seq.numel() >= 2
            and hasattr(tokenizer, "bos_id")
            and hasattr(tokenizer, "eos_id")
            and int(seq[0]) == int(tokenizer.bos_id)
            and int(seq[-1]) == int(tokenizer.eos_id)
        ):
            seq = seq[1:-1]
        return seq.numpy().astype(np.int32, copy=False)

    @staticmethod
    def _tokenize_doc_text(text: Any, mode: str, tokenizer: Any) -> np.ndarray:
        """Tokenize one document-level text payload for non-interleave modes."""
        if mode == "sft":
            from vision_tokenization.discrete.conversation import apply_conversation_policy

            messages = apply_conversation_policy(text, tokenizer.conversation_policy)
            text_tokens, _num_images, _image_positions = tokenizer._tokenize_conversation_text_cpu(messages)
            seq = text_tokens.cpu() if text_tokens.is_cuda else text_tokens
            return seq.numpy().astype(np.int32, copy=False)

        encoded = tokenizer.text_tokenizer(
            text,
            truncation=False,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        seq = encoded.cpu() if encoded.is_cuda else encoded
        return seq.numpy().astype(np.int32, copy=False)

    def _write_interleave_batch(
        self,
        *,
        image_tokens: List[torch.Tensor],
        texts: Optional[List[Any]],
        component_indices: np.ndarray,
        group_slices: Optional[np.ndarray],
        resize_height: int,
        resize_width: int,
        plan: Any,
        tokenizer: Any,
        stats: WorkerStats,
    ) -> None:
        IMAGE_KIND = int(IMAGE)
        TEXT_KIND = int(TEXT)

        if texts is None or group_slices is None:
            raise ValueError("Interleave spill requires grouped parsed documents")

        # Loader-owned contract for interleave:
        # - valid docs arrive as structured segment dicts
        # - invalid / manifest-mismatched docs arrive as None
        normalized_texts: List[Optional[List[Dict[str, Any]]]] = []
        for text_payload in texts:
            if text_payload is None:
                normalized_texts.append(None)
                continue

            if isinstance(text_payload, list) and all(isinstance(seg, dict) for seg in text_payload):
                normalized_texts.append(text_payload)
                continue

            logger.warning(
                "Interleave backend received non-structured text payload "
                "(type=%s); expected List[dict] from loader. Skipping document.",
                type(text_payload).__name__,
            )
            normalized_texts.append(None)

        # Batch-tokenize all non-empty text segments across the grouped docs.
        flat_texts: List[str] = []
        doc_text_map: List[List[tuple[int, int]]] = []
        doc_image_comp_indices: List[List[int]] = []
        for segments in normalized_texts:
            text_entries: List[tuple[int, int]] = []
            image_component_positions: List[int] = []
            runtime_ci = 0
            if segments is None:
                doc_text_map.append(text_entries)
                doc_image_comp_indices.append(image_component_positions)
                continue
            for seg in segments or []:
                seg_type = seg.get("type")
                if seg_type == "text":
                    seg_text = seg.get("text")
                    if seg_text:
                        text_entries.append((runtime_ci, len(flat_texts)))
                        flat_texts.append(seg_text)
                        runtime_ci += 1
                elif seg_type == "image":
                    image_component_positions.append(runtime_ci)
                    runtime_ci += 1
            doc_text_map.append(text_entries)
            doc_image_comp_indices.append(image_component_positions)

        text_tokens_by_flat_idx: List[np.ndarray] = []
        if flat_texts:
            encoded = tokenizer.text_tokenizer(
                flat_texts,
                truncation=False,
                add_special_tokens=False,
                return_tensors=None,
                padding=False,
            )
            text_tokens_by_flat_idx = [
                np.asarray(ids, dtype=np.int32) for ids in encoded["input_ids"]
            ]

        for g_idx, (start, end) in enumerate(group_slices):
            start, end = int(start), int(end)
            if start >= end:
                continue

            doc_comp_indices = component_indices[start:end]
            doc_id = int(plan.components.document_id[int(doc_comp_indices[0])])
            image_indices = plan.components.image_index[doc_comp_indices]
            if normalized_texts[g_idx] is None:
                logger.warning(
                    "Skipping interleave doc %s: loader returned no validated segments",
                    doc_id,
                )
                stats.samples_skipped += 1
                continue
            image_component_positions = doc_image_comp_indices[g_idx]

            max_image_index = int(image_indices.max()) if len(image_indices) > 0 else -1
            if max_image_index >= len(image_component_positions):
                logger.warning(
                    "Interleave doc %s has image_index outside parsed segment range — skipping batch fragment",
                    doc_id,
                )
                stats.samples_skipped += 1
                continue

            # Spill text segments once per document, from the batch containing image_index=0.
            if np.any(image_indices == 0):
                for runtime_ci, text_flat_idx in doc_text_map[g_idx]:
                    text_np = text_tokens_by_flat_idx[text_flat_idx]
                    self._writer.add_component(
                        document_id=doc_id,
                        component_index=runtime_ci,
                        kind=TEXT_KIND,
                        tokens=text_np,
                    )
                    stats.text_tokens += len(text_np)
                    stats.tokens_generated += len(text_np)

            for local_pos, comp_idx in enumerate(doc_comp_indices):
                image_index = int(plan.components.image_index[int(comp_idx)])
                runtime_ci = int(image_component_positions[image_index])
                tokens_np = self._strip_component_wrapper(image_tokens[start + local_pos], tokenizer)
                self._writer.add_component(
                    document_id=doc_id,
                    component_index=runtime_ci,
                    kind=IMAGE_KIND,
                    tokens=tokens_np,
                    resize_height=resize_height,
                    resize_width=resize_width,
                )
                stats.samples_processed += 1
                stats.image_tokens += len(tokens_np)
                stats.tokens_generated += len(tokens_np)


