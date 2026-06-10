"""Output backends for the unified tokenization loop.

- ``DirectBackend``: assembles and writes final bin/idx immediately.
- ``SpillBackend``: writes keyed component payloads for offline rebuild.

The executor instantiates one based on ``use_spill = multi_image or mode == "interleave"``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..runtime.checkpoint import WorkerStats
from ...indexing.planning.tokenization_plan import IMAGE, TEXT
from ...discrete.sft_segments import build_segment_component_maps

logger = logging.getLogger(__name__)


def write_rank_success_marker(output_dir: Path, rank: int) -> None:
    """Mark this rank as cleanly finalized by touching ``rank_NNNN/_SUCCESS``.

    Read by ``merge._all_ranks_done`` to gate the merge step. Both backends
    write the same marker at the same path so the merge contract is uniform.
    """
    rank_dir = output_dir / f"rank_{rank:04d}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    (rank_dir / "_SUCCESS").touch()


class DirectBackend:
    """Assemble and write final bin/idx sequences immediately.

    Each batch produces complete sequences: BOS + image_struct + text + EOS.
    No intermediate spill, no offline rebuild.
    """

    def __init__(self, mode: str):
        self._mode = mode
        self._handler = None
        self._output_dir: Optional[Path] = None
        self._rank: Optional[int] = None

    def open(self, output_dir: str, rank: int, writer_state: Optional[dict] = None, tokenizer=None) -> None:
        """*writer_state* is the opaque dict this backend returned from
        ``checkpoint()`` — round-tripped through the checkpoint, interpreted
        only by the writer that produced it."""
        from .direct.handler import TokenizationHandler
        from .direct.writer import MicroShardWriter

        writer = MicroShardWriter()
        self._writer = writer
        needs_text = self._mode in ("sft", "image2text", "text2image")
        self._handler = TokenizationHandler(writer, needs_text)

        self._output_dir = Path(output_dir)
        self._rank = rank

        start_chunk = MicroShardWriter.resume_chunk(writer_state) if writer_state else 0
        self._handler.setup_writer(output_dir, rank, start_chunk, tokenizer)
        if writer_state:
            writer.restore(writer_state)

    def completed_files(self) -> list:
        """Finalized-shard records for the rank completion manifest."""
        return list(self._writer.finalized_files)

    def write_batch(
        self,
        images: List,
        resize_size: Tuple[int, int],
        texts: Optional[List[Any]],
        group_slices: Optional[np.ndarray],
        tokenizer: Any,
        stats: WorkerStats,
        device: str,
        timing_enabled: bool = False,
    ) -> dict:
        """Tokenize + write in one step. The handler owns the full pipeline."""
        return self._handler.process_batch(
            images, resize_size, tokenizer, stats, device,
            texts=texts, group_slices=group_slices, timing_enabled=timing_enabled,
        )

    def checkpoint(self) -> dict:
        """Roll the chunk; return the writer's opaque resume state."""
        return self._handler.checkpoint_writer()

    def finalize(self) -> None:
        if self._handler:
            self._handler.finalize_writer()
        if self._output_dir is not None and self._rank is not None:
            write_rank_success_marker(self._output_dir, self._rank)


class SpillBackend:
    """Write keyed component payloads for offline rebuild."""

    def __init__(self):
        self._writer = None
        self._dropped_sft_docs: set[int] = set()
        self._output_dir: Optional[Path] = None
        self._rank: Optional[int] = None

    def open(self, output_dir: str, rank: int, writer_state: Optional[dict] = None) -> None:
        """Spill resume is filesystem-truth: *writer_state* marks that a
        resume was requested; the shard cursor is recovered from disk."""
        from .spill import ComponentSpillWriter, recover_worker_shards

        self._writer = ComponentSpillWriter(output_dir, rank, token_dtype=np.int32)
        self._dropped_sft_docs.clear()
        self._output_dir = Path(output_dir)
        self._rank = rank
        start_shard = 0
        if writer_state is not None:
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
        """Spill one batch's components to disk for offline rebuild."""
        import time
        t0 = time.perf_counter()
        kwargs = dict(
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
        if plan.mode == "interleave":
            self._write_interleave_batch(**kwargs)
        elif plan.mode == "sft":
            self._write_sft_batch(**kwargs)
        else:
            self._write_unsegmented_batch(**kwargs)
        return {"write_ms": (time.perf_counter() - t0) * 1000}

    def _write_unsegmented_batch(
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
        """Spill image components and one doc-level text per doc.

        Used by image_only / image2text / text2image modes — every component
        has a stable plan-level component_index, so spilling is direct.
        """
        IMAGE_KIND = int(IMAGE)

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
        if texts is None:
            return
        if group_slices is None:
            # Spill mode implies multi_image, and the executor always builds
            # group_slices for multi_image runs — flat text spill would silently
            # produce docs with missing text at rebuild.
            raise ValueError("Unsegmented spill with texts requires group_slices")
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
            self._spill_doc_text(text, doc_id, plan, tokenizer, stats)

    def _spill_doc_text(
        self,
        text: Any,
        doc_id: int,
        plan: Any,
        tokenizer: Any,
        stats: WorkerStats,
    ) -> None:
        """Tokenize and spill one document-level text payload under its canonical component index."""
        text_np = self._tokenize_doc_text(text, plan.mode, tokenizer)
        text_ci = int(plan.documents.num_images[doc_id])
        self._writer.add_component(
            document_id=doc_id,
            component_index=text_ci,
            kind=int(TEXT),
            tokens=text_np,
        )
        stats.text_tokens += len(text_np)
        stats.tokens_generated += len(text_np)

    def checkpoint(self) -> dict:
        """Flush the shard; resume state is recovered from disk, not stored."""
        self._writer.checkpoint()
        return {}

    def finalize(self) -> None:
        if self._writer:
            self._writer.finalize()
        if self._output_dir is not None and self._rank is not None:
            write_rank_success_marker(self._output_dir, self._rank)

    def _mark_sft_doc_dropped(
        self,
        *,
        doc_id: int,
        stats: WorkerStats,
        reason: str,
    ) -> None:
        """Mark one SFT doc incomplete so later fragments do not spill partial data."""
        if doc_id in self._dropped_sft_docs:
            return
        self._dropped_sft_docs.add(doc_id)
        logger.warning("Dropping SFT doc %s from spill: %s", doc_id, reason)
        stats.samples_skipped += 1

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

        self._write_segmented_batch(
            normalized_texts=normalized_texts,
            image_tokens=image_tokens,
            component_indices=component_indices,
            group_slices=group_slices,
            resize_height=resize_height,
            resize_width=resize_width,
            plan=plan,
            tokenizer=tokenizer,
            stats=stats,
        )

    def _write_sft_batch(
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
        if texts is None or group_slices is None:
            raise ValueError("SFT spill requires grouped conversations")

        # normalized_texts must stay index-aligned with group_slices; every
        # branch below appends exactly one entry per group.
        normalized_texts: List[Optional[List[Dict[str, Any]]]] = []
        for g_idx, (start, end) in enumerate(group_slices):
            start, end = int(start), int(end)
            if start >= end:
                normalized_texts.append(None)
                continue

            raw_text = texts[g_idx] if g_idx < len(texts) else None
            if raw_text is None:
                normalized_texts.append(None)
                continue

            doc_comp_indices = component_indices[start:end]
            doc_id = int(plan.components.document_id[int(doc_comp_indices[0])])

            try:
                rendered_doc = tokenizer.render_sft_document(
                    raw_text,
                    expected_num_images=int(plan.documents.num_images[doc_id]),
                )
                normalized_texts.append(rendered_doc.segments)
            except ValueError as exc:
                logger.warning(
                    "Failed to render grouped SFT conversation %d for spill — skipping: %s",
                    g_idx,
                    exc,
                )
                normalized_texts.append(None)

        self._write_segmented_batch(
            normalized_texts=normalized_texts,
            image_tokens=image_tokens,
            component_indices=component_indices,
            group_slices=group_slices,
            resize_height=resize_height,
            resize_width=resize_width,
            plan=plan,
            tokenizer=tokenizer,
            stats=stats,
        )

    def _write_segmented_batch(
        self,
        *,
        normalized_texts: Sequence[Optional[Sequence[Dict[str, Any]]]],
        image_tokens: List[torch.Tensor],
        component_indices: np.ndarray,
        group_slices: np.ndarray,
        resize_height: int,
        resize_width: int,
        plan: Any,
        tokenizer: Any,
        stats: WorkerStats,
    ) -> None:
        """Spill grouped structured documents with runtime component ordering."""
        IMAGE_KIND = int(IMAGE)
        TEXT_KIND = int(TEXT)

        flat_texts, doc_text_map, doc_image_comp_indices = build_segment_component_maps(normalized_texts)

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
            if plan.mode == "sft" and doc_id in self._dropped_sft_docs:
                continue
            image_indices = plan.components.image_index[doc_comp_indices]
            if normalized_texts[g_idx] is None:
                if plan.mode == "sft":
                    self._mark_sft_doc_dropped(
                        doc_id=doc_id,
                        stats=stats,
                        reason="loader/render step returned no structured segments",
                    )
                else:
                    logger.warning(
                        "Skipping %s doc %s: loader/render step returned no structured segments",
                        plan.mode,
                        doc_id,
                    )
                    stats.samples_skipped += 1
                continue
            image_component_positions = doc_image_comp_indices[g_idx]

            max_image_index = int(image_indices.max()) if len(image_indices) > 0 else -1
            if max_image_index >= len(image_component_positions):
                if plan.mode == "sft":
                    self._mark_sft_doc_dropped(
                        doc_id=doc_id,
                        stats=stats,
                        reason="image_index lies outside runtime segment range",
                    )
                else:
                    logger.warning(
                        "%s doc %s has image_index outside runtime segment range — skipping batch fragment",
                        plan.mode,
                        doc_id,
                    )
                    stats.samples_skipped += 1
                continue

            if not doc_text_map[g_idx]:
                if plan.mode == "sft":
                    self._mark_sft_doc_dropped(
                        doc_id=doc_id,
                        stats=stats,
                        reason="rendered structure has no text segments",
                    )
                else:
                    logger.warning("%s doc %s has no text segments — skipping", plan.mode, doc_id)
                    stats.samples_skipped += 1
                continue

            # Spill text segments exactly once per document — only from the
            # fragment that owns image_index == 0, so fragmented docs don't
            # emit duplicate text.
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
