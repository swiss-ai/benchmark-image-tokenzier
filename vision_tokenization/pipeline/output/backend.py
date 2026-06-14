"""Output backends for the unified tokenization loop.

- ``DirectBackend``: assembles and writes final bin/idx immediately.
- ``SpillBackend``: writes keyed component payloads for offline rebuild.
- ``AlignmentPayloadBackend``: writes final alignment views/tokens/raw payload.

The executor instantiates one based on mode/multi_image.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ..runtime.checkpoint import WorkerStats
from ...indexing.alignment.payload import (
    RAW_SCHEMA,
    TOKEN_DTYPE,
    VIEW_SCHEMA,
    _alignment_media_refs,
    split_payload_rows,
)
from ...indexing.planning.tokenization_plan import IMAGE, TEXT
from ...discrete.sft_segments import build_segment_component_maps

logger = logging.getLogger(__name__)


class EncodeIncompleteError(RuntimeError):
    """A view row references a media that was never encoded.

    The alignment publish-stage completeness gate, raised from
    ``AlignmentPayloadBackend.finalize``. The executor only calls finalize on a
    clean loop, so this surfaces a genuine encode gap — never a masked loop
    error.
    """


def write_rank_success_marker(output_dir: Path, rank: int) -> None:
    """Mark this rank's SPILL as complete by touching ``rank_NNNN/_SUCCESS``.

    Spill-only: ``rebuild_rank`` refuses rank dirs without it. Rank-level
    completion is asserted by the completion manifest, not this marker.

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

    name = "direct"

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
        # No marker: rank completion is asserted by the manifest the
        # executor publishes after finalize succeeds.
        if self._handler:
            self._handler.finalize_writer()


class SpillBackend:
    """Write keyed component payloads for offline rebuild."""

    name = "spill"

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


def _fsync_file(path) -> None:
    """fsync a closed file's contents to stable storage (by path) before its
    atomic os.replace, matching the token-file fsync."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class AlignmentPayloadBackend:
    """Write the final alignment payload directly.

    The executor still GPU-encodes unique media, but this backend writes the
    public ``views/``, ``tokens/``, and ``raw/`` artifacts directly. It does not
    create a content-addressed media store on disk.
    """

    name = "alignment_payload"

    def __init__(
        self,
        media_inventory: List[Any],
        view_rows: List[dict],
        *,
        public_output_dir: str | Path,
        requested_validation_rows: int,
        split_key: str = "prompt_id",
        seed: int = 42,
        raw_flush_rows: int = 1024,
    ):
        self._inventory = media_inventory
        self._public_dir = Path(public_output_dir)
        train_rows, validation_rows = split_payload_rows(
            view_rows,
            requested_validation_rows=requested_validation_rows,
            split_key=split_key,
            seed=seed,
        )
        self._rows_by_split: dict[str, list[dict]] = {"train": train_rows}
        if validation_rows:
            self._rows_by_split["validation"] = validation_rows

        self._media_splits: dict[str, set[str]] = {}
        self._first_occurrence: dict[str, dict[str, tuple[str, int]]] = {
            split: {} for split in self._rows_by_split
        }
        self._image_ref_counts: dict[str, int] = {split: 0 for split in self._rows_by_split}
        for split, rows in self._rows_by_split.items():
            first = self._first_occurrence[split]
            for row in rows:
                sample_id = str(row.get("prompt_id", ""))
                for image_index, media_id in enumerate(_alignment_media_refs(row)):
                    self._image_ref_counts[split] += 1
                    self._media_splits.setdefault(media_id, set()).add(split)
                    first.setdefault(media_id, (sample_id, image_index))

        self._raw_flush_rows = int(raw_flush_rows)
        self._token_files: dict[str, Any] = {}
        self._token_tmp: dict[str, Path] = {}
        self._token_final: dict[str, Path] = {}
        self._token_offsets: dict[str, int] = {}
        self._raw_tmp: dict[str, Path] = {}
        self._raw_final: dict[str, Path] = {}
        self._raw_writers: dict[str, pq.ParquetWriter] = {}
        self._raw_buffers: dict[str, list[dict]] = {}
        self._raw_counts: dict[str, int] = {}
        self._locations: dict[str, dict[str, dict]] = {
            split: {} for split in self._rows_by_split
        }
        self._completed_files: list[dict] = []
        self._n_media = 0
        self._n_tokens = 0
        self.result: dict | None = None

    def open(self, output_dir: str, rank: int, writer_state: Optional[dict] = None) -> None:
        if rank != 0:
            raise ValueError("alignment payload backend is single-rank")
        if writer_state:
            raise ValueError("alignment payload backend is seal-at-end; resume is unsupported")

        self._public_dir.mkdir(parents=True, exist_ok=True)
        # No upfront wipe — a failed re-run keeps the prior store intact; the
        # old payload and manifest survive until finalize's atomic os.replace.
        for split in self._rows_by_split:
            token_rel = f"tokens/{split}-00000.i32"
            raw_rel = f"raw/{split}-00000.parquet"
            token_final = self._public_dir / token_rel
            raw_final = self._public_dir / raw_rel
            token_final.parent.mkdir(parents=True, exist_ok=True)
            raw_final.parent.mkdir(parents=True, exist_ok=True)

            token_tmp = token_final.with_suffix(token_final.suffix + ".tmp")
            raw_tmp = raw_final.with_suffix(raw_final.suffix + ".tmp")
            for tmp in (token_tmp, raw_tmp):
                if tmp.exists():
                    tmp.unlink()

            self._token_tmp[split] = token_tmp
            self._token_final[split] = token_final
            self._token_files[split] = open(token_tmp, "wb")
            self._token_offsets[split] = 0
            self._raw_tmp[split] = raw_tmp
            self._raw_final[split] = raw_final
            self._raw_buffers[split] = []
            self._raw_counts[split] = 0

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
        t0 = time.perf_counter()
        for seq, comp_idx in zip(image_tokens, component_indices):
            media = self._inventory[int(plan.components.source_ref[int(comp_idx)])]
            row = seq.cpu() if seq.is_cuda else seq
            if int(row[0]) != tokenizer.bos_id or int(row[-1]) != tokenizer.eos_id:
                raise ValueError(
                    f"media {media.media_id[:12]}: encode did not return a "
                    f"BOS..EOS-wrapped block"
                )
            block = np.ascontiguousarray(row[1:-1].numpy(), dtype=TOKEN_DTYPE)
            for split in sorted(self._media_splits.get(media.media_id, ())):
                if media.media_id in self._locations[split]:
                    continue
                self._write_media_to_split(
                    split,
                    media,
                    block,
                    resize_height=resize_height,
                    resize_width=resize_width,
                )

            stats.samples_processed += 1
            stats.image_tokens += int(block.size)
            stats.tokens_generated += int(block.size)
            self._n_media += 1
            self._n_tokens += int(block.size)
        return {"write_ms": (time.perf_counter() - t0) * 1000}

    def _write_media_to_split(
        self,
        split: str,
        media: Any,
        tokens: np.ndarray,
        *,
        resize_height: int,
        resize_width: int,
    ) -> None:
        token_offset = self._token_offsets[split]
        raw_row = self._raw_counts[split]
        self._token_files[split].write(tokens.tobytes())
        self._token_offsets[split] += int(tokens.size)

        width = int(media.width)
        height = int(media.height)
        sample_id, image_index = self._first_occurrence[split][media.media_id]
        self._locations[split][media.media_id] = {
            "media_id": media.media_id,
            "width": width,
            "height": height,
            "resize_height": int(resize_height),
            "resize_width": int(resize_width),
            "token_offset": token_offset,
            "token_length": int(tokens.size),
            "raw_row": raw_row,
        }
        self._raw_buffers[split].append({
            "sample_id": sample_id,
            "image_index": int(image_index),
            "media_id": media.media_id,
            "width": width,
            "height": height,
            "resize_height": int(resize_height),
            "resize_width": int(resize_width),
            "raw_ext": str(media.raw_ext),
            "raw_bytes": media.raw,
        })
        self._raw_counts[split] += 1
        if len(self._raw_buffers[split]) >= self._raw_flush_rows:
            self._flush_raw(split)

    def _flush_raw(self, split: str) -> None:
        rows = self._raw_buffers[split]
        if not rows:
            return
        writer = self._raw_writers.get(split)
        if writer is None:
            writer = pq.ParquetWriter(self._raw_tmp[split], RAW_SCHEMA)
            self._raw_writers[split] = writer
        writer.write_table(pa.Table.from_pylist(rows, schema=RAW_SCHEMA))
        rows.clear()

    def checkpoint(self) -> dict:
        return {}

    def _build_view_rows(self, split: str) -> tuple[list[dict], int]:
        locations = self._locations[split]
        view_rows = []
        dropped = 0
        for row in self._rows_by_split[split]:
            media_ids = _alignment_media_refs(row)
            missing = next((mid for mid in media_ids if mid not in locations), None)
            if missing is not None:
                dropped += 1
                logger.warning(
                    "dropping pair %s: media %s did not encode (undecodable source image)",
                    row.get("prompt_id", ""), missing[:12],
                )
                continue
            images = [dict(locations[mid]) for mid in media_ids]

            prompt = row.get("prompt") or []
            chosen = str(row.get("chosen", ""))
            rejected = str(row.get("rejected", ""))
            text_chars = row.get("text_chars")
            if text_chars is None:
                text_chars = (
                    sum(len(str(m.get("content", ""))) for m in prompt)
                    + len(chosen)
                    + len(rejected)
                )
            view_rows.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "prompt_id": str(row.get("prompt_id", "")),
                "text_chars": int(text_chars),
                "media_tokens_total": sum(int(img["token_length"]) for img in images),
                "images": images,
            })
        if self._rows_by_split[split] and not view_rows:
            raise EncodeIncompleteError(
                f"all {len(self._rows_by_split[split])} {split} pairs dropped — every "
                f"referenced media failed to encode (systematic, not sporadic corruption)"
            )
        return view_rows, dropped

    def finalize(self) -> None:
        files: dict[str, int] = {}
        views: dict[str, list[dict]] = {}
        completed: list[dict] = []
        n_dropped_pairs = 0

        for split in self._rows_by_split:
            self._flush_raw(split)
            raw_writer = self._raw_writers.pop(split, None)
            if raw_writer is None:
                pq.write_table(pa.Table.from_pylist([], schema=RAW_SCHEMA), self._raw_tmp[split])
            else:
                raw_writer.close()

            token_file = self._token_files.pop(split)
            token_file.flush()
            os.fsync(token_file.fileno())
            token_file.close()

            view_rel = f"views/{split}-00000.parquet"
            token_rel = f"tokens/{split}-00000.i32"
            raw_rel = f"raw/{split}-00000.parquet"
            view_final = self._public_dir / view_rel
            view_tmp = view_final.with_suffix(view_final.suffix + ".tmp")
            view_final.parent.mkdir(parents=True, exist_ok=True)

            split_rows, n_dropped = self._build_view_rows(split)
            n_dropped_pairs += n_dropped
            pq.write_table(
                pa.Table.from_pylist(split_rows, schema=VIEW_SCHEMA),
                view_tmp,
            )
            _fsync_file(self._raw_tmp[split])
            _fsync_file(view_tmp)
            os.replace(self._token_tmp[split], self._token_final[split])
            os.replace(self._raw_tmp[split], self._raw_final[split])
            os.replace(view_tmp, view_final)

            for rel in (view_rel, token_rel, raw_rel):
                files[rel] = (self._public_dir / rel).stat().st_size
            spec = {
                "view": view_rel,
                "tokens": token_rel,
                "raw": raw_rel,
                "n_rows": len(split_rows),
                "n_image_refs": self._image_ref_counts[split],
                "n_media": self._raw_counts[split],
                "token_elements": self._token_offsets[split],
                "token_dtype": "<i4",
            }
            views[split] = [spec]
            completed.extend([
                {"name": view_rel, "bytes": files[view_rel], "sequences": 0, "tokens": 0},
                {
                    "name": token_rel,
                    "bytes": files[token_rel],
                    "sequences": self._raw_counts[split],
                    "tokens": self._token_offsets[split],
                },
                {"name": raw_rel, "bytes": files[raw_rel], "sequences": 0, "tokens": 0},
            ])

        if n_dropped_pairs:
            logger.warning(
                "dropped %d pair(s) whose source image failed to decode; published "
                "the rest", n_dropped_pairs,
            )
        self._completed_files = completed
        self.result = {"files": files, "views": views, "n_dropped_pairs": n_dropped_pairs}

    def completed_files(self) -> list:
        return list(self._completed_files)
