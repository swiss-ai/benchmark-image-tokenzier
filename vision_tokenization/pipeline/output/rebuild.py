"""Offline rebuild: join spilled tokens to TokenizationPlan, validate, assemble.

``rebuild_rank`` (called by the executor after spill finalization) reads one
rank's keyed component payloads, joins them back to the plan by
``(document_id, component_index)`` — duplicate keys resolve first-wins via
provenance assignment — and assembles final sequences in document output
order.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .spill import ComponentSpillReader
from ...common.assembly import (
    StructureTokenIds,
    assemble_image2text,
    assemble_sequence,
    assemble_sft_sequence,
    assemble_text2image,
    split_interleaved_sequence,
)
from ...indexing.planning.tokenization_plan import (
    IMAGE, TEXT,
    TokenizationPlan,
)

logger = logging.getLogger(__name__)

_PROVENANCE_KEY_SHIFT = 32  # compound key = (doc_id << 32) | component_index


def finalize_builders(builders: dict, prefixes: dict) -> None:
    """Finalize builders; drop empty shards so Megatron mmap won't crash."""
    for key, builder in builders.items():
        bin_path = str(prefixes[key]) + ".bin"
        idx_path = str(prefixes[key]) + ".idx"
        builder.finalize(idx_path)
        if os.path.exists(bin_path) and os.path.getsize(bin_path) == 0:
            os.unlink(bin_path)
            if os.path.exists(idx_path):
                os.unlink(idx_path)


def _split_components_by_kind(
    components: List[Tuple[dict, torch.Tensor]],
) -> Tuple[List[dict], List[torch.Tensor], List[torch.Tensor]]:
    """Convert spilled components into structured segments + text/image chunks."""
    segments: List[dict] = []
    text_chunks: List[torch.Tensor] = []
    image_chunks: List[torch.Tensor] = []
    for row, tokens in components:
        if row["kind"] == int(TEXT):
            segments.append({"type": "text", "text": True})
            text_chunks.append(tokens)
        elif row["kind"] == int(IMAGE):
            segments.append({"type": "image"})
            image_chunks.append(tokens)
    return segments, text_chunks, image_chunks


def _validate_sft_rebuild(
    text_chunks: List[torch.Tensor],
    image_chunks: List[torch.Tensor],
    expected_num_images: Optional[int],
) -> Optional[str]:
    """Return a failure reason if the SFT doc is incomplete, or None if valid."""
    if not text_chunks:
        return "no text components"
    if expected_num_images is not None and len(image_chunks) != expected_num_images:
        return f"expected {expected_num_images} images but found {len(image_chunks)}"
    return None


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------

def _assemble_document(
    mode: str,
    components: List[Tuple[dict, torch.Tensor]],
    token_ids: StructureTokenIds,
    max_sequence_tokens: Optional[int] = None,
    expected_num_images: Optional[int] = None,
) -> List[torch.Tensor]:
    """Assemble final sequence(s) for one document."""
    if mode == "image_only":
        image_structs = [t for row, t in components if row["kind"] == int(IMAGE)]
        return [assemble_sequence(
            bos_id=token_ids.bos_id,
            eos_id=token_ids.eos_id,
            component_tokens=image_structs,
        )]

    if mode == "image2text":
        image_structs = [t for row, t in components if row["kind"] == int(IMAGE)]
        text_parts = [t for row, t in components if row["kind"] == int(TEXT)]
        text_tokens = text_parts[0] if text_parts else torch.tensor([], dtype=torch.long)
        return [assemble_image2text(
            bos_id=token_ids.bos_id,
            eos_id=token_ids.eos_id,
            image_structures=image_structs,
            text_tokens=text_tokens,
        )]

    if mode == "text2image":
        image_structs = [t for row, t in components if row["kind"] == int(IMAGE)]
        text_parts = [t for row, t in components if row["kind"] == int(TEXT)]
        text_tokens = text_parts[0] if text_parts else torch.tensor([], dtype=torch.long)
        return [assemble_text2image(
            bos_id=token_ids.bos_id,
            eos_id=token_ids.eos_id,
            text_tokens=text_tokens,
            image_structures=image_structs,
        )]

    if mode == "sft":
        segments, text_chunks, image_chunks = _split_components_by_kind(components)
        reason = _validate_sft_rebuild(text_chunks, image_chunks, expected_num_images)
        if reason is not None:
            logger.warning("Skipping SFT document during rebuild: %s", reason)
            return []
        return [assemble_sft_sequence(
            bos_id=token_ids.bos_id,
            eos_id=token_ids.eos_id,
            segments=segments,
            text_token_chunks=text_chunks,
            image_token_chunks=image_chunks,
        )]

    if mode == "interleave":
        segments, text_chunks, image_chunks = _split_components_by_kind(components)
        return split_interleaved_sequence(
            bos_id=token_ids.bos_id,
            eos_id=token_ids.eos_id,
            segments=segments,
            text_token_chunks=text_chunks,
            image_token_chunks=image_chunks,
            max_sequence_tokens=max_sequence_tokens,
        )

    raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Shared spill → sorted arrays + provenance + token loader
# ---------------------------------------------------------------------------

def _extract_sorted_spill(spill_table: pa.Table):
    """Extract spill columns as sorted numpy arrays and compute doc boundaries.

    Returns a dict with sorted arrays, doc boundary arrays, and unique doc ids.
    """
    doc_ids = spill_table.column("document_id").to_numpy()
    comp_idx = spill_table.column("component_index").to_numpy()
    kinds = spill_table.column("kind").to_numpy()
    offsets = spill_table.column("token_offset").to_numpy()
    lengths = spill_table.column("token_length").to_numpy()
    rh = spill_table.column("resize_height").to_numpy()
    rw = spill_table.column("resize_width").to_numpy()

    order = np.lexsort((comp_idx, doc_ids))
    doc_ids = doc_ids[order]
    comp_idx = comp_idx[order]
    kinds = kinds[order]
    offsets = offsets[order]
    lengths = lengths[order]
    rh = rh[order]
    rw = rw[order]

    n = len(doc_ids)
    if n > 1:
        breaks = np.where(np.diff(doc_ids) != 0)[0] + 1
        doc_starts = np.concatenate([[0], breaks])
        doc_ends = np.concatenate([breaks, [n]])
    elif n == 1:
        doc_starts = np.array([0])
        doc_ends = np.array([1])
    else:
        doc_starts = np.array([], dtype=np.int64)
        doc_ends = np.array([], dtype=np.int64)

    unique_docs = doc_ids[doc_starts] if len(doc_starts) > 0 else np.array([], dtype=np.int64)

    return {
        "doc_ids": doc_ids, "comp_idx": comp_idx, "kinds": kinds,
        "offsets": offsets, "lengths": lengths, "rh": rh, "rw": rw,
        "doc_starts": doc_starts, "doc_ends": doc_ends,
        "unique_docs": unique_docs, "n": n,
    }


def _build_provenance_single_rank(rank_dir: Path, spill_key: np.ndarray, n_spill: int):
    """Build per-row provenance arrays from one rank's shard component tables.

    Returns (prov_shard_id, prov_offset, prov_length, token_mmaps).
    """
    prov_shard_id = np.full(n_spill, -1, dtype=np.int32)
    prov_offset = np.zeros(n_spill, dtype=np.int64)
    prov_length = np.zeros(n_spill, dtype=np.int64)

    token_mmaps: Dict[int, np.ndarray] = {}
    for tf in rank_dir.glob("tokens.*.bin"):
        sid = int(tf.stem.split(".")[-1])
        if tf.stat().st_size > 0:
            token_mmaps[sid] = np.memmap(str(tf), dtype=np.uint8, mode="r")

    for sf in sorted(rank_dir.glob("components.*.parquet")):
        sid = int(sf.stem.split(".")[-1])
        ct = pq.read_table(sf)
        ct_doc = ct.column("document_id").to_numpy()
        ct_comp = ct.column("component_index").to_numpy()
        ct_off = ct.column("token_offset").to_numpy()
        ct_len = ct.column("token_length").to_numpy()
        ct_key = ct_doc.astype(np.int64) * (1 << _PROVENANCE_KEY_SHIFT) + ct_comp.astype(np.int64)
        positions = np.searchsorted(spill_key, ct_key)
        valid = (positions < n_spill) & (spill_key[np.minimum(positions, n_spill - 1)] == ct_key)
        valid_idx = np.where(valid)[0]
        valid_pos = positions[valid_idx]
        unset = prov_shard_id[valid_pos] < 0
        assign = valid_idx[unset]
        assign_pos = valid_pos[unset]
        prov_shard_id[assign_pos] = sid
        prov_offset[assign_pos] = ct_off[assign]
        prov_length[assign_pos] = ct_len[assign]

    return prov_shard_id, prov_offset, prov_length, token_mmaps


def _make_token_loader_single_rank(
    prov_shard_id, prov_offset, prov_length, token_mmaps, rank_dir,
):
    """Create a token loader closure for a single rank's provenance."""
    _dtype = np.dtype(np.int32)
    _itemsize = _dtype.itemsize

    def load(row_idx: int) -> np.ndarray:
        si = int(prov_shard_id[row_idx])
        off = int(prov_offset[row_idx])
        length = int(prov_length[row_idx])
        buf = token_mmaps.get(si)
        if buf is not None:
            return np.frombuffer(buf[off:off + length * _itemsize], dtype=_dtype).copy()
        return ComponentSpillReader.load_tokens(rank_dir, si, off, length, token_dtype=_dtype)

    return load


# ---------------------------------------------------------------------------
# Shared document assembly loop
# ---------------------------------------------------------------------------

def _assemble_and_write(
    *,
    doc_ids_to_process: np.ndarray,
    spill,
    load_tokens,
    prov_check_field,
    mode: str,
    token_ids: StructureTokenIds,
    max_sequence_tokens: Optional[int],
    expected_num_images_to_process: Optional[np.ndarray],
    megatron_dtype,
    builders: dict,
    log_prefix: str = "",
    reject_doc_ids: Optional[set] = None,
) -> Dict:
    """Shared assembly loop: iterate documents, assemble, route to builders.

    Args:
        doc_ids_to_process: Ordered array of doc_ids to process.
        spill: Dict from _extract_sorted_spill.
        load_tokens: Callable(row_idx) -> np.ndarray.
        prov_check_field: Array to check for valid provenance (>= 0).
        mode: Tokenization mode.
        token_ids: Special token IDs.
        max_sequence_tokens: Max tokens per sequence.
        megatron_dtype: Numpy dtype for output.
        builders: Dict with key 'main'.
        log_prefix: Prefix for log messages.
        reject_doc_ids: Optional set of document IDs to skip.

    Returns:
        Stats dict.
    """
    unique_docs = spill["unique_docs"]
    doc_starts = spill["doc_starts"]
    doc_ends = spill["doc_ends"]
    spill_kinds = spill["kinds"]
    spill_comp_idx = spill["comp_idx"]
    spill_rh = spill["rh"]
    spill_rw = spill["rw"]

    total_sequences = 0
    total_tokens_out = 0

    n_processed = 0
    n_rejected = 0
    for doc_pos, doc_id in enumerate(doc_ids_to_process):
        doc_id = int(doc_id)
        if reject_doc_ids is not None and doc_id in reject_doc_ids:
            n_rejected += 1
            continue
        pos = np.searchsorted(unique_docs, doc_id)
        if pos >= len(unique_docs) or int(unique_docs[pos]) != doc_id:
            continue
        cs = int(doc_starts[pos])
        ce = int(doc_ends[pos])

        components = []
        for ri in range(cs, ce):
            if prov_check_field[ri] < 0:
                continue
            tokens_np = load_tokens(ri)
            tokens = torch.from_numpy(tokens_np).long()
            components.append(({
                "kind": int(spill_kinds[ri]),
                "component_index": int(spill_comp_idx[ri]),
                "resize_height": int(spill_rh[ri]),
                "resize_width": int(spill_rw[ri]),
            }, tokens))

        if not components:
            continue

        expected_num_images = None
        if expected_num_images_to_process is not None:
            expected_num_images = int(expected_num_images_to_process[doc_pos])

        sequences = _assemble_document(
            mode,
            components,
            token_ids,
            max_sequence_tokens,
            expected_num_images=expected_num_images,
        )
        if not sequences:
            n_rejected += 1
            continue

        for seq in sequences:
            seq_np = seq.numpy().astype(megatron_dtype)
            seq_len = len(seq)

            builders["main"].add_item(seq_np)
            builders["main"].end_document()
            total_sequences += 1
            total_tokens_out += seq_len

        n_processed += 1
        if log_prefix and n_processed % 100_000 == 0:
            logger.info(f"{log_prefix}Rebuild progress: {n_processed:,} documents")

    return {
        "sequences": total_sequences,
        "tokens": total_tokens_out,
        "rejected_documents": n_rejected,
    }


# ---------------------------------------------------------------------------
# Per-rank rebuild (called from executor after spill)
# ---------------------------------------------------------------------------

def rebuild_rank(
    plan: TokenizationPlan,
    rank: int,
    spill_dir: str | Path,
    token_ids: StructureTokenIds,
    vocab_size: int,
    *,
    output_dir: Optional[str | Path] = None,
    max_sequence_tokens: Optional[int] = None,
    reject_doc_ids: Optional[set] = None,
) -> Dict:
    """Per-rank rebuild: read ``spill_dir/rank_NNNN/``, assemble documents,
    write one flat ``output_dir/rank_NNNN_chunk_0000.{bin,idx}`` stream —
    atomically (tmp + fsync + rename), banding at merge time. ``output_dir``
    defaults to ``spill_dir``. Returns stats incl. ``files``: the shard
    records this rank ships, for the completion manifest.
    """
    from vision_tokenization.formats.megatron import DType
    from ..runtime.checkpoint import finalize_shard_writer, open_chunk_writer

    spill_dir = Path(spill_dir)
    output_dir = Path(output_dir) if output_dir is not None else spill_dir
    rank_dir = spill_dir / f"rank_{rank:04d}"
    mode = plan.mode

    if not (rank_dir / "_SUCCESS").exists():
        logger.warning(f"[rank {rank}] Skipping rebuild: no _SUCCESS in {rank_dir}")
        return {"rank": rank, "sequences": 0, "tokens": 0, "files": []}

    spill_table = ComponentSpillReader.read_rank(rank_dir)
    if len(spill_table) == 0:
        logger.info(f"[rank {rank}] No components to rebuild")
        return {"rank": rank, "sequences": 0, "tokens": 0, "files": []}

    logger.info(f"[rank {rank}] Rebuilding {len(spill_table):,} spilled components")

    spill = _extract_sorted_spill(spill_table)
    spill_key = spill["doc_ids"].astype(np.int64) * (1 << _PROVENANCE_KEY_SHIFT) + spill["comp_idx"].astype(np.int64)
    prov_shard_id, prov_offset, prov_length, token_mmaps = _build_provenance_single_rank(
        rank_dir, spill_key, spill["n"],
    )
    load_tokens = _make_token_loader_single_rank(
        prov_shard_id, prov_offset, prov_length, token_mmaps, rank_dir,
    )

    # Build ordered doc_ids: only iterate docs in this rank's spill,
    # sorted by plan output order.
    plan_doc_order = np.argsort(plan.documents.output_order)
    plan_doc_ids_ordered = plan.documents.document_id[plan_doc_order]
    rank_doc_set = set(spill["unique_docs"].tolist())
    ordered_num_images = plan.documents.num_images[plan_doc_order]
    selected_docs = [
        (int(doc_id), int(num_images))
        for doc_id, num_images in zip(plan_doc_ids_ordered, ordered_num_images)
        if int(doc_id) in rank_doc_set
    ]
    doc_ids_to_process = np.array([doc_id for doc_id, _ in selected_docs], dtype=np.int64)
    expected_num_images_to_process = np.array(
        [num_images for _, num_images in selected_docs],
        dtype=np.int64,
    )

    builder, tmp_bin, tmp_idx, bin_path, idx_path = open_chunk_writer(
        str(output_dir), rank, 0, vocab_size,
    )
    megatron_dtype = DType.optimal_dtype(vocab_size)

    stats = _assemble_and_write(
        doc_ids_to_process=doc_ids_to_process,
        spill=spill,
        load_tokens=load_tokens,
        prov_check_field=prov_shard_id,
        mode=mode,
        token_ids=token_ids,
        max_sequence_tokens=max_sequence_tokens,
        expected_num_images_to_process=expected_num_images_to_process,
        megatron_dtype=megatron_dtype,
        builders={"main": builder},
        reject_doc_ids=reject_doc_ids,
    )

    stats["files"] = []
    if stats["sequences"] > 0:
        finalize_shard_writer(builder, tmp_bin, tmp_idx, bin_path, idx_path)
        stats["files"].append({
            "name": os.path.basename(bin_path),
            "bytes": os.path.getsize(bin_path),
            "sequences": stats["sequences"],
            "tokens": stats["tokens"],
        })
    else:
        # No sequences survived: leave nothing behind (empty shards crash
        # Megatron mmap), not even the tmp files.
        builder.finalize(tmp_idx)
        os.unlink(tmp_bin)
        os.unlink(tmp_idx)

    logger.info(
        f"[rank {rank}] Rebuild: {stats['sequences']:,} seqs, "
        f"{stats['tokens']:,} tokens"
    )

    stats["rank"] = rank
    return stats
