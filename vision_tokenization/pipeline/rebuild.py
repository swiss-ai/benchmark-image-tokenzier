"""Offline rebuild: join spilled tokens to TokenizationPlan, validate, assemble.

Reads keyed component payloads from all ranks' spill shards, validates
every planned component exists exactly once (dedup on key+hash collision),
and assembles final sequences in document output order.

Usage::

    python -m vision_tokenization.pipeline.rebuild \\
        --plan /path/to/plan.pt \\
        --spill-dir /path/to/output \\
        --vocab-size 200000
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .spill import COMPONENTS_SCHEMA, ComponentSpillReader
from .assembly import (
    StructureTokenIds,
    assemble_image2text,
    assemble_sequence,
    assemble_text2image,
    replace_image_placeholders,
    split_interleaved_sequence,
)
from ..indexing.planning.tokenization_plan import (
    IMAGE, TEXT,
    TokenizationPlan,
)

logger = logging.getLogger(__name__)


def _validate_spill(
    plan: TokenizationPlan,
    spill_table: pa.Table,
) -> pa.Table:
    """Validate and dedup spilled components against the plan.

    Returns deduplicated spill table (one row per planned component).
    Raises on missing components or conflicting duplicates.
    """
    n_spill = len(spill_table)
    n_plan = plan.total_components

    # Extract spill keys
    spill_doc_ids = spill_table.column("document_id").to_numpy()
    spill_comp_idx = spill_table.column("component_index").to_numpy()
    spill_hashes = spill_table.column("token_hash").to_pylist()

    # Build spill lookup: (doc_id, comp_idx) → list of (row_index, hash)
    spill_lookup: Dict[Tuple[int, int], List[Tuple[int, str]]] = {}
    for i in range(n_spill):
        key = (int(spill_doc_ids[i]), int(spill_comp_idx[i]))
        spill_lookup.setdefault(key, []).append((i, spill_hashes[i]))

    keep_indices = []
    conflicts = []
    for key, entries in spill_lookup.items():
        hashes = set(h for _, h in entries)
        if len(hashes) == 1:
            keep_indices.append(entries[0][0])
        else:
            conflicts.append((key, [h for _, h in entries]))

    if conflicts:
        sample = conflicts[:5]
        raise ValueError(
            f"Rebuild: {len(conflicts)} components have conflicting token hashes "
            f"(non-deterministic tokenization?). First 5: {sample}"
        )

    deduped = spill_table.take(sorted(keep_indices))

    if plan.mode == "interleave":
        spill_doc_ids = deduped.column("document_id").to_numpy()
        spill_kinds = deduped.column("kind").to_numpy()
        known_docs = {int(doc_id) for doc_id in plan.documents.document_id}
        spilled_docs = {int(doc_id) for doc_id in spill_doc_ids}
        orphan_docs = sorted(spilled_docs - known_docs)
        if orphan_docs:
            raise ValueError(
                f"Rebuild: spill contains {len(orphan_docs)} interleave documents not in plan. "
                f"First 10: {orphan_docs[:10]}"
            )

        image_counts: Dict[int, int] = {}
        for i in range(len(deduped)):
            if int(spill_kinds[i]) == int(IMAGE):
                doc_id = int(spill_doc_ids[i])
                image_counts[doc_id] = image_counts.get(doc_id, 0) + 1

        wrong_counts = []
        for doc_idx in range(plan.total_documents):
            doc_id = int(plan.documents.document_id[doc_idx])
            expected = int(plan.documents.num_images[doc_idx])
            actual = image_counts.get(doc_id, 0)
            if actual != expected:
                wrong_counts.append((doc_id, actual, expected))
        if wrong_counts:
            raise ValueError(
                f"Rebuild: {len(wrong_counts)} interleave documents have wrong image "
                f"counts. First 10: {wrong_counts[:10]}"
            )

        logger.info(
            f"Rebuild validation (interleave): {plan.total_documents:,} planned docs, "
            f"{n_spill:,} spilled, {len(deduped):,} kept, {n_spill - len(deduped):,} deduped"
        )
        return deduped

    missing = []
    dedup_lookup = {
        (
            int(deduped.column("document_id")[i].as_py()),
            int(deduped.column("component_index")[i].as_py()),
        ): i
        for i in range(len(deduped))
    }
    keep_plan_indices = []
    for ci in range(n_plan):
        key = (
            int(plan.components.document_id[ci]),
            int(plan.components.component_index[ci]),
        )
        row_idx = dedup_lookup.get(key)
        if row_idx is None:
            missing.append(key)
            continue
        keep_plan_indices.append(row_idx)

    # Check for orphan spill rows (not in plan)
    plan_keys = set(
        (int(plan.components.document_id[i]), int(plan.components.component_index[i]))
        for i in range(n_plan)
    )
    orphan_count = sum(1 for k in spill_lookup if k not in plan_keys)
    if orphan_count > 0:
        logger.warning(
            f"Rebuild: {orphan_count} spill components not in plan (orphans, ignored)"
        )

    if missing:
        n_missing = len(missing)
        sample = missing[:10]
        raise ValueError(
            f"Rebuild: {n_missing} planned components missing from spill. "
            f"First 10: {sample}"
        )

    logger.info(
        f"Rebuild validation: {n_plan:,} planned, {n_spill:,} spilled, "
        f"{len(keep_plan_indices):,} kept, {n_spill - len(deduped):,} deduped"
    )

    return deduped.take(keep_plan_indices)


def _assemble_document(
    mode: str,
    components: List[Tuple[dict, torch.Tensor]],
    token_ids: StructureTokenIds,
    max_sequence_tokens: Optional[int] = None,
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
        text_parts = [t for row, t in components if row["kind"] == int(TEXT)]
        image_parts = [t for row, t in components if row["kind"] == int(IMAGE)]
        if not text_parts:
            raise ValueError("SFT document has no text component")
        text_tokens = text_parts[0]
        image_positions = (text_tokens == token_ids.image_token_id).nonzero(as_tuple=True)[0].tolist()
        return [replace_image_placeholders(text_tokens, image_positions, image_parts)]

    if mode == "interleave":
        segments = []
        text_chunks = []
        image_chunks = []
        for row, tokens in components:
            if row["kind"] == int(TEXT):
                segments.append({"type": "text", "text": True})
                text_chunks.append(tokens)
            elif row["kind"] == int(IMAGE):
                segments.append({"type": "image"})
                image_chunks.append(tokens)

        return split_interleaved_sequence(
            bos_id=token_ids.bos_id,
            eos_id=token_ids.eos_id,
            segments=segments,
            text_token_chunks=text_chunks,
            image_token_chunks=image_chunks,
            max_sequence_tokens=max_sequence_tokens,
        )

    raise ValueError(f"Unknown mode: {mode}")


def rebuild_from_plan(
    plan: TokenizationPlan,
    spill_dir: str | Path,
    token_ids: StructureTokenIds,
    vocab_size: int,
    max_sequence_tokens: Optional[int] = None,
    seqlen_threshold: Optional[int] = None,
    output_name: str = "rebuilt",
) -> Path:
    """Read spill, validate against plan, assemble, write Megatron bin/idx.

    Args:
        plan: TokenizationPlan (source of truth).
        spill_dir: Directory containing rank_XXXX/ spill subdirs.
        token_ids: Special token IDs for assembly.
        vocab_size: Vocab size for optimal dtype selection.
        max_sequence_tokens: Max tokens per sequence (interleave splitting).
        seqlen_threshold: Route sequences to stage2/ or lct/ by length.
        output_name: Output prefix name.

    Returns:
        Path prefix of the output files.
    """
    from vision_tokenization.formats.megatron import DType, IndexedDatasetBuilder

    spill_dir = Path(spill_dir)
    token_dtype = np.int32
    mode = plan.mode

    # Read all spill shards
    logger.info(f"Reading spill shards from {spill_dir}")
    spill_table = ComponentSpillReader.read_all_ranks(spill_dir)
    logger.info(f"Read {len(spill_table):,} spill components from all ranks")

    # Validate + dedup
    spill_table = _validate_spill(plan, spill_table)

    # Extract columns as numpy arrays (vectorized, no Python loops)
    spill_doc_ids = spill_table.column("document_id").to_numpy()
    spill_comp_idx = spill_table.column("component_index").to_numpy()
    spill_kinds = spill_table.column("kind").to_numpy()
    spill_offsets = spill_table.column("token_offset").to_numpy()
    spill_lengths = spill_table.column("token_length").to_numpy()
    spill_rh = spill_table.column("resize_height").to_numpy()
    spill_rw = spill_table.column("resize_width").to_numpy()

    # Sort spill by (doc_id, comp_idx) — vectorized O(N log N)
    sort_order = np.lexsort((spill_comp_idx, spill_doc_ids))
    spill_doc_ids = spill_doc_ids[sort_order]
    spill_comp_idx = spill_comp_idx[sort_order]
    spill_kinds = spill_kinds[sort_order]
    spill_offsets = spill_offsets[sort_order]
    spill_lengths = spill_lengths[sort_order]
    spill_rh = spill_rh[sort_order]
    spill_rw = spill_rw[sort_order]

    # Find document boundaries in sorted spill — vectorized
    n_spill = len(spill_doc_ids)
    if n_spill > 1:
        doc_breaks = np.where(np.diff(spill_doc_ids) != 0)[0] + 1
        doc_starts = np.concatenate([[0], doc_breaks])
        doc_ends = np.concatenate([doc_breaks, [n_spill]])
    elif n_spill == 1:
        doc_starts = np.array([0])
        doc_ends = np.array([1])
    else:
        doc_starts = np.array([], dtype=np.int64)
        doc_ends = np.array([], dtype=np.int64)

    # Build doc_id → (start, end) index into sorted spill — vectorized
    spill_unique_docs = spill_doc_ids[doc_starts] if len(doc_starts) > 0 else np.array([], dtype=np.int64)

    # Read provenance (rank_dir, shard_id) per spill row in one pass.
    # Build arrays instead of a dict — one entry per sorted spill row.
    rank_dirs = sorted(spill_dir.glob("rank_*"))
    token_mmaps: Dict[Tuple[str, int], np.ndarray] = {}
    # Provenance arrays: rank_dir_idx and shard_id per spill row
    prov_rank_idx = np.full(n_spill, -1, dtype=np.int32)
    prov_shard_id = np.full(n_spill, -1, dtype=np.int32)
    prov_offset = np.zeros(n_spill, dtype=np.int64)
    prov_length = np.zeros(n_spill, dtype=np.int64)
    rank_dir_list: List[Path] = []
    # Pre-compute sorted compound key once for provenance matching
    spill_key = spill_doc_ids.astype(np.int64) * 1_000_000 + spill_comp_idx.astype(np.int64)

    for rd_idx, rd in enumerate(rank_dirs):
        if not (rd / "_SUCCESS").exists():
            continue
        rank_dir_list.append(rd)
        actual_rd_idx = len(rank_dir_list) - 1
        for tf in rd.glob("tokens.*.bin"):
            sid = int(tf.stem.split(".")[-1])
            if tf.stat().st_size > 0:
                token_mmaps[(actual_rd_idx, sid)] = np.memmap(str(tf), dtype=np.uint8, mode="r")

        for sf in sorted(rd.glob("components.*.parquet")):
            sid = int(sf.stem.split(".")[-1])
            ct = pq.read_table(sf)
            ct_doc = ct.column("document_id").to_numpy()
            ct_comp = ct.column("component_index").to_numpy()
            ct_off = ct.column("token_offset").to_numpy()
            ct_len = ct.column("token_length").to_numpy()
            ct_key = ct_doc.astype(np.int64) * 1_000_000 + ct_comp.astype(np.int64)
            positions = np.searchsorted(spill_key, ct_key)
            valid = (positions < n_spill) & (spill_key[np.minimum(positions, n_spill - 1)] == ct_key)
            # Vectorized provenance assignment (first occurrence wins)
            valid_idx = np.where(valid)[0]
            valid_pos = positions[valid_idx]
            unset = prov_rank_idx[valid_pos] < 0
            assign = valid_idx[unset]
            assign_pos = valid_pos[unset]
            prov_rank_idx[assign_pos] = actual_rd_idx
            prov_shard_id[assign_pos] = sid
            prov_offset[assign_pos] = ct_off[assign]
            prov_length[assign_pos] = ct_len[assign]

    _dtype = np.dtype(token_dtype)
    _itemsize = _dtype.itemsize

    def _load_tokens(row_idx: int) -> np.ndarray:
        ri = int(prov_rank_idx[row_idx])
        si = int(prov_shard_id[row_idx])
        off = int(prov_offset[row_idx])
        length = int(prov_length[row_idx])
        buf = token_mmaps.get((ri, si))
        if buf is not None:
            return np.frombuffer(buf[off:off + length * _itemsize], dtype=_dtype).copy()
        return ComponentSpillReader.load_tokens(rank_dir_list[ri], si, off, length, token_dtype=_dtype)

    # Output
    megatron_dtype = DType.optimal_dtype(vocab_size)
    output_prefix = spill_dir / output_name
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    builder = IndexedDatasetBuilder(str(output_prefix) + ".bin", dtype=megatron_dtype)

    # Assemble documents in plan output order
    n_docs = plan.total_documents
    doc_order = np.argsort(plan.documents.output_order)

    total_sequences = 0
    total_tokens_out = 0

    for di in range(n_docs):
        doc_idx = int(doc_order[di])
        doc_id = int(plan.documents.document_id[doc_idx])

        # Find this doc's components in sorted spill via searchsorted
        pos = np.searchsorted(spill_unique_docs, doc_id)
        if pos >= len(spill_unique_docs) or int(spill_unique_docs[pos]) != doc_id:
            continue
        cs = int(doc_starts[pos])
        ce = int(doc_ends[pos])

        # Components are already sorted by (doc_id, comp_idx)
        components = []
        for ri in range(cs, ce):
            if prov_rank_idx[ri] < 0:
                continue
            tokens_np = _load_tokens(ri)
            tokens = torch.from_numpy(tokens_np).long()
            row = {
                "kind": int(spill_kinds[ri]),
                "component_index": int(spill_comp_idx[ri]),
                "resize_height": int(spill_rh[ri]),
                "resize_width": int(spill_rw[ri]),
            }
            components.append((row, tokens))

        if not components:
            continue

        sequences = _assemble_document(mode, components, token_ids, max_sequence_tokens)

        for seq in sequences:
            builder.add_item(seq.numpy().astype(megatron_dtype))
            builder.end_document()
            total_sequences += 1
            total_tokens_out += len(seq)

        if (di + 1) % 100_000 == 0:
            logger.info(f"Rebuild progress: {di + 1:,}/{n_docs:,} documents")

    idx_path = str(output_prefix) + ".idx"
    builder.finalize(idx_path)

    logger.info(
        f"Rebuild complete: {total_sequences:,} sequences, "
        f"{total_tokens_out:,} tokens -> {output_prefix}"
    )
    return output_prefix
