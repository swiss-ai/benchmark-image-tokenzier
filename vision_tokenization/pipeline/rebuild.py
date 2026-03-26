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

    # Build component lookup: (doc_id, comp_idx) → spill row
    spill_doc_ids = spill_table.column("document_id").to_numpy()
    spill_comp_idx = spill_table.column("component_index").to_numpy()
    spill_kinds = spill_table.column("kind").to_numpy()
    spill_offsets = spill_table.column("token_offset").to_numpy()
    spill_lengths = spill_table.column("token_length").to_numpy()
    spill_rh = spill_table.column("resize_height").to_numpy()
    spill_rw = spill_table.column("resize_width").to_numpy()

    comp_lookup: Dict[Tuple[int, int], int] = {}
    for i in range(len(spill_table)):
        comp_lookup[(int(spill_doc_ids[i]), int(spill_comp_idx[i]))] = i

    spill_comp_by_doc: Dict[int, List[Tuple[int, int]]] = {}
    if mode == "interleave":
        for i in range(len(spill_table)):
            doc_id = int(spill_doc_ids[i])
            spill_comp_by_doc.setdefault(doc_id, []).append(
                (int(spill_comp_idx[i]), i)
            )

    # TODO: mmap token files for zero-copy reads
    # For now, load per-component from file (functional but not optimal at 90M scale)

    # Determine output dtype
    megatron_dtype = DType.optimal_dtype(vocab_size)

    # Output paths — megatron expects {prefix}.bin and {prefix}.idx
    output_prefix = spill_dir / output_name
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    bin_path = str(output_prefix) + ".bin"

    builder = IndexedDatasetBuilder(bin_path, dtype=megatron_dtype)

    # Pre-build plan component index for modes with fully known components.
    plan_comp_by_doc: Dict[int, List[Tuple[int, int]]] = {}
    if plan.mode != "interleave":
        plan_doc_ids = plan.components.document_id
        plan_comp_idxs = plan.components.component_index
        for i in range(plan.total_components):
            doc_id = int(plan_doc_ids[i])
            plan_comp_by_doc.setdefault(doc_id, []).append(
                (int(plan_comp_idxs[i]), i)
            )

    # Memory-map all token files for zero-copy reads
    rank_dirs = sorted(spill_dir.glob("rank_*"))
    token_mmaps: Dict[Tuple[str, int], np.ndarray] = {}
    for rd in rank_dirs:
        for tf in rd.glob("tokens.*.bin"):
            shard_id = int(tf.stem.split(".")[-1])
            if tf.stat().st_size > 0:
                token_mmaps[(str(rd), shard_id)] = np.memmap(str(tf), dtype=np.uint8, mode="r")

    # We need to know which rank/shard each spill row came from.
    # Re-read spill per rank to track provenance.
    spill_provenance: Dict[Tuple[int, int], Tuple[Path, int, int, int]] = {}
    for rd in rank_dirs:
        if not (rd / "_SUCCESS").exists():
            continue
        for sf in sorted(rd.glob("components.*.parquet")):
            shard_id = int(sf.stem.split(".")[-1])
            ct = pq.read_table(sf)
            doc_ids_arr = ct.column("document_id").to_numpy()
            comp_idx_arr = ct.column("component_index").to_numpy()
            offsets_arr = ct.column("token_offset").to_numpy()
            lengths_arr = ct.column("token_length").to_numpy()
            for i in range(len(ct)):
                key = (int(doc_ids_arr[i]), int(comp_idx_arr[i]))
                if key not in spill_provenance:
                    spill_provenance[key] = (rd, shard_id, int(offsets_arr[i]), int(lengths_arr[i]))

    _dtype = np.dtype(token_dtype)
    _itemsize = _dtype.itemsize

    def _load_tokens_mmap(rd: Path, shard_id: int, offset: int, length: int) -> np.ndarray:
        buf = token_mmaps.get((str(rd), shard_id))
        if buf is not None:
            return np.frombuffer(buf[offset:offset + length * _itemsize], dtype=_dtype).copy()
        return ComponentSpillReader.load_tokens(rd, shard_id, offset, length, token_dtype=_dtype)

    # Assemble documents in output order
    n_docs = plan.total_documents
    doc_order = np.argsort(plan.documents.output_order)

    total_sequences = 0
    total_tokens_out = 0

    for di in range(n_docs):
        doc_idx = int(doc_order[di])
        doc_id = int(plan.documents.document_id[doc_idx])

        components = []
        if mode == "interleave":
            iter_rows = spill_comp_by_doc.get(doc_id, [])
            iter_rows.sort(key=lambda x: x[0])
        else:
            plan_comps = plan_comp_by_doc.get(doc_id, [])
            plan_comps.sort(key=lambda x: x[0])  # sort by component_index
            iter_rows = [
                (comp_idx, comp_lookup.get((doc_id, comp_idx)))
                for comp_idx, _plan_row in plan_comps
            ]

        for comp_idx, spill_row in iter_rows:
            spill_key = (doc_id, comp_idx)
            if spill_row is None:
                continue

            prov = spill_provenance.get(spill_key)
            if prov is None:
                continue

            rd, shard_id, byte_offset, tok_length = prov
            tokens_np = _load_tokens_mmap(rd, shard_id, byte_offset, tok_length)
            tokens = torch.from_numpy(tokens_np).long()

            row = {
                "kind": int(spill_kinds[spill_row]),
                "component_index": comp_idx,
                "resize_height": int(spill_rh[spill_row]),
                "resize_width": int(spill_rw[spill_row]),
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
