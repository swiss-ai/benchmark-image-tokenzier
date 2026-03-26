"""Offline rebuild: read spill shards, assemble final sequences, write bin/idx.

Reads the SHAR-like spill format (documents.parquet + components.parquet +
tokens.bin) produced by the online tokenization stage.  Assembles final
training sequences per document using the extracted assembly helpers, applies
the boundary-preserving interleave split policy, and writes Megatron bin/idx.

No cross-document packing in v1 — each document produces one or more
sequences independently.

Usage::

    python -m vision_tokenization.pipeline.pooled.rebuild \\
        /path/to/output --max-seq-len 32768 --vocab-size 200000
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch

from .spill import (
    COMPONENTS_SCHEMA,
    DOCUMENTS_SCHEMA,
    SpillReader,
)
from ..assembly import (
    StructureTokenIds,
    assemble_image2text,
    assemble_interleaved_sequence,
    assemble_sequence,
    assemble_text2image,
    replace_image_placeholders,
    split_interleaved_sequence,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core rebuild logic
# ---------------------------------------------------------------------------

def _load_document_components(
    worker_dir: Path,
    shard_id: int,
    comp_rows: List[dict],
    token_dtype: np.dtype,
) -> List[Tuple[dict, np.ndarray]]:
    """Load component metadata + token arrays for one document from one shard.

    Returns list of (comp_dict, token_array) sorted by component_index.
    """
    result = []
    for row in comp_rows:
        tokens = SpillReader.load_component_tokens(
            worker_dir,
            shard_id=shard_id,
            token_offset=row["token_offset"],
            token_length=row["token_length"],
            token_dtype=token_dtype,
        )
        result.append((row, tokens))
    result.sort(key=lambda x: x[0]["component_index"])
    return result


def assemble_document_sequences(
    mode: str,
    components: List[Tuple[dict, np.ndarray]],
    token_ids: StructureTokenIds,
    max_sequence_tokens: Optional[int] = None,
) -> List[torch.Tensor]:
    """Assemble final training sequence(s) for one document.

    Args:
        mode: Document mode (image_only, image2text, text2image, sft, interleave).
        components: Sorted list of (comp_dict, token_array) tuples.
        token_ids: Special token IDs for assembly.
        max_sequence_tokens: Max tokens per output sequence (interleave splitting).

    Returns:
        List of assembled torch tensors (one per output sequence).
    """
    # Convert numpy arrays to torch tensors
    comp_tensors = [(row, torch.from_numpy(arr).long()) for row, arr in components]

    if mode == "image_only":
        # All image components in the group become one sequence:
        # BOS + img_struct_0 + img_struct_1 + ... + EOS
        # This preserves group semantics — one sequence per logical document.
        image_structs = [tokens for row, tokens in comp_tensors if row["kind"] == "image"]
        return [assemble_sequence(
            bos_id=token_ids.bos_id,
            eos_id=token_ids.eos_id,
            component_tokens=image_structs,
        )]

    if mode == "image2text":
        image_structs = [tokens for row, tokens in comp_tensors if row["kind"] == "image"]
        text_parts = [tokens for row, tokens in comp_tensors if row["kind"] == "text"]
        text_tokens = text_parts[0] if text_parts else torch.tensor([], dtype=torch.long)
        return [assemble_image2text(
            bos_id=token_ids.bos_id,
            eos_id=token_ids.eos_id,
            image_structures=image_structs,
            text_tokens=text_tokens,
        )]

    if mode == "text2image":
        image_structs = [tokens for row, tokens in comp_tensors if row["kind"] == "image"]
        text_parts = [tokens for row, tokens in comp_tensors if row["kind"] == "text"]
        text_tokens = text_parts[0] if text_parts else torch.tensor([], dtype=torch.long)
        return [assemble_text2image(
            bos_id=token_ids.bos_id,
            eos_id=token_ids.eos_id,
            text_tokens=text_tokens,
            image_structures=image_structs,
        )]

    if mode == "sft":
        # SFT: text with <|image|> placeholders replaced by image structures
        text_parts = [tokens for row, tokens in comp_tensors if row["kind"] == "text"]
        image_parts = [tokens for row, tokens in comp_tensors if row["kind"] == "image"]
        if not text_parts:
            raise ValueError("SFT document has no text component")
        text_tokens = text_parts[0]
        # Find placeholder positions
        image_positions = (text_tokens == token_ids.image_token_id).nonzero(as_tuple=True)[0].tolist()
        result = replace_image_placeholders(text_tokens, image_positions, image_parts)
        # Wrap with BOS/EOS (the text already has BOS/EOS from chat template)
        return [result]

    if mode == "interleave":
        # Rebuild segments list from ordered components
        segments = []
        text_chunks = []
        image_chunks = []
        for row, tokens in comp_tensors:
            if row["kind"] == "text":
                segments.append({"type": "text", "text": True})  # non-empty marker
                text_chunks.append(tokens)
            elif row["kind"] == "image":
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


# ---------------------------------------------------------------------------
# Full rebuild pipeline
# ---------------------------------------------------------------------------

def rebuild(
    output_dir: str | Path,
    token_ids: StructureTokenIds,
    vocab_size: int,
    max_sequence_tokens: Optional[int] = None,
    seqlen_threshold: Optional[int] = None,
    output_name: str = "rebuilt",
) -> Path:
    """Read spill shards and write final Megatron bin/idx.

    Args:
        output_dir: Directory containing worker_XX/ spill subdirs.
        token_ids: Special token IDs for assembly.
        vocab_size: Vocab size for optimal dtype selection.
        max_sequence_tokens: Max tokens per sequence (interleave splitting).
        seqlen_threshold: If set, route sequences to stage2/ or lct/.
        output_name: Output prefix name.

    Returns:
        Path prefix of the output files.
    """
    from vision_tokenization.formats.megatron import (
        DType,
        IndexedDatasetBuilder,
    )

    output_dir = Path(output_dir)
    token_dtype = np.int32
    # Single-pass read: process each worker once, build component lookup directly
    output_dir_p = output_dir
    worker_dirs = sorted(output_dir_p.glob("worker_*"))
    if not worker_dirs:
        logger.warning("No worker directories found in %s", output_dir_p)
        return output_dir_p / output_name

    all_doc_tables = []
    comp_by_doc: Dict[int, List[Tuple[Path, int, dict]]] = defaultdict(list)
    # Cache open mmapped token files to avoid per-component open/close
    _token_mmaps: Dict[Tuple[str, int], np.ndarray] = {}

    for worker_dir in worker_dirs:
        if not (worker_dir / "_SUCCESS").exists():
            logger.warning(f"Skipping incomplete worker: {worker_dir}")
            continue

        # Read shard files individually to track shard provenance
        shard_files = sorted(worker_dir.glob("components.*.parquet"))
        for sf in shard_files:
            shard_id = int(sf.stem.split(".")[-1])
            comp_table = pq.read_table(sf)
            doc_ids = comp_table.column("document_id").to_pylist()
            comp_idxs = comp_table.column("component_index").to_pylist()
            kinds = comp_table.column("kind").to_pylist()
            offsets = comp_table.column("token_offset").to_pylist()
            lengths = comp_table.column("token_length").to_pylist()
            rh = comp_table.column("resize_height").to_pylist()
            rw = comp_table.column("resize_width").to_pylist()
            manifest_rows = comp_table.column("manifest_row").to_pylist()

            for i in range(len(doc_ids)):
                row = {
                    "document_id": doc_ids[i],
                    "component_index": comp_idxs[i],
                    "kind": kinds[i],
                    "token_offset": offsets[i],
                    "token_length": lengths[i],
                    "resize_height": rh[i],
                    "resize_width": rw[i],
                    "manifest_row": manifest_rows[i],
                }
                comp_by_doc[doc_ids[i]].append((worker_dir, shard_id, row))

            # Memory-map token file for zero-copy reads
            token_path = worker_dir / f"tokens.{shard_id:06d}.bin"
            if token_path.exists() and token_path.stat().st_size > 0:
                _token_mmaps[(str(worker_dir), shard_id)] = np.memmap(
                    str(token_path), dtype=np.uint8, mode="r",
                )

        # Read document metadata
        doc_files = sorted(worker_dir.glob("documents.*.parquet"))
        for df in doc_files:
            all_doc_tables.append(pq.read_table(df))

    if not all_doc_tables:
        logger.warning("No documents found in spill shards")
        return output_dir_p / output_name

    import pyarrow as pa
    docs_table = pa.concat_tables(all_doc_tables)

    logger.info(
        f"Rebuild: {len(docs_table):,} documents, "
        f"{sum(len(v) for v in comp_by_doc.values()):,} components "
        f"from {len(worker_dirs)} workers"
    )

    _dtype = np.dtype(token_dtype)
    _itemsize = _dtype.itemsize

    def _load_tokens_fast(worker_dir: Path, shard_id: int, offset: int, length: int) -> np.ndarray:
        """Load tokens from mmap (zero-copy), falling back to file read."""
        buf = _token_mmaps.get((str(worker_dir), shard_id))
        if buf is not None:
            byte_start = offset
            byte_end = offset + length * _itemsize
            return np.frombuffer(buf[byte_start:byte_end], dtype=_dtype).copy()
        return SpillReader.load_component_tokens(
            worker_dir, shard_id, offset, length, token_dtype=_dtype,
        )

    # Setup output builders
    out_dtype = DType.optimal_dtype(vocab_size)

    if seqlen_threshold is not None:
        stage2_dir = output_dir / "stage2"
        lct_dir = output_dir / "lct"
        stage2_dir.mkdir(parents=True, exist_ok=True)
        lct_dir.mkdir(parents=True, exist_ok=True)
        stage2_builder = IndexedDatasetBuilder(
            str(stage2_dir / f"{output_name}.bin"), dtype=out_dtype,
        )
        lct_builder = IndexedDatasetBuilder(
            str(lct_dir / f"{output_name}.bin"), dtype=out_dtype,
        )
    else:
        out_bin = str(output_dir / f"{output_name}.bin")
        builder = IndexedDatasetBuilder(out_bin, dtype=out_dtype)

    # Process documents (extract only needed columns)
    doc_ids = docs_table.column("document_id").to_pylist()
    doc_modes = docs_table.column("mode").to_pylist()
    n_docs = len(doc_ids)
    total_sequences = 0
    total_tokens = 0

    for i in range(n_docs):
        doc_id = doc_ids[i]
        mode = doc_modes[i]

        comp_entries = comp_by_doc.get(doc_id, [])
        if not comp_entries:
            logger.warning(f"Document {doc_id} has no components — skipping")
            continue

        # Load component tokens
        components = []
        for worker_dir, shard_id, row in comp_entries:
            tokens = _load_tokens_fast(
                worker_dir, shard_id,
                row["token_offset"], row["token_length"],
            )
            components.append((row, tokens))
        components.sort(key=lambda x: x[0]["component_index"])

        # Assemble sequences
        try:
            sequences = assemble_document_sequences(
                mode, components, token_ids,
                max_sequence_tokens=max_sequence_tokens,
            )
        except Exception as exc:
            logger.warning(f"Failed to assemble document {doc_id}: {exc}")
            continue

        # Write sequences
        for seq in sequences:
            n_tok = seq.numel()

            if seqlen_threshold is not None:
                target = stage2_builder if n_tok <= seqlen_threshold else lct_builder
            else:
                target = builder

            target.add_item(seq)
            target.end_document()
            total_sequences += 1
            total_tokens += n_tok

    # Finalize
    if seqlen_threshold is not None:
        stage2_builder.finalize(str(stage2_dir / f"{output_name}.idx"))
        lct_builder.finalize(str(lct_dir / f"{output_name}.idx"))
        logger.info(
            f"Rebuild complete: {total_sequences:,} sequences, {total_tokens:,} tokens "
            f"-> stage2/ + lct/ in {output_dir}"
        )
    else:
        out_idx = str(output_dir / f"{output_name}.idx")
        builder.finalize(out_idx)
        logger.info(
            f"Rebuild complete: {total_sequences:,} sequences, {total_tokens:,} tokens "
            f"-> {output_dir / output_name}"
        )

    return output_dir / output_name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild final Megatron bin/idx from spill shards.",
    )
    parser.add_argument("output_dir", help="Directory containing worker_XX/ spill subdirs")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Max sequence length for interleave splitting")
    parser.add_argument("--seqlen-threshold", type=int, default=None, help="Route to stage2/lct by length")
    parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size for dtype selection")
    parser.add_argument("--output-name", default="rebuilt", help="Output prefix name")

    # Token IDs (required for assembly)
    parser.add_argument("--bos-id", type=int, required=True)
    parser.add_argument("--eos-id", type=int, required=True)
    parser.add_argument("--img-start-id", type=int, required=True)
    parser.add_argument("--img-end-id", type=int, required=True)
    parser.add_argument("--img-token-start-id", type=int, required=True)
    parser.add_argument("--eol-id", type=int, required=True)
    parser.add_argument("--eof-id", type=int, required=True)
    parser.add_argument("--vision-token-offset", type=int, required=True)
    parser.add_argument("--image-token-id", type=int, default=-1)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    token_ids = StructureTokenIds(
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        img_start_id=args.img_start_id,
        img_end_id=args.img_end_id,
        img_token_start_id=args.img_token_start_id,
        eol_id=args.eol_id,
        eof_id=args.eof_id,
        vision_token_offset=args.vision_token_offset,
        image_token_id=args.image_token_id,
    )

    rebuild(
        args.output_dir,
        token_ids=token_ids,
        vocab_size=args.vocab_size,
        max_sequence_tokens=args.max_seq_len,
        seqlen_threshold=args.seqlen_threshold,
        output_name=args.output_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
