"""Provenance sidecars: map tokenized output sequences back to source rows.

When tokenization runs with ``emit_provenance: true``, each micro-shard
``rank_XXXX_chunk_YYYY.{bin,idx}`` gets a parallel ``.src.npy`` holding one
int64 *source manifest row* per output sequence (one per ``end_document()``).
Merge concatenates these in shard order into ``merged.src.npy``.

This module owns the offline-resolution side so the hot tokenize loop stays
untouched:

- ``concat_shard_sidecars`` — merge-time concatenation with per-shard length checks.
- ``doc_first_source_ref`` — per-document source row (rebuild path).
- ``write_provenance_parquet`` — resolve manifest rows to typed ``source_id``
  (``sample_key`` for WDS, ``sample_index`` for HF) → ``merged.provenance.parquet``.
- ``build_group_map`` — join two runs' provenance into
  ``group_map.parquet`` (``source_id, pos_in_A, pos_in_B``).
"""

from __future__ import annotations

import logging
import os
import struct
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SIDECAR_SUFFIX = ".src.npy"


# ---------------------------------------------------------------------------
# Low-level sidecar I/O
# ---------------------------------------------------------------------------


def sidecar_path(prefix: str) -> str:
    """Return the ``.src.npy`` sidecar path for a shard/merged *prefix*."""
    return str(prefix) + SIDECAR_SUFFIX


def save_source_ids(path: str, values) -> None:
    """Write an int64 array to *path* without ``np.save``'s ``.npy`` munging."""
    arr = np.asarray(values, dtype=np.int64)
    with open(path, "wb") as f:
        np.save(f, arr)


def load_source_ids(path: str) -> np.ndarray:
    """Load an int64 source-row array written by :func:`save_source_ids`."""
    with open(path, "rb") as f:
        return np.load(f)


def read_seq_count(prefix: str) -> int:
    """Read the sequence count from a Megatron ``.idx`` header (no mmap).

    MMIDIDX v1 layout: header(9) + version(8) + dtype(1) = 18 bytes, then
    ``sequence_count`` and ``document_count`` as little-endian uint64.
    """
    with open(str(prefix) + ".idx", "rb") as f:
        f.seek(18)
        seq_count, _doc_count = struct.unpack("<QQ", f.read(16))
    return int(seq_count)


# ---------------------------------------------------------------------------
# Plan-derived per-document source rows (rebuild path)
# ---------------------------------------------------------------------------


def doc_first_source_ref(plan) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(sorted_doc_ids, source_ref_of_first_component)`` parallel arrays.

    The source row for a document is the ``source_ref`` of its lowest
    ``component_index`` component (the first image, ``image_index`` 0). For
    single-image documents this is the only row; for multi-image documents it is
    a stable representative. Both tokenization runs apply this identical rule, so
    the resulting keys join consistently across runs.
    """
    comp = plan.components
    order = np.lexsort((comp.component_index, comp.document_id))
    sorted_doc = comp.document_id[order]
    _, first = np.unique(sorted_doc, return_index=True)
    return sorted_doc[first], comp.source_ref[order[first]].astype(np.int64, copy=False)


def doc_source_ids_for(plan, doc_ids_to_process: np.ndarray) -> np.ndarray:
    """Map an ordered ``doc_ids_to_process`` array to per-document source rows."""
    sorted_doc, src = doc_first_source_ref(plan)
    pos = np.searchsorted(sorted_doc, doc_ids_to_process)
    return src[pos]


# ---------------------------------------------------------------------------
# Merge-time concatenation
# ---------------------------------------------------------------------------


def concat_shard_sidecars(
    prefixes: List[str],
    *,
    require: Optional[bool] = None,
) -> Optional[np.ndarray]:
    """Concatenate per-shard ``.src.npy`` in *prefixes* order.

    *require* controls behaviour when sidecars are missing:
    ``None`` (auto) → emit only when every shard has one, else skip silently;
    ``True`` → raise if any are missing; ``False`` → never emit (return None).

    Each shard's sidecar length is asserted against its ``.idx`` sequence count
    so any drift fails loudly rather than producing a misaligned map.
    """
    if require is False:
        return None

    present = [os.path.exists(sidecar_path(p)) for p in prefixes]
    if not any(present):
        if require is True:
            raise FileNotFoundError("Provenance required but no .src.npy sidecars were found")
        return None
    if not all(present):
        msg = (
            f"Partial provenance: {sum(present)}/{len(present)} shards have a "
            f".src.npy sidecar (run must enable emit_provenance from the start)"
        )
        if require is True:
            raise FileNotFoundError(msg)
        logger.warning("%s — skipping merged.src.npy", msg)
        return None

    parts = []
    for prefix in prefixes:
        arr = load_source_ids(sidecar_path(prefix))
        seq_count = read_seq_count(prefix)
        if len(arr) != seq_count:
            raise ValueError(f"Sidecar length {len(arr)} != {seq_count} sequences for shard {prefix}")
        parts.append(arr)
    # ``parts`` is non-empty here: the ``not all(present)`` guard above ensures
    # every prefix contributed a sidecar.
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Manifest resolution → typed Parquet
# ---------------------------------------------------------------------------


def _resolve_source_ids(manifest_path: str, rows: np.ndarray):
    """Resolve manifest *rows* to the typed source identity column.

    Returns ``(column_name, pyarrow_array)`` where the column is ``sample_key``
    (WDS, string) or ``sample_index`` (HF, int).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pq.read_schema(manifest_path)
    names = set(schema.names)
    if "sample_key" in names:
        col = "sample_key"
    elif "sample_index" in names:
        col = "sample_index"
    else:
        raise ValueError(
            f"Manifest {manifest_path} has neither 'sample_key' (WDS) nor "
            f"'sample_index' (HF); cannot resolve source ids"
        )
    column = pq.read_table(manifest_path, columns=[col]).column(col)
    taken = column.take(pa.array(np.asarray(rows, dtype=np.int64)))
    return col, taken


def write_provenance_parquet(
    merged_prefix: str,
    manifest_path: str,
    out_path: Optional[str] = None,
) -> str:
    """Resolve ``merged.src.npy`` to ``merged.provenance.parquet``.

    Columns: ``output_index`` (int64), ``manifest_row`` (int64),
    ``source_id`` (string|int — the dataset's native identity).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    src = load_source_ids(sidecar_path(merged_prefix))
    seq_count = read_seq_count(merged_prefix)
    if len(src) != seq_count:
        raise ValueError(f"merged.src.npy length {len(src)} != {seq_count} merged sequences")
    col, source_ids = _resolve_source_ids(manifest_path, src)
    table = pa.table(
        {
            "output_index": pa.array(np.arange(len(src), dtype=np.int64)),
            "manifest_row": pa.array(src.astype(np.int64, copy=False)),
            "source_id": source_ids,
        }
    )
    out = out_path or (str(merged_prefix) + ".provenance.parquet")
    pq.write_table(table, out, compression="zstd")
    logger.info("Wrote provenance parquet (%s id): %s (%d rows)", col, out, len(src))
    return out


# ---------------------------------------------------------------------------
# Two-run grouping
# ---------------------------------------------------------------------------


def build_group_map(
    prov_a_path: str,
    prov_b_path: str,
    out_path: str,
) -> str:
    """Join two runs' ``*.provenance.parquet`` into ``group_map.parquet``.

    Output columns: ``source_id``, ``pos_in_A``, ``pos_in_B`` — one row per
    source sample present in **both** runs, giving the output index in each.

    Assumes one output sequence per source manifest row (true for image2text
    and sft); raises on duplicate rows (e.g. interleave splits one document
    into several sequences), which would make a positional join ambiguous.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    a = pq.read_table(prov_a_path)
    b = pq.read_table(prov_b_path)
    a_row = a.column("manifest_row").to_numpy()
    b_row = b.column("manifest_row").to_numpy()
    a_pos = a.column("output_index").to_numpy()
    b_pos = b.column("output_index").to_numpy()

    for name, rows in (("A", a_row), ("B", b_row)):
        if len(np.unique(rows)) != len(rows):
            raise ValueError(
                f"Run {name} maps multiple sequences to the same source row "
                f"(interleave/multi-sequence docs); build_group_map requires one "
                f"output sequence per source row"
            )

    n = int(max(a_row.max(initial=-1), b_row.max(initial=-1))) + 1
    pos_a = np.full(n, -1, dtype=np.int64)
    pos_b = np.full(n, -1, dtype=np.int64)
    a_idx = np.full(n, -1, dtype=np.int64)
    pos_a[a_row] = a_pos
    pos_b[b_row] = b_pos
    a_idx[a_row] = np.arange(len(a_row), dtype=np.int64)

    common = np.nonzero((pos_a >= 0) & (pos_b >= 0))[0]
    source_id = a.column("source_id").take(pa.array(a_idx[common]))
    table = pa.table(
        {
            "source_id": source_id,
            "pos_in_A": pa.array(pos_a[common]),
            "pos_in_B": pa.array(pos_b[common]),
        }
    )
    pq.write_table(table, out_path, compression="zstd")
    logger.info(
        "Wrote group map: %s (%d shared of %d/%d source rows)",
        out_path,
        len(common),
        len(a_row),
        len(b_row),
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_res = sub.add_parser("resolve", help="merged.src.npy + manifest -> merged.provenance.parquet")
    p_res.add_argument("merged_prefix", help="Merged dataset prefix (no extension)")
    p_res.add_argument("--manifest", required=True, help="Manifest parquet used to build the plan")
    p_res.add_argument("--out", default=None, help="Output parquet path")

    p_grp = sub.add_parser("group", help="Join two runs' provenance into group_map.parquet")
    p_grp.add_argument("provenance_a", help="Run A *.provenance.parquet")
    p_grp.add_argument("provenance_b", help="Run B *.provenance.parquet")
    p_grp.add_argument("--out", required=True, help="Output group_map.parquet path")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "resolve":
        out = write_provenance_parquet(args.merged_prefix, args.manifest, args.out)
        print(f"Wrote {out}")
    elif args.cmd == "group":
        out = build_group_map(args.provenance_a, args.provenance_b, args.out)
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
