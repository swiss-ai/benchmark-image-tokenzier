#!/usr/bin/env python3
"""Filter an existing HF manifest with a contamination ID file."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vision_tokenization.utils.contamination import (
    ContaminationIndex,
    chunk_offsets,
    load_contamination_index,
)
from vision_tokenization.utils.json import json_dump


def _required_columns(schema: pa.Schema) -> None:
    required = {"shard_path", "chunk_index", "row_in_chunk"}
    missing = sorted(required.difference(schema.names))
    if missing:
        raise ValueError(f"Manifest is missing required HF location columns: {missing}")


def _filter_batch(
    table: pa.Table,
    *,
    index: ContaminationIndex,
    path_cache: dict[str, tuple[frozenset[int] | None, list[int] | None]],
    matched_docs: set[tuple[str, int]],
    dropped_by_source: dict[str, int],
) -> tuple[pa.Table, int]:
    keep = np.ones(table.num_rows, dtype=bool)
    shard_paths = table.column("shard_path").to_pylist()
    chunk_indices = table.column("chunk_index").combine_chunks().to_numpy(
        zero_copy_only=False
    )
    rows_in_chunk = table.column("row_in_chunk").combine_chunks().to_numpy(
        zero_copy_only=False
    )

    for row_idx, shard_path in enumerate(shard_paths):
        cached = path_cache.get(shard_path)
        if cached is None:
            rows = index.rows_for_path(shard_path)
            offsets = chunk_offsets(shard_path) if rows else None
            cached = (rows, offsets)
            path_cache[shard_path] = cached
        rows, offsets = cached
        if not rows or offsets is None:
            continue

        chunk_idx = int(chunk_indices[row_idx])
        source_row = int(offsets[chunk_idx]) + int(rows_in_chunk[row_idx])
        if source_row not in rows:
            continue

        keep[row_idx] = False
        key = (Path(shard_path).stem, source_row)
        if key not in matched_docs:
            matched_docs.add(key)
            dropped_by_source[key[0]] = dropped_by_source.get(key[0], 0) + 1

    dropped_rows = int((~keep).sum())
    if dropped_rows == 0:
        return table, 0
    return table.filter(pa.array(keep)), dropped_rows


def decontaminate_manifest(
    input_manifest: str | Path,
    output_manifest: str | Path,
    contamination_ids: str | Path,
    *,
    contamination_format: str = "innovator_vl",
    batch_size: int = 1_000_000,
    overwrite: bool = False,
) -> dict:
    input_manifest = Path(input_manifest)
    output_manifest = Path(output_manifest)
    if output_manifest.exists() and not overwrite:
        raise FileExistsError(f"Output manifest already exists: {output_manifest}")

    index = load_contamination_index(
        contamination_ids,
        format=contamination_format,
    )
    parquet_file = pq.ParquetFile(input_manifest)
    schema = parquet_file.schema_arrow
    _required_columns(schema)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
    if tmp_manifest.exists():
        tmp_manifest.unlink()

    path_cache: dict[str, tuple[frozenset[int] | None, list[int] | None]] = {}
    matched_docs: set[tuple[str, int]] = set()
    dropped_by_source: dict[str, int] = {}
    rows_in = 0
    rows_out = 0
    manifest_rows_dropped = 0

    writer = pq.ParquetWriter(str(tmp_manifest), schema, compression="zstd")
    success = False
    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            rows_in += table.num_rows
            filtered, dropped = _filter_batch(
                table,
                index=index,
                path_cache=path_cache,
                matched_docs=matched_docs,
                dropped_by_source=dropped_by_source,
            )
            manifest_rows_dropped += dropped
            rows_out += filtered.num_rows
            writer.write_table(filtered)
        success = True
    finally:
        writer.close()
        if not success and tmp_manifest.exists():
            tmp_manifest.unlink()

    os.replace(tmp_manifest, output_manifest)
    summary = {
        "input_manifest": str(input_manifest),
        "output_manifest": str(output_manifest),
        "contamination_ids_path": index.path,
        "contamination_format": index.format,
        "contamination_ids": index.total_ids,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "manifest_rows_dropped": manifest_rows_dropped,
        "source_docs_dropped": len(matched_docs),
        "unmatched_contamination_ids": index.total_ids - len(matched_docs),
        "dropped_by_source": dict(sorted(dropped_by_source.items())),
    }
    meta_path = output_manifest.with_name(
        output_manifest.stem + "_decontamination_meta.json"
    )
    json_dump(summary, meta_path)
    summary["metadata_path"] = str(meta_path)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--contamination-ids", required=True)
    parser.add_argument("--format", default="innovator_vl", dest="contamination_format")
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = decontaminate_manifest(
        args.input_manifest,
        args.output_manifest,
        args.contamination_ids,
        contamination_format=args.contamination_format,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(
        "Decontaminated manifest: "
        f"{summary['rows_in']:,} -> {summary['rows_out']:,} rows, "
        f"{summary['source_docs_dropped']:,} source docs dropped "
        f"({summary['metadata_path']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
