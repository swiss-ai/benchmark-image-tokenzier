#!/usr/bin/env python3
"""Filter an existing HF or WebDataset manifest with a contamination ID file."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402  (repo is not pip-installable)

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from vision_tokenization.utils.contamination import (  # noqa: E402
    ContaminationIndex,
    HfSourceRowResolver,
    assert_unique_resolution,
    load_contamination_index,
)
from vision_tokenization.utils.json import json_dump  # noqa: E402


def _detect_kind(schema: pa.Schema) -> str:
    """Return 'hf' or 'wds' from the manifest's columns."""
    names = set(schema.names)
    if {"shard_path", "chunk_index", "row_in_chunk"} <= names:
        return "hf"
    if {"sample_key", "tar_path"} <= names:
        return "wds"
    raise ValueError("Manifest is neither HF (shard_path/chunk_index/row_in_chunk) nor WDS (sample_key/tar_path).")


def _record_drop(
    matched_docs: set[tuple[str, object]],
    dropped_by_source: dict[str, int],
    source: str,
    doc: object,
) -> None:
    """Mark one source doc dropped, counting it once even if it spans many rows."""
    key = (source, doc)
    if key not in matched_docs:
        matched_docs.add(key)
        dropped_by_source[source] = dropped_by_source.get(source, 0) + 1


def _filter_batch(
    table: pa.Table,
    *,
    resolver: HfSourceRowResolver,
    matched_docs: set[tuple[str, int]],
    dropped_by_source: dict[str, int],
) -> tuple[pa.Table, int]:
    keep = np.ones(table.num_rows, dtype=bool)
    shard_paths = table.column("shard_path").to_pylist()
    chunk_indices = table.column("chunk_index").combine_chunks().to_numpy(zero_copy_only=False)
    rows_in_chunk = table.column("row_in_chunk").combine_chunks().to_numpy(zero_copy_only=False)

    for row_idx, shard_path in enumerate(shard_paths):
        source_row = resolver.contaminated_source_row(shard_path, chunk_indices[row_idx], rows_in_chunk[row_idx])
        if source_row is None:
            continue
        keep[row_idx] = False
        # Key on the full shard path (not the stem) so docs in different subsets
        # that share a filename stem are counted separately.
        _record_drop(matched_docs, dropped_by_source, shard_path, source_row)

    dropped_rows = int((~keep).sum())
    if dropped_rows == 0:
        return table, 0
    return table.filter(pa.array(keep)), dropped_rows


def _filter_batch_wds(
    table: pa.Table,
    *,
    index: ContaminationIndex,
    path_cache: dict[str, frozenset],
    matched_docs: set[tuple[str, str]],
    dropped_by_source: dict[str, int],
) -> tuple[pa.Table, int]:
    keep = np.ones(table.num_rows, dtype=bool)
    tar_paths = table.column("tar_path").to_pylist()
    sample_keys = table.column("sample_key").to_pylist()

    for row_idx, tar_path in enumerate(tar_paths):
        excluded = path_cache.get(tar_path)
        if excluded is None:
            excluded = index.rows_for_path(tar_path)
            path_cache[tar_path] = excluded
        if not excluded:
            continue

        sample_key = sample_keys[row_idx]
        if sample_key not in excluded:
            continue

        keep[row_idx] = False
        # A sample may span multiple image rows (multi-image); count it once in
        # source_docs_dropped while dropping every one of its rows.
        _record_drop(matched_docs, dropped_by_source, tar_path, sample_key)

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
    kind = _detect_kind(schema)
    id_column = "shard_path" if kind == "hf" else "tar_path"

    # Each contamination id must resolve to exactly one shard/tar in the manifest;
    # abort on ambiguity (would over-exclude) before rewriting anything.
    distinct_sources = pq.read_table(input_manifest, columns=[id_column]).column(id_column).unique().to_pylist()
    unmatched = assert_unique_resolution(index, distinct_sources, source="manifest")
    if unmatched:
        print(
            f"WARNING: {len(unmatched)} contamination id key(s) matched no {id_column} in the manifest "
            f"(first few: {unmatched[:5]})",
            file=sys.stderr,
        )

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
    if tmp_manifest.exists():
        tmp_manifest.unlink()

    resolver = HfSourceRowResolver(index) if kind == "hf" else None
    path_cache: dict = {}
    matched_docs: set = set()
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
            if kind == "hf":
                filtered, dropped = _filter_batch(
                    table,
                    resolver=resolver,
                    matched_docs=matched_docs,
                    dropped_by_source=dropped_by_source,
                )
            else:
                filtered, dropped = _filter_batch_wds(
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
    # Count of id tokens that dropped nothing. An id ``(key, doc)`` matched iff its
    # key resolved to a shard/tar ``P`` and ``(P, doc)`` was dropped. Deriving this
    # from ``total_ids - len(matched_docs)`` would over-report when the same physical
    # id is listed under two key forms (e.g. ``stem:5`` and ``subset/stem:5``), so
    # resolve per key instead.
    key_paths = index.resolve(distinct_sources)
    unmatched_ids = sum(
        1
        for key, docs in index.by_source.items()
        for doc in docs
        if not key_paths[key] or (key_paths[key][0], doc) not in matched_docs
    )
    summary = {
        "input_manifest": str(input_manifest),
        "output_manifest": str(output_manifest),
        "manifest_kind": kind,
        "contamination_ids_path": index.path,
        "contamination_format": index.format,
        "contamination_ids": index.total_ids,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "manifest_rows_dropped": manifest_rows_dropped,
        "source_docs_dropped": len(matched_docs),
        "unmatched_contamination_ids": unmatched_ids,
        "dropped_by_source": dict(sorted(dropped_by_source.items())),
    }
    meta_path = output_manifest.with_name(output_manifest.stem + "_decontamination_meta.json")
    json_dump(summary, meta_path)
    summary["metadata_path"] = str(meta_path)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--contamination-ids", required=True)
    parser.add_argument("--contamination-format", "--format", default="innovator_vl", dest="contamination_format")
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
