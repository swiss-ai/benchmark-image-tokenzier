"""Parallel HF dataset scanner for Arrow and Parquet shard files."""

import glob
import logging
import os
import time
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from vision_tokenization.indexing._scan_hf_arrow_worker import scan_single_hf_arrow_shard
from vision_tokenization.indexing._scan_hf_parquet_worker import scan_single_hf_parquet_shard
from vision_tokenization.indexing.manifest import (
    HF_SCHEMA_PHYSICAL,
    HF_SCHEMA_PHYSICAL_MULTI_IMAGE,
)

from vision_tokenization.indexing._parallel import run_ordered_pool

logger = logging.getLogger(__name__)

_HF_SHARD_SUFFIXES = {".arrow", ".parquet"}
_HF_WRITE_BUFFER_ROWS = 500_000


def _filter_shards(paths: Iterable[Union[str, Path]]) -> list[str]:
    shards = []
    for path in paths:
        p = Path(path)
        if not p.is_file():
            continue
        if p.suffix not in _HF_SHARD_SUFFIXES:
            continue
        if p.name.startswith("manifest"):
            continue
        shards.append(str(p))
    return sorted(shards)


def _discover_shards(input_pattern: Union[str, Path]) -> list[str]:
    """Discover HF Arrow/Parquet shards from a directory, path, or glob."""
    input_pattern = str(input_pattern)
    path = Path(input_pattern)

    if path.is_dir():
        shard_paths = _filter_shards(path.rglob("*"))
        if shard_paths:
            return shard_paths

    if path.is_file():
        shard_paths = _filter_shards([path])
        if shard_paths:
            return shard_paths

    if "{" in input_pattern and ".." in input_pattern:
        try:
            import braceexpand

            expanded = list(braceexpand.braceexpand(input_pattern))
            shard_paths = _filter_shards(expanded)
            if shard_paths:
                logger.info(
                    f"Braceexpand: {len(expanded)} paths expanded, "
                    f"{len(shard_paths)} shard files found"
                )
                return shard_paths
            logger.warning(
                "Braceexpand produced paths but no Arrow/Parquet shards were found. "
                "Falling back to glob."
            )
        except ImportError:
            logger.warning("braceexpand not installed, falling back to glob")
        except Exception as exc:
            logger.warning(f"braceexpand failed ({exc}), falling back to glob")

    shard_paths = _filter_shards(glob.glob(input_pattern, recursive=True))
    if not shard_paths:
        raise FileNotFoundError(
            f"No Arrow/Parquet shard files found matching: {input_pattern}"
        )
    return shard_paths


def _make_shard_path_array(shard_path: str, n_rows: int) -> pa.DictionaryArray:
    indices = pa.array(np.zeros(n_rows, dtype=np.int32), type=pa.int32())
    dictionary = pa.array([shard_path], type=pa.string())
    return pa.DictionaryArray.from_arrays(indices, dictionary)


def _finalize_shard_table(
    table: pa.Table,
    sample_offset: int,
    shard_path: str,
    schema: pa.Schema,
    is_multi: bool,
) -> pa.Table:
    sample_offset_scalar = pa.scalar(sample_offset, type=pa.int64())
    arrays = {
        "sample_index": pc.add(table.column("sample_index"), sample_offset_scalar),
        "width": table.column("width"),
        "height": table.column("height"),
        "shard_path": _make_shard_path_array(shard_path, len(table)),
        "chunk_index": table.column("chunk_index"),
        "row_in_chunk": table.column("row_in_chunk"),
    }
    if is_multi:
        arrays["group_id"] = pc.add(table.column("group_id"), sample_offset_scalar)
        arrays["image_index"] = table.column("image_index")
    return pa.table(arrays, schema=schema)


def _flush_table_buffer(writer: pq.ParquetWriter, buffer: list[pa.Table]) -> None:
    if not buffer:
        return
    writer.write_table(pa.concat_tables(buffer, promote_options="none"))
    buffer.clear()


def _scan_single_hf_shard(
    shard_path: str,
    image_column: str = "image",
    image_list_column: Optional[str] = None,
):
    if shard_path.endswith(".arrow"):
        return scan_single_hf_arrow_shard(
            shard_path,
            image_column=image_column,
            image_list_column=image_list_column,
        )
    if shard_path.endswith(".parquet"):
        return scan_single_hf_parquet_shard(
            shard_path,
            image_column=image_column,
            image_list_column=image_list_column,
        )
    raise ValueError(f"Unsupported HF shard format: {shard_path}")


def _process_shard_result(
    shard_path: str,
    result,
    *,
    total_source_rows: int,
    total_manifest_rows: int,
    total_failed_dims: int,
    skipped_shards: int,
    is_multi: bool,
    buffer: list[pa.Table],
    buffered_rows: int,
    writer: pq.ParquetWriter,
    schema: pa.Schema,
):
    table, source_rows, failed_dims, skip_reason = result
    if skip_reason is not None:
        skipped_shards += 1
        logger.warning("Skipping HF shard %s: %s", shard_path, skip_reason)
        return (
            total_source_rows,
            total_manifest_rows,
            total_failed_dims,
            skipped_shards,
            buffered_rows,
        )

    table = _finalize_shard_table(
        table,
        total_source_rows,
        shard_path,
        schema,
        is_multi,
    )
    total_source_rows += source_rows
    total_failed_dims += failed_dims
    total_manifest_rows += len(table)

    if len(table):
        buffer.append(table)
        buffered_rows += len(table)
        if buffered_rows >= _HF_WRITE_BUFFER_ROWS:
            _flush_table_buffer(writer, buffer)
            buffered_rows = 0

    return (
        total_source_rows,
        total_manifest_rows,
        total_failed_dims,
        skipped_shards,
        buffered_rows,
    )


def scan_hf_dataset(
    input_pattern: Union[str, Path],
    output_manifest: Union[str, Path],
    image_column: str = "image",
    image_list_column: Optional[str] = None,
    num_workers: int = 8,
) -> str:
    """Scan HF Arrow/Parquet shards and write a Parquet manifest.

    Args:
        input_pattern: Directory, file path, braceexpand pattern, or glob
            matching HF ``.arrow`` or ``.parquet`` shards.
        output_manifest: Destination Parquet path.
        image_column: Column name containing a single image.
        image_list_column: Column name for multi-image ``List[Image]`` data.
        num_workers: Number of worker processes to scan shards in parallel.

    Returns:
        The output manifest path as a string.
    """
    t0 = time.time()
    is_multi = image_list_column is not None
    schema = HF_SCHEMA_PHYSICAL_MULTI_IMAGE if is_multi else HF_SCHEMA_PHYSICAL

    shard_paths = _discover_shards(input_pattern)
    num_arrow = sum(path.endswith(".arrow") for path in shard_paths)
    num_parquet = sum(path.endswith(".parquet") for path in shard_paths)
    logger.info(
        f"Scanning {len(shard_paths)} HF shards "
        f"({num_arrow} arrow, {num_parquet} parquet) "
        f"with {num_workers} workers"
    )

    output_manifest = str(output_manifest)
    Path(output_manifest).parent.mkdir(parents=True, exist_ok=True)

    total_source_rows = 0
    total_manifest_rows = 0
    total_failed_dims = 0
    skipped_shards = 0

    writer = pq.ParquetWriter(output_manifest, schema, compression="zstd")
    buffer: list[pa.Table] = []
    buffered_rows = 0

    def _submit(pool, idx):
        return pool.submit(
            _scan_single_hf_shard,
            shard_paths[idx],
            image_column,
            image_list_column,
        )

    def _emit(idx, result):
        nonlocal total_source_rows, total_manifest_rows, total_failed_dims
        nonlocal skipped_shards, buffered_rows
        shard_path = shard_paths[idx]
        (
            total_source_rows,
            total_manifest_rows,
            total_failed_dims,
            skipped_shards,
            buffered_rows,
        ) = _process_shard_result(
            shard_path,
            result,
            total_source_rows=total_source_rows,
            total_manifest_rows=total_manifest_rows,
            total_failed_dims=total_failed_dims,
            skipped_shards=skipped_shards,
            is_multi=is_multi,
            buffer=buffer,
            buffered_rows=buffered_rows,
            writer=writer,
            schema=schema,
        )

    try:
        run_ordered_pool(
            n_items=len(shard_paths),
            submit_fn=_submit,
            emit_fn=_emit,
            num_workers=num_workers,
            progress_fn=lambda done, total: logger.info(
                f"Progress: {done}/{total} shards scanned, "
                f"{total_source_rows:,} source rows, "
                f"{total_manifest_rows:,} manifest rows "
                f"(latest: {os.path.basename(shard_paths[done - 1])})"
            ),
        )
    finally:
        _flush_table_buffer(writer, buffer)
        writer.close()

    elapsed = time.time() - t0
    logger.info(
        f"Manifest saved: {total_manifest_rows:,} rows from {total_source_rows:,} "
        f"source rows -> {output_manifest} ({elapsed:.1f}s, "
        f"{total_failed_dims:,} failed dimension extractions, "
        f"{skipped_shards:,} skipped shards)"
    )
    return output_manifest
