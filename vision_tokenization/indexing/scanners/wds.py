"""Parallel WDS tar scanner — discovers shards, scans in parallel, writes manifest."""

import glob
import logging
import os
from collections import Counter
from pathlib import Path
from typing import FrozenSet, Optional, Union

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from vision_tokenization.indexing.scanners._workers.wds import (
    DEFAULT_IMAGE_EXTENSIONS,
    DEFAULT_TEXT_EXTENSIONS,
    scan_single_tar,
)
from vision_tokenization.indexing.manifest import (
    WDS_SCHEMA,
    WDS_SCHEMA_MULTI_IMAGE,
    WDS_SCHEMA_MULTI_IMAGE_WITH_TEXT,
    WDS_SCHEMA_WITH_TEXT,
    records_to_table,
)

from vision_tokenization.indexing.scanners._parallel import run_ordered_pool

logger = logging.getLogger(__name__)

_WDS_WRITE_BUFFER_ROWS = 500_000


def _discover_shards(input_pattern: str) -> list[str]:
    """Discover tar shards from a braceexpand or glob pattern.

    Tries braceexpand first (e.g. ``data_{000..100}.tar``), then falls
    back to standard glob.
    """
    tar_paths: list[str] = []

    if "{" in input_pattern and ".." in input_pattern:
        try:
            import braceexpand

            expanded = list(braceexpand.braceexpand(input_pattern))
            tar_paths = sorted(p for p in expanded if os.path.isfile(p))
            if tar_paths:
                logger.info(
                    f"Braceexpand: {len(expanded)} paths expanded, "
                    f"{len(tar_paths)} existing tar files found"
                )
                return tar_paths
            logger.warning("Braceexpand produced paths but none exist. Falling back to glob.")
        except ImportError:
            logger.warning("braceexpand not installed, falling back to glob")
        except Exception as exc:
            logger.warning(f"braceexpand failed ({exc}), falling back to glob")

    tar_paths = sorted(glob.glob(input_pattern, recursive=True))
    if not tar_paths:
        raise FileNotFoundError(f"No tar files found matching pattern: {input_pattern}")

    logger.info(f"Glob: {len(tar_paths)} tar files found matching '{input_pattern}'")
    return tar_paths


def _find_duplicate_sample(records: list[dict]) -> Optional[tuple[str, int]]:
    """Return the first normalized sample key with multiple images, if any."""
    sample_counts = Counter(rec["sample_key"] for rec in records)
    for sample_key, size in sample_counts.items():
        if size > 1:
            return sample_key, size
    return None


def _validate_single_image_records(
    records: list[dict],
    *,
    tar_path: str,
    image_field_pattern: Optional[str],
) -> None:
    """Validate that a single-image tar does not emit duplicate normalized keys."""
    if image_field_pattern is None:
        return
    duplicate = _find_duplicate_sample(records)
    if duplicate is None:
        return
    sample_key, size = duplicate
    raise ValueError(
        "scan_wds_dataset(..., multi_image=False) found a sample with multiple images "
        f"after normalizing {image_field_pattern!r}: sample_key={sample_key!r}, "
        f"tar_path={tar_path!r}, images={size}. Set multi_image=true for grouped output."
    )


def _summarize_multi_image_records(records: list[dict]) -> tuple[int, bool]:
    """Return (num_groups, has_non_singleton_group) for a tar result."""
    if not records:
        return 0, False
    num_groups = len({rec["group_id"] for rec in records})
    return num_groups, num_groups < len(records)


def _finalize_record_table(
    records: list[dict],
    *,
    schema: pa.Schema,
    group_id_offset: int = 0,
) -> pa.Table:
    """Convert records to a typed Arrow table and offset group ids if needed."""
    table = records_to_table(records, schema)
    if "group_id" not in schema.names or len(table) == 0 or group_id_offset == 0:
        return table

    arrays = {name: table.column(name) for name in schema.names}
    arrays["group_id"] = pc.add(
        table.column("group_id"),
        pa.scalar(group_id_offset, type=pa.int64()),
    )
    return pa.table(arrays, schema=schema)


def _flush_table_buffer(writer: pq.ParquetWriter, buffer: list[pa.Table]) -> None:
    """Flush buffered Arrow tables to Parquet."""
    if not buffer:
        return
    writer.write_table(pa.concat_tables(buffer, promote_options="none"))
    buffer.clear()


def _select_wds_schema(*, include_text: bool, multi_image: bool) -> pa.Schema:
    """Choose the manifest schema for the requested WDS scan mode."""
    if multi_image:
        return WDS_SCHEMA_MULTI_IMAGE_WITH_TEXT if include_text else WDS_SCHEMA_MULTI_IMAGE
    return WDS_SCHEMA_WITH_TEXT if include_text else WDS_SCHEMA


def scan_wds_dataset(
    input_pattern: str,
    output_manifest: Union[str, Path],
    num_workers: int = 64,
    image_extensions: Optional[FrozenSet[str]] = None,
    text_extensions: Optional[FrozenSet[str]] = None,
    image_field_pattern: Optional[str] = None,
    multi_image: bool = False,
) -> str:
    """Scan all WDS tars in parallel and write a Parquet manifest."""
    from ._metadata import ScanTimer, write_scan_metadata
    _timer = ScanTimer()
    _timer.__enter__()
    if image_extensions is None:
        image_extensions = DEFAULT_IMAGE_EXTENSIONS
    if multi_image and image_field_pattern is None:
        raise ValueError(
            "scan_wds_dataset(..., multi_image=True) requires image_field_pattern."
        )

    tar_paths = _discover_shards(input_pattern)
    include_text = text_extensions is not None
    schema = _select_wds_schema(include_text=include_text, multi_image=multi_image)
    logger.info(
        f"Scanning {len(tar_paths)} tar files with {num_workers} workers"
        f"{' (with text sidecars)' if include_text else ''}"
        f"{f' (field pattern: {image_field_pattern}*)' if image_field_pattern is not None else ''}"
        f"{' (grouped multi-image)' if multi_image else ''}..."
    )

    output_manifest = str(output_manifest)
    output_path = Path(output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_manifest.exists():
        tmp_manifest.unlink()

    completed = 0
    failed_tars: list[str] = []
    total_rows = 0
    buffered_rows = 0
    global_group_offset = 0
    total_groups = 0
    saw_multi_image_group = False
    writer = pq.ParquetWriter(str(tmp_manifest), schema, compression="zstd")
    buffer: list[pa.Table] = []
    success = False

    try:
        def _submit(pool, idx):
            return pool.submit(
                scan_single_tar,
                tar_paths[idx],
                image_extensions,
                text_extensions,
                image_field_pattern,
                multi_image,
            )

        def _emit(idx, records):
            nonlocal completed, total_rows, buffered_rows
            nonlocal global_group_offset, total_groups, saw_multi_image_group
            tar_path = tar_paths[idx]

            if multi_image:
                num_groups, has_non_singleton = _summarize_multi_image_records(records)
                total_groups += num_groups
                saw_multi_image_group |= has_non_singleton
                table = _finalize_record_table(
                    records, schema=schema, group_id_offset=global_group_offset,
                )
                global_group_offset += num_groups
            else:
                _validate_single_image_records(
                    records, tar_path=tar_path,
                    image_field_pattern=image_field_pattern,
                )
                table = _finalize_record_table(records, schema=schema)

            if len(table):
                buffer.append(table)
                buffered_rows += len(table)
                total_rows += len(table)
                if buffered_rows >= _WDS_WRITE_BUFFER_ROWS:
                    _flush_table_buffer(writer, buffer)
                    buffered_rows = 0

            completed += 1

        def _error(idx, exc):
            logger.exception(f"Failed to scan {tar_paths[idx]}", exc_info=exc)
            failed_tars.append(tar_paths[idx])

        run_ordered_pool(
            n_items=len(tar_paths),
            submit_fn=_submit,
            emit_fn=_emit,
            num_workers=num_workers,
            error_fn=_error,
            progress_fn=lambda done, total: logger.info(
                f"Progress: {done}/{total} tars scanned, "
                f"{total_rows:,} images found so far"
            ),
        )

        _flush_table_buffer(writer, buffer)
        success = True
    finally:
        writer.close()
        if not success and tmp_manifest.exists():
            tmp_manifest.unlink()

    tmp_manifest.replace(output_path)

    if failed_tars:
        logger.error(
            f"{len(failed_tars)}/{len(tar_paths)} tar files failed to scan: "
            f"{failed_tars[:10]}{'...' if len(failed_tars) > 10 else ''}"
        )

    if multi_image and total_groups and not saw_multi_image_group:
        logger.warning(
            "scan_wds_dataset(..., multi_image=True) found only singleton groups after "
            "parsing with image_field_pattern=%r. Consider multi_image=false.",
            image_field_pattern,
        )

    _timer.__exit__(None, None, None)
    write_scan_metadata(
        output_path,
        num_workers=num_workers,
        elapsed_seconds=_timer.elapsed,
        total_rows=total_rows,
        dataset_type="wds",
        extra={
            "num_tars": len(tar_paths),
            "failed_tars": len(failed_tars),
            "multi_image": multi_image,
        },
    )

    logger.info(
        f"Scan complete: {total_rows:,} images from {len(tar_paths)} tars "
        f"in {_timer.elapsed:.1f}s with {num_workers} workers"
    )
    return output_manifest
