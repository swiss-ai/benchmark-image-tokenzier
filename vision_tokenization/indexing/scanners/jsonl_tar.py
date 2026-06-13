"""Generic scanner for raw JSONL + tar datasets."""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import orjson
import pyarrow.parquet as pq

from vision_tokenization.indexing.manifest import (
    INTERLEAVE_JSONL_TAR_SCHEMA,
    append_parquet_records,
    with_media_sha256,
)
from vision_tokenization.indexing.scanners._parallel import run_ordered_pool
from vision_tokenization.indexing.scanners._workers.tar_index import build_tar_index
from vision_tokenization.utils.interleave_documents import (
    extract_local_image_refs,
    parse_interleave_segments,
)

logger = logging.getLogger(__name__)

_WRITE_BUFFER_ROWS = 250_000


def _discover_jsonl_files(input_pattern: str) -> list[str]:
    paths = sorted(glob.glob(input_pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No JSONL files found matching pattern: {input_pattern}")
    return [path for path in paths if os.path.isfile(path)]


def _discover_scope_tars(scope_key: str, tar_pattern: str) -> list[str]:
    pattern = tar_pattern
    if not os.path.isabs(pattern):
        pattern = os.path.join(scope_key, pattern)
    tar_paths = sorted(glob.glob(pattern, recursive=True))
    return [path for path in tar_paths if os.path.isfile(path)]


def _default_global_tar_root(jsonl_paths: Sequence[str]) -> str:
    """Infer a dataset root for global tar lookup from JSONL locations."""
    jsonl_dirs = [str(Path(path).parent) for path in jsonl_paths]
    common_dir = Path(os.path.commonpath(jsonl_dirs))
    unique_dirs = {Path(path) for path in jsonl_dirs}
    if len(unique_dirs) == 1:
        parent = common_dir.parent
        if str(parent) != "":
            return str(parent)
    return str(common_dir)


def _normalize_image_ref(path: str, prefix_strip: str | None) -> str:
    if path.startswith("./"):
        path = path[2:]
    if prefix_strip and path.startswith(prefix_strip):
        path = path[len(prefix_strip):]
    return path


def _extract_image_refs_from_field(
    row: dict[str, Any],
    field: str,
    prefix_strip: str | None,
) -> list[str] | None:
    """Extract image paths from a JSON field. Returns None to skip the row."""
    value = row.get(field)
    if isinstance(value, str):
        paths = [value]
    elif isinstance(value, list) and all(isinstance(v, str) for v in value):
        paths = value
    else:
        return None
    return [_normalize_image_ref(path, prefix_strip) for path in paths]


def _scan_jsonl_scope(
    scope_key: str,
    jsonl_paths: Sequence[str],
    *,
    image_field: Optional[str],
    document_format: Optional[str],
    document_field: Optional[str],
    local_image_prefixes: Optional[Sequence[str]],
    tar_pattern: str,
    image_path_prefix_strip: Optional[str],
    compute_media_sha256: bool = False,
) -> tuple[list[dict], int, int, int, int]:
    """Scan all JSONL files within one tar-resolution scope."""
    tar_paths = _discover_scope_tars(scope_key, tar_pattern)
    if not tar_paths:
        raise FileNotFoundError(
            f"No tar files found for scope={scope_key!r} with tar_pattern={tar_pattern!r}"
        )
    tar_index = build_tar_index(tar_paths, compute_media_sha256=compute_media_sha256)

    records: list[dict] = []
    skipped_missing = 0
    skipped_no_images = 0
    skipped_invalid = 0
    next_group_id = 0

    for jsonl_path in jsonl_paths:
        with open(jsonl_path, "rb") as fh:
            while True:
                line_start = fh.tell()
                raw_line = fh.readline()
                if not raw_line:
                    break
                line_length = len(raw_line)
                sample = orjson.loads(raw_line)

                if image_field is not None:
                    if not isinstance(sample, dict):
                        logger.warning(
                            "Skipping JSONL row at offset %d in %s: expected object row for image_field=%r",
                            line_start,
                            jsonl_path,
                            image_field,
                        )
                        skipped_invalid += 1
                        continue
                    refs = _extract_image_refs_from_field(
                        sample,
                        image_field,
                        image_path_prefix_strip,
                    )
                    if refs is None:
                        logger.warning(
                            "Skipping JSONL row at offset %d in %s: invalid %r field for image refs",
                            line_start,
                            jsonl_path,
                            image_field,
                        )
                        skipped_invalid += 1
                        continue
                else:
                    segments = parse_interleave_segments(
                        sample,
                        document_format=document_format,
                        document_field=document_field,
                        local_prefixes=local_image_prefixes,
                    )
                    refs = extract_local_image_refs(segments)

                if not refs:
                    skipped_no_images += 1
                    continue

                resolved: list[tuple[str, dict]] = []
                missing = False
                for ref in refs:
                    meta = tar_index.get(ref)
                    if meta is None:
                        missing = True
                        break
                    resolved.append((ref, meta))

                if missing:
                    skipped_missing += 1
                    continue

                for image_index, (ref, meta) in enumerate(resolved):
                    record = {
                        "tar_path": meta["tar_path"],
                        "offset_data": meta["offset_data"],
                        "file_size": meta["file_size"],
                        "width": meta["width"],
                        "height": meta["height"],
                        "group_id": next_group_id,
                        "image_index": image_index,
                        "jsonl_path": jsonl_path,
                        "line_start": int(line_start),
                        "line_length": int(line_length),
                        "image_ref": ref,
                    }
                    if compute_media_sha256:
                        record["media_sha256"] = meta["media_sha256"]
                    records.append(record)
                next_group_id += 1

    return records, next_group_id, skipped_no_images, skipped_missing, skipped_invalid


def _offset_group_ids(records: list[dict], offset: int) -> None:
    if offset == 0:
        return
    for rec in records:
        rec["group_id"] += offset


def _scan_jsonl_tar_dataset_impl(
    input_pattern: str,
    output_manifest: Union[str, Path],
    *,
    image_field: Optional[str],
    document_format: Optional[str],
    document_field: Optional[str],
    local_image_prefixes: Optional[Sequence[str]],
    tar_pattern: str,
    tar_scope: str,
    tar_root: Optional[Union[str, Path]],
    image_path_prefix_strip: Optional[str],
    num_workers: int,
    compute_media_sha256: bool,
    metadata_dataset_type: str,
    metadata_extra: Optional[dict[str, Any]] = None,
) -> str:
    from ._metadata import ScanTimer, write_scan_metadata

    if (image_field is None) == (document_format is None):
        raise ValueError("Exactly one of image_field or document_format must be set")
    if tar_scope not in {"parent_dir", "global"}:
        raise ValueError(f"Unsupported tar_scope: {tar_scope!r}")
    if tar_root is not None and tar_scope != "global":
        raise ValueError("tar_root is only supported when tar_scope='global'")

    _timer = ScanTimer()
    _timer.__enter__()  # not a with-block because __exit__ must run after os.replace

    jsonl_paths = _discover_jsonl_files(input_pattern)
    output_path = Path(output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_manifest.exists():
        tmp_manifest.unlink()

    manifest_schema = (
        with_media_sha256(INTERLEAVE_JSONL_TAR_SCHEMA)
        if compute_media_sha256
        else INTERLEAVE_JSONL_TAR_SCHEMA
    )
    writer = pq.ParquetWriter(str(tmp_manifest), manifest_schema, compression="zstd")
    buffer: list[dict] = []
    total_rows = 0
    total_groups = 0
    skipped_missing = 0
    skipped_no_images = 0
    skipped_invalid = 0

    def flush() -> None:
        nonlocal total_rows
        if not buffer:
            return
        total_rows += append_parquet_records(writer, buffer, manifest_schema)
        buffer.clear()

    if tar_scope == "global":
        default_root = _default_global_tar_root(jsonl_paths)
        if tar_root is None:
            global_root = default_root
        else:
            tar_root_path = Path(tar_root).expanduser()
            if tar_root_path.is_absolute():
                global_root = str(tar_root_path)
            else:
                global_root = str((Path(default_root) / tar_root_path).resolve())
        scope_items = [(global_root, jsonl_paths)]
    else:
        grouped: Dict[str, list[str]] = {}
        for jsonl_path in jsonl_paths:
            scope_key = str(Path(jsonl_path).parent)
            grouped.setdefault(scope_key, []).append(jsonl_path)
        scope_items = sorted(grouped.items())

    logger.info(
        "Scanning %d JSONL files across %d scope(s) with %d worker(s)",
        len(jsonl_paths),
        len(scope_items),
        num_workers,
    )

    try:
        next_group_id = 0
        if num_workers <= 1 or len(scope_items) <= 1:
            for scope_idx, (scope_key, scope_jsonl_paths) in enumerate(scope_items, start=1):
                records, num_groups, skipped_no, skipped_missing_local, skipped_invalid_local = _scan_jsonl_scope(
                    scope_key,
                    scope_jsonl_paths,
                    image_field=image_field,
                    document_format=document_format,
                    document_field=document_field,
                    local_image_prefixes=local_image_prefixes,
                    tar_pattern=tar_pattern,
                    image_path_prefix_strip=image_path_prefix_strip,
                    compute_media_sha256=compute_media_sha256,
                )
                _offset_group_ids(records, next_group_id)
                next_group_id += num_groups
                total_groups += num_groups
                skipped_no_images += skipped_no
                skipped_missing += skipped_missing_local
                skipped_invalid += skipped_invalid_local
                if records:
                    buffer.extend(records)
                    if len(buffer) >= _WRITE_BUFFER_ROWS:
                        flush()
                if scope_idx % 100 == 0 or scope_idx == len(scope_items):
                    logger.info(
                        "JSONL+tar scan progress: %d/%d scopes, %d groups, %d rows",
                        scope_idx,
                        len(scope_items),
                        total_groups,
                        total_rows + len(buffer),
                    )
        else:

            def _submit(pool, idx):
                scope_key, scope_jsonl_paths = scope_items[idx]
                return pool.submit(
                    _scan_jsonl_scope,
                    scope_key,
                    scope_jsonl_paths,
                    image_field=image_field,
                    document_format=document_format,
                    document_field=document_field,
                    local_image_prefixes=local_image_prefixes,
                    tar_pattern=tar_pattern,
                    image_path_prefix_strip=image_path_prefix_strip,
                    compute_media_sha256=compute_media_sha256,
                )

            def _emit(_idx, result):
                nonlocal next_group_id, total_groups, skipped_no_images, skipped_missing, skipped_invalid
                records, num_groups, skipped_no, skipped_missing_local, skipped_invalid_local = result
                _offset_group_ids(records, next_group_id)
                next_group_id += num_groups
                total_groups += num_groups
                skipped_no_images += skipped_no
                skipped_missing += skipped_missing_local
                skipped_invalid += skipped_invalid_local
                if records:
                    buffer.extend(records)
                    if len(buffer) >= _WRITE_BUFFER_ROWS:
                        flush()

            run_ordered_pool(
                n_items=len(scope_items),
                submit_fn=_submit,
                emit_fn=_emit,
                num_workers=num_workers,
                progress_fn=lambda done, total: logger.info(
                    "JSONL+tar scan progress: %d/%d scopes, %d groups, %d rows",
                    done,
                    total,
                    total_groups,
                    total_rows + len(buffer),
                ),
            )

        flush()
    finally:
        writer.close()

    os.replace(tmp_manifest, output_path)
    _timer.__exit__(None, None, None)
    extra = {
        "total_groups": total_groups,
        "skipped_no_images": skipped_no_images,
        "skipped_missing": skipped_missing,
        "skipped_invalid": skipped_invalid,
        "compute_media_sha256": compute_media_sha256,
    }
    if metadata_extra:
        extra.update(metadata_extra)
    write_scan_metadata(
        output_path,
        num_workers=num_workers,
        elapsed_seconds=_timer.elapsed,
        total_rows=total_rows,
        dataset_type=metadata_dataset_type,
        extra=extra,
    )

    logger.info(
        "JSONL+tar scan complete: %d groups, %d rows in %.1fs with %d workers",
        total_groups,
        total_rows,
        _timer.elapsed,
        num_workers,
    )
    return str(output_path)


def scan_jsonl_tar_dataset(
    input_pattern: str,
    output_manifest: str | Path,
    *,
    image_field: str | None = None,
    document_format: str | None = None,
    document_field: str | None = None,
    local_image_prefixes: Sequence[str] | None = None,
    tar_pattern: str,
    tar_scope: str = "parent_dir",
    tar_root: str | Path | None = None,
    image_path_prefix_strip: str | None = None,
    num_workers: int = 64,
    compute_media_sha256: bool = False,
) -> str:
    """Scan a raw JSONL + tar dataset into a grouped manifest."""
    return _scan_jsonl_tar_dataset_impl(
        input_pattern,
        output_manifest,
        image_field=image_field,
        document_format=document_format,
        document_field=document_field,
        local_image_prefixes=local_image_prefixes,
        tar_pattern=tar_pattern,
        tar_scope=tar_scope,
        tar_root=tar_root,
        image_path_prefix_strip=image_path_prefix_strip,
        num_workers=num_workers,
        compute_media_sha256=compute_media_sha256,
        metadata_dataset_type="jsonl_tar",
        metadata_extra={
            "image_field": image_field,
            "document_format": document_format,
        },
    )
