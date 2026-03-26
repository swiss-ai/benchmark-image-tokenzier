"""Scanner for raw JSONL + tar interleaved datasets."""

from __future__ import annotations

import glob
import logging
import os
import tarfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import orjson
import pyarrow.parquet as pq

from vision_tokenization.indexing.scanners._parallel import run_ordered_pool
from vision_tokenization.indexing.scanners._workers.wds import (
    DEFAULT_IMAGE_EXTENSIONS,
    _get_image_dims,
)
from vision_tokenization.indexing.manifest import (
    INTERLEAVE_JSONL_TAR_SCHEMA,
    append_parquet_records,
)
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
    """Infer a dataset root for global tar lookup from JSONL locations.

    If all JSONL files live in the same directory, treat that directory as a
    dataset leaf such as ``jsonl/`` and resolve global tar globs from its
    parent. Otherwise, use the common parent directory across JSONL parents.
    """
    jsonl_dirs = [str(Path(path).parent) for path in jsonl_paths]
    common_dir = Path(os.path.commonpath(jsonl_dirs))
    unique_dirs = {Path(path) for path in jsonl_dirs}
    if len(unique_dirs) == 1:
        parent = common_dir.parent
        if str(parent) != "":
            return str(parent)
    return str(common_dir)


def _build_tar_index(
    tar_paths: Sequence[str],
    image_extensions: Sequence[str] = tuple(DEFAULT_IMAGE_EXTENSIONS),
) -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    image_exts = frozenset(ext.lower().lstrip(".") for ext in image_extensions)

    for tar_path in tar_paths:
        try:
            tf = tarfile.open(tar_path, "r")
        except Exception:
            logger.warning("Skipping unreadable tar: %s", tar_path, exc_info=True)
            continue
        try:
            for member in tf:
                if not member.isfile():
                    continue
                name = member.name
                basename = os.path.basename(name)
                if "." not in basename:
                    continue
                _stem, ext = basename.rsplit(".", 1)
                ext = ext.lower()
                if ext not in image_exts:
                    continue
                if name in index:
                    continue
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                width, height = _get_image_dims(fobj, ext)
                if width < 0 or height < 0:
                    continue
                index[name] = {
                    "tar_path": tar_path,
                    "offset_data": int(member.offset_data),
                    "file_size": int(member.size),
                    "width": int(width),
                    "height": int(height),
                }
        except Exception:
            logger.warning("Error reading tar (truncated?): %s", tar_path, exc_info=True)
        finally:
            tf.close()
    return index


def _scan_jsonl_scope(
    scope_key: str,
    jsonl_paths: Sequence[str],
    *,
    document_format: str,
    document_field: Optional[str],
    local_image_prefixes: Optional[Sequence[str]],
    tar_pattern: str,
 ) -> tuple[list[dict], int, int, int]:
    """Scan all JSONL files within one tar-resolution scope.

    Returns:
        (records, num_groups, skipped_no_images, skipped_missing)
    """
    tar_paths = _discover_scope_tars(scope_key, tar_pattern)
    if not tar_paths:
        raise FileNotFoundError(
            f"No tar files found for scope={scope_key!r} with tar_pattern={tar_pattern!r}"
        )
    tar_index = _build_tar_index(tar_paths)

    records: list[dict] = []
    skipped_missing = 0
    skipped_no_images = 0
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

                resolved = []
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
                    records.append(
                        {
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
                    )
                next_group_id += 1

    return records, next_group_id, skipped_no_images, skipped_missing


def _offset_group_ids(records: list[dict], offset: int) -> None:
    if offset == 0:
        return
    for rec in records:
        rec["group_id"] += offset


def scan_jsonl_tar_interleave_dataset(
    input_pattern: str,
    output_manifest: Union[str, Path],
    *,
    document_format: str,
    document_field: Optional[str] = None,
    local_image_prefixes: Optional[Sequence[str]] = None,
    tar_pattern: str = "content_image.tar*",
    tar_scope: str = "parent_dir",
    tar_root: Optional[Union[str, Path]] = None,
    num_workers: int = 64,
) -> str:
    """Scan a raw JSONL + tar interleave dataset into a grouped manifest.

    Emits one manifest row per resolved local image occurrence. Documents with
    zero local images are dropped. Documents with any unresolved local image
    ref are skipped entirely.
    """
    from ._metadata import ScanTimer, write_scan_metadata
    _timer = ScanTimer()
    _timer.__enter__()
    if tar_scope not in {"parent_dir", "global"}:
        raise ValueError(f"Unsupported tar_scope: {tar_scope!r}")
    if tar_root is not None and tar_scope != "global":
        raise ValueError("tar_root is only supported when tar_scope='global'")

    jsonl_paths = _discover_jsonl_files(input_pattern)
    output_path = Path(output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_manifest.exists():
        tmp_manifest.unlink()

    writer = pq.ParquetWriter(str(tmp_manifest), INTERLEAVE_JSONL_TAR_SCHEMA, compression="zstd")
    buffer: list[dict] = []
    total_rows = 0
    total_groups = 0
    skipped_missing = 0
    skipped_no_images = 0

    def flush() -> None:
        nonlocal total_rows
        if not buffer:
            return
        total_rows += append_parquet_records(writer, buffer, INTERLEAVE_JSONL_TAR_SCHEMA)
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
                records, num_groups, skipped_no, skipped_missing_local = _scan_jsonl_scope(
                    scope_key,
                    scope_jsonl_paths,
                    document_format=document_format,
                    document_field=document_field,
                    local_image_prefixes=local_image_prefixes,
                    tar_pattern=tar_pattern,
                )
                _offset_group_ids(records, next_group_id)
                next_group_id += num_groups
                total_groups += num_groups
                skipped_no_images += skipped_no
                skipped_missing += skipped_missing_local
                if records:
                    buffer.extend(records)
                    if len(buffer) >= _WRITE_BUFFER_ROWS:
                        flush()
                if scope_idx % 100 == 0 or scope_idx == len(scope_items):
                    logger.info(
                        "Interleave scan progress: %d/%d scopes, %d groups, %d rows",
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
                    document_format=document_format,
                    document_field=document_field,
                    local_image_prefixes=local_image_prefixes,
                    tar_pattern=tar_pattern,
                )

            def _emit(_idx, result):
                nonlocal next_group_id, total_groups, skipped_no_images, skipped_missing
                records, num_groups, skipped_no, skipped_missing_local = result
                _offset_group_ids(records, next_group_id)
                next_group_id += num_groups
                total_groups += num_groups
                skipped_no_images += skipped_no
                skipped_missing += skipped_missing_local
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
                    "Interleave scan progress: %d/%d scopes, %d groups, %d rows",
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
    write_scan_metadata(
        output_path,
        num_workers=num_workers,
        elapsed_seconds=_timer.elapsed,
        total_rows=total_rows,
        dataset_type="jsonl_tar_interleave",
        extra={
            "total_groups": total_groups,
            "skipped_no_images": skipped_no_images,
            "skipped_missing": skipped_missing,
            "document_format": document_format,
        },
    )

    logger.info(
        "Interleave scan complete: %d groups, %d rows in %.1fs with %d workers",
        total_groups,
        total_rows,
        _timer.elapsed,
        num_workers,
    )
    return str(output_path)
