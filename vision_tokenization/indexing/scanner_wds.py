"""Parallel WDS tar scanner — discovers shards, scans in parallel, writes manifest."""

import glob
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import FrozenSet, Optional, Union

from vision_tokenization.indexing._scan_worker import (
    DEFAULT_IMAGE_EXTENSIONS,
    DEFAULT_TEXT_EXTENSIONS,
    scan_single_tar,
)
from vision_tokenization.indexing.manifest import save_wds_manifest

logger = logging.getLogger(__name__)


def _discover_shards(input_pattern: str) -> list:
    """Discover tar shards from a braceexpand or glob pattern.

    Tries braceexpand first (e.g. ``data_{000..100}.tar``), then falls
    back to standard glob.
    """
    tar_paths: list = []

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
            else:
                logger.warning("Braceexpand produced paths but none exist. Falling back to glob.")
        except ImportError:
            logger.warning("braceexpand not installed, falling back to glob")
        except Exception as e:
            logger.warning(f"braceexpand failed ({e}), falling back to glob")

    tar_paths = sorted(glob.glob(input_pattern))
    if not tar_paths:
        raise FileNotFoundError(f"No tar files found matching pattern: {input_pattern}")

    logger.info(f"Glob: {len(tar_paths)} tar files found matching '{input_pattern}'")
    return tar_paths


def scan_wds_dataset(
    input_pattern: str,
    output_manifest: Union[str, Path],
    num_workers: int = 64,
    image_extensions: Optional[FrozenSet[str]] = None,
    text_extensions: Optional[FrozenSet[str]] = None,
    image_field_pattern: Optional[str] = None,
) -> str:
    """Scan all WDS tars in parallel and write a Parquet manifest.

    Args:
        input_pattern: Braceexpand or glob pattern for tar files.
        output_manifest: Path where the Parquet manifest will be written.
        num_workers: Number of parallel workers for tar scanning.
        image_extensions: Image extensions to scan for (default: common formats).
        text_extensions: Text sidecar extensions to index (e.g. ``{"json", "txt"}``).
            Pass ``None`` for image-only manifests, or ``DEFAULT_TEXT_EXTENSIONS``
            for manifests that need text loading at tokenization time.
        image_field_pattern: Prefix for multi-image field names (e.g.
            ``"img"``).  ``None`` → single-image mode (default).

    Returns:
        The output manifest path as a string.
    """
    if image_extensions is None:
        image_extensions = DEFAULT_IMAGE_EXTENSIONS

    tar_paths = _discover_shards(input_pattern)
    include_text = text_extensions is not None
    is_multi = image_field_pattern is not None
    logger.info(
        f"Scanning {len(tar_paths)} tar files with {num_workers} workers"
        f"{' (with text sidecars)' if include_text else ''}"
        f"{f' (multi-image: {image_field_pattern}*)' if is_multi else ''}..."
    )

    all_records: list = []
    completed = 0
    failed_tars: list = []

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(
                scan_single_tar, tp, image_extensions, text_extensions,
                image_field_pattern,
            ): tp
            for tp in tar_paths
        }

        for future in as_completed(futures):
            tar_path = futures[future]
            try:
                records = future.result()
                all_records.extend(records)
            except Exception:
                logger.exception(f"Failed to scan {tar_path}")
                failed_tars.append(tar_path)

            completed += 1
            if completed % 100 == 0 or completed == len(tar_paths):
                logger.info(
                    f"Progress: {completed}/{len(tar_paths)} tars scanned, "
                    f"{len(all_records):,} images found so far"
                )

    if failed_tars:
        logger.error(
            f"{len(failed_tars)}/{len(tar_paths)} tar files failed to scan: "
            f"{failed_tars[:10]}{'...' if len(failed_tars) > 10 else ''}"
        )

    # For multi-image manifests, reassign global group_ids.
    # Workers produce local group_ids per tar; we need globally unique ones.
    if is_multi and all_records:
        # Sort by (tar_path, sample_key, image_index) for determinism
        all_records.sort(
            key=lambda r: (r["tar_path"], r["sample_key"], r["image_index"])
        )
        # Assign monotonic global group_id per unique (tar_path, sample_key)
        global_gid = 0
        prev_key = None
        for rec in all_records:
            cur_key = (rec["tar_path"], rec["sample_key"])
            if cur_key != prev_key:
                if prev_key is not None:
                    global_gid += 1
                prev_key = cur_key
            rec["group_id"] = global_gid

    logger.info(f"Scan complete: {len(all_records):,} images from {len(tar_paths)} tars")

    if is_multi:
        from vision_tokenization.indexing.manifest import (
            WDS_SCHEMA_MULTI_IMAGE,
            WDS_SCHEMA_MULTI_IMAGE_WITH_TEXT,
        )
        schema = WDS_SCHEMA_MULTI_IMAGE_WITH_TEXT if include_text else WDS_SCHEMA_MULTI_IMAGE
        return save_wds_manifest(all_records, output_manifest, include_text=include_text, schema=schema)
    return save_wds_manifest(all_records, output_manifest, include_text=include_text)
