"""Parallel WDS tar scanner — discovers shards, scans in parallel, writes manifest."""

import glob
import logging
import os
from collections import Counter
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


def _sample_group_sizes(records: list[dict]) -> Counter:
    """Count images per logical sample across all scanned tar members."""
    return Counter((rec["tar_path"], rec["sample_key"]) for rec in records)


def scan_wds_dataset(
    input_pattern: str,
    output_manifest: Union[str, Path],
    num_workers: int = 64,
    image_extensions: Optional[FrozenSet[str]] = None,
    text_extensions: Optional[FrozenSet[str]] = None,
    image_field_pattern: Optional[str] = None,
    multi_image: bool = False,
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
            ``"img"``). Used to normalize image stems like
            ``sample.img1.jpg`` to ``sample`` when matching text sidecars.
        multi_image: When true, emit grouped rows with ``group_id`` and
            ``image_index``. When false, write a plain single-image manifest
            even if ``image_field_pattern`` is set.

    Returns:
        The output manifest path as a string.
    """
    if image_extensions is None:
        image_extensions = DEFAULT_IMAGE_EXTENSIONS
    if multi_image and image_field_pattern is None:
        raise ValueError(
            "scan_wds_dataset(..., multi_image=True) requires image_field_pattern."
        )

    tar_paths = _discover_shards(input_pattern)
    include_text = text_extensions is not None
    logger.info(
        f"Scanning {len(tar_paths)} tar files with {num_workers} workers"
        f"{' (with text sidecars)' if include_text else ''}"
        f"{f' (field pattern: {image_field_pattern}*)' if image_field_pattern is not None else ''}"
        f"{' (grouped multi-image)' if multi_image else ''}..."
    )

    all_records: list = []
    completed = 0
    failed_tars: list = []

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(
                scan_single_tar, tp, image_extensions, text_extensions,
                image_field_pattern, multi_image,
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

    group_sizes = _sample_group_sizes(all_records)
    if not multi_image and image_field_pattern is not None:
        oversized = [(tar_path, sample_key, size) for (tar_path, sample_key), size in group_sizes.items() if size > 1]
        if oversized:
            tar_path, sample_key, size = oversized[0]
            raise ValueError(
                "scan_wds_dataset(..., multi_image=False) found a sample with multiple images "
                f"after normalizing {image_field_pattern!r}: sample_key={sample_key!r}, "
                f"tar_path={tar_path!r}, images={size}. Set multi_image=true for grouped output."
            )

    # For multi-image manifests, reassign global group_ids.
    # Workers produce local group_ids per tar; we need globally unique ones.
    if multi_image and all_records:
        if group_sizes and all(size == 1 for size in group_sizes.values()):
            logger.warning(
                "scan_wds_dataset(..., multi_image=True) found only singleton groups after "
                "parsing with image_field_pattern=%r. Consider multi_image=false.",
                image_field_pattern,
            )
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

    if multi_image:
        from vision_tokenization.indexing.manifest import (
            WDS_SCHEMA_MULTI_IMAGE,
            WDS_SCHEMA_MULTI_IMAGE_WITH_TEXT,
        )
        schema = WDS_SCHEMA_MULTI_IMAGE_WITH_TEXT if include_text else WDS_SCHEMA_MULTI_IMAGE
        return save_wds_manifest(all_records, output_manifest, include_text=include_text, schema=schema)
    return save_wds_manifest(all_records, output_manifest, include_text=include_text)
