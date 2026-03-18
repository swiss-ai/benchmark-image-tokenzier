#!/usr/bin/env python3
"""Parallel scan of LLaVA-OneVision 85M mid-train arrow shards.

Extracts (width, height) from image bytes headers using imagesize.
Writes HF manifest Parquet, then runs batch planner for statistics.
"""

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import imagesize
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

ARROW_DIR = Path(
    "/capstor/store/cscs/swissai/infra01/vision-datasets/hf_datasets_cache/"
    "mvp-lab___l_la_va-one_vision-1.5-mid-training-85_m/default/0.0.0/"
    "c5218cad785eba7d218137e8ce4997bda568a050"
)
OUTPUT_DIR = Path(
    "/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scratch/indexing_test"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUTPUT_DIR / "llava85m_full_manifest.parquet"

NUM_WORKERS = 200


# --- Top-level function (pickleable for ProcessPoolExecutor) ---------------

def scan_arrow_shard(shard_path: str):
    """Scan a single arrow shard, return list of (global_row_idx_placeholder, w, h).

    We use a placeholder for the row index — the caller assigns global indices
    after collecting all results in shard order.
    """
    import pyarrow as pa

    records = []
    failed = 0

    with open(shard_path, "rb") as f:
        reader = pa.ipc.open_stream(f)
        for batch in reader:
            image_col = batch.column("image")
            for row_i in range(batch.num_rows):
                img_struct = image_col[row_i].as_py()
                img_bytes = img_struct["bytes"]
                if img_bytes is None:
                    failed += 1
                    records.append(None)
                    continue

                header = img_bytes[:4096]
                w, h = imagesize.get(BytesIO(header))
                if w < 0 or h < 0:
                    w, h = imagesize.get(BytesIO(img_bytes))
                if w < 0 or h < 0:
                    failed += 1
                    records.append(None)
                    continue

                records.append((w, h))

    return records, failed


def main():
    t_start = time.time()

    # --- Discover shards ---
    shard_files = sorted(ARROW_DIR.glob("*.arrow"))
    logger.info(f"Found {len(shard_files):,} arrow shards in {ARROW_DIR}")

    # --- Parallel scan ---
    logger.info(f"Scanning with {NUM_WORKERS} workers...")
    shard_results = [None] * len(shard_files)  # preserve order
    total_failed = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        future_to_idx = {
            pool.submit(scan_arrow_shard, str(sf)): i
            for i, sf in enumerate(shard_files)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                records, failed = future.result()
                shard_results[idx] = records
                total_failed += failed
            except Exception:
                logger.exception(f"Failed shard {idx}: {shard_files[idx]}")
                shard_results[idx] = []

            completed += 1
            if completed % 500 == 0 or completed == len(shard_files):
                elapsed = time.time() - t_start
                rate = completed / elapsed
                eta = (len(shard_files) - completed) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {completed:,}/{len(shard_files):,} shards "
                    f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)"
                )

    scan_time = time.time() - t_start
    logger.info(f"Scan done in {scan_time:.1f}s")

    # --- Build manifest records (assign global indices in shard order) ---
    logger.info("Building manifest...")
    manifest_records = []
    global_idx = 0
    for shard_recs in shard_results:
        if shard_recs is None:
            continue
        for entry in shard_recs:
            if entry is not None:
                w, h = entry
                manifest_records.append(
                    {"sample_index": global_idx, "width": w, "height": h}
                )
            global_idx += 1

    total_scanned = global_idx
    total_valid = len(manifest_records)
    logger.info(
        f"Total scanned: {total_scanned:,}, valid: {total_valid:,}, "
        f"failed: {total_failed:,}"
    )

    # --- Save manifest ---
    from vision_tokenization.indexing.manifest import save_hf_manifest, load_resolution_arrays

    save_hf_manifest(manifest_records, str(MANIFEST_PATH))

    # --- Statistics ---
    logger.info("=== Resolution Statistics ===")
    widths, heights = load_resolution_arrays(MANIFEST_PATH)
    pixels = widths.astype(np.int64) * heights.astype(np.int64)
    ars = widths.astype(np.float64) / heights.astype(np.float64)

    logger.info(f"Samples:       {len(widths):,}")
    logger.info(f"Width:         min={widths.min()}, max={widths.max()}, "
                f"mean={widths.mean():.0f}, median={np.median(widths):.0f}")
    logger.info(f"Height:        min={heights.min()}, max={heights.max()}, "
                f"mean={heights.mean():.0f}, median={np.median(heights):.0f}")
    logger.info(f"Pixels:        min={pixels.min():,}, max={pixels.max():,}, "
                f"mean={pixels.mean():,.0f}, median={np.median(pixels):,.0f}")
    logger.info(f"Aspect ratio:  min={ars.min():.3f}, max={ars.max():.3f}, "
                f"mean={ars.mean():.3f}, median={np.median(ars):.3f}")

    # Pixel-count distribution buckets
    thresholds = [
        ("< 64*128 (too small)", 0, 64 * 128),
        ("64*128 – 256*256", 64 * 128, 256 * 256),
        ("256*256 – 512*512", 256 * 256, 512 * 512),
        ("512*512 – 1024*1024", 512 * 512, 1024 * 1024),
        ("1024*1024 – 2048*2048", 1024 * 1024, 2048 * 2048),
        ("> 2048*2048 (very large)", 2048 * 2048, int(1e18)),
    ]
    logger.info("--- Pixel count distribution ---")
    for label, lo, hi in thresholds:
        count = int(np.sum((pixels >= lo) & (pixels < hi)))
        pct = count / len(pixels) * 100
        logger.info(f"  {label:30s}: {count:>12,}  ({pct:5.1f}%)")

    # Aspect ratio distribution
    ar_buckets = [
        ("Ultra-portrait (< 0.5)", 0, 0.5),
        ("Portrait (0.5 – 0.8)", 0.5, 0.8),
        ("Near-square (0.8 – 1.25)", 0.8, 1.25),
        ("Landscape (1.25 – 2.0)", 1.25, 2.0),
        ("Ultra-landscape (> 2.0)", 2.0, 100),
    ]
    logger.info("--- Aspect ratio distribution ---")
    for label, lo, hi in ar_buckets:
        count = int(np.sum((ars >= lo) & (ars < hi)))
        pct = count / len(ars) * 100
        logger.info(f"  {label:30s}: {count:>12,}  ({pct:5.1f}%)")

    # --- Batch planning ---
    logger.info("=== Batch Planning ===")
    from vision_tokenization.indexing.clustered_batch_planner import plan_clustered_batches

    t0 = time.time()
    plan = plan_clustered_batches(
        manifest_path=str(MANIFEST_PATH),
        batch_size=4,
        min_pixels=64 * 128,
        max_pixels=2048 * 2048,
        num_clusters=2000,
    )
    plan_time = time.time() - t0

    logger.info(f"Planning took {plan_time:.1f}s")
    logger.info(f"Batches:        {len(plan.batches):,}")
    logger.info(f"Total samples:  {plan.total_samples:,}")
    logger.info(f"Filtered out:   {plan.total_filtered:,}")

    batch_sizes = [len(b.sample_indices) for b in plan.batches]
    logger.info(f"Batch sizes:    min={min(batch_sizes)}, max={max(batch_sizes)}, "
                f"mean={np.mean(batch_sizes):.1f}")

    # Within-batch AR std vs global
    within_stds = []
    for b in plan.batches:
        idx = b.sample_indices
        bar = widths[idx].astype(np.float64) / heights[idx].astype(np.float64)
        if len(bar) > 1:
            within_stds.append(np.std(bar))
    global_ar_std = np.std(ars)
    mean_within = np.mean(within_stds)
    logger.info(f"AR std global:  {global_ar_std:.4f}")
    logger.info(f"AR std within:  {mean_within:.4f} (mean across batches)")
    logger.info(f"AR std ratio:   {mean_within / global_ar_std:.4f}")

    # Worker split example
    for nw in [4, 8, 16, 32, 64]:
        chunks = plan.split_for_workers(nw)
        sizes = [sum(len(b.sample_indices) for b in c) for c in chunks]
        logger.info(f"  split_for_workers({nw:2d}): "
                    f"min={min(sizes):,}, max={max(sizes):,}, "
                    f"imbalance={max(sizes)/min(sizes):.3f}x")

    total_time = time.time() - t_start
    logger.info(f"=== Total wall time: {total_time:.1f}s ({total_time/60:.1f} min) ===")


if __name__ == "__main__":
    main()
