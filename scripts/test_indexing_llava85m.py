#!/usr/bin/env python3
"""Smoke-test the indexing pipeline on LLaVA-OneVision 85M mid-train.

Reads arrow files directly from the HF cache (bypasses builder lock).
Scans first N shards, builds a manifest, plans clustered batches.
"""

import logging
import time
from io import BytesIO
from pathlib import Path

import imagesize
import numpy as np
import pyarrow as pa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARROW_DIR = Path(
    "/capstor/store/cscs/swissai/infra01/vision-datasets/hf_datasets_cache/"
    "mvp-lab___l_la_va-one_vision-1.5-mid-training-85_m/default/0.0.0/"
    "c5218cad785eba7d218137e8ce4997bda568a050"
)
OUTPUT_DIR = Path("/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/scratch/indexing_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = OUTPUT_DIR / "llava85m_manifest.parquet"

# Number of arrow shards to scan (each ~1000 rows). Set to None for all 84911.
MAX_SHARDS = 100


def scan_arrow_shards(arrow_dir: Path, max_shards=None):
    """Scan HF arrow shards directly, extracting image dims from raw bytes."""
    shard_files = sorted(arrow_dir.glob("*.arrow"))
    if max_shards is not None:
        shard_files = shard_files[:max_shards]

    logger.info(f"Scanning {len(shard_files)} arrow shards from {arrow_dir}")

    records = []
    global_idx = 0
    failed = 0

    for shard_i, shard_path in enumerate(shard_files):
        with open(shard_path, "rb") as f:
            reader = pa.ipc.open_stream(f)
            for batch in reader:
                image_col = batch.column("image")
                for row_i in range(batch.num_rows):
                    img_struct = image_col[row_i].as_py()
                    img_bytes = img_struct["bytes"]
                    if img_bytes is None:
                        failed += 1
                        global_idx += 1
                        continue

                    # Header-only dimension read
                    header = img_bytes[:4096]
                    w, h = imagesize.get(BytesIO(header))
                    if w < 0 or h < 0:
                        # Fallback: try full bytes
                        w, h = imagesize.get(BytesIO(img_bytes))
                    if w < 0 or h < 0:
                        failed += 1
                        global_idx += 1
                        continue

                    records.append({"sample_index": global_idx, "width": w, "height": h})
                    global_idx += 1

        if (shard_i + 1) % 10 == 0 or (shard_i + 1) == len(shard_files):
            logger.info(f"  Shards: {shard_i + 1}/{len(shard_files)}, samples: {len(records):,}, failed: {failed}")

    return records, global_idx, failed


def main():
    # --- Step 1: Scan arrow files ---
    logger.info(f"=== Step 1: Scanning arrow shards (max_shards={MAX_SHARDS}) ===")
    t0 = time.time()
    records, total_scanned, total_failed = scan_arrow_shards(ARROW_DIR, max_shards=MAX_SHARDS)
    scan_time = time.time() - t0
    logger.info(f"Scan complete: {len(records):,} valid / {total_scanned:,} total / {total_failed} failed in {scan_time:.1f}s")

    # --- Step 2: Save manifest ---
    logger.info("=== Step 2: Saving HF manifest ===")
    from vision_tokenization.indexing.manifest import save_hf_manifest, load_resolution_arrays

    save_hf_manifest(records, str(MANIFEST_PATH))

    widths, heights = load_resolution_arrays(MANIFEST_PATH)
    logger.info(f"Width range: [{widths.min()}, {widths.max()}]")
    logger.info(f"Height range: [{heights.min()}, {heights.max()}]")
    pixels = widths.astype(np.int64) * heights.astype(np.int64)
    logger.info(f"Pixel count range: [{pixels.min():,}, {pixels.max():,}]")
    logger.info(f"Mean resolution: {widths.mean():.0f}x{heights.mean():.0f}")

    # --- Step 3: Plan clustered batches ---
    logger.info("=== Step 3: Planning clustered batches ===")
    from vision_tokenization.indexing.clustered_batch_planner import plan_clustered_batches

    t0 = time.time()
    plan = plan_clustered_batches(
        manifest_path=str(MANIFEST_PATH),
        batch_size=4,
        min_pixels=64 * 128,
        max_pixels=2048 * 2048,
        num_clusters=50,
    )
    plan_time = time.time() - t0
    logger.info(f"Planning took {plan_time:.3f}s")
    logger.info(f"Batches: {len(plan.batches)}")
    logger.info(f"Total samples: {plan.total_samples:,}")
    logger.info(f"Filtered out: {plan.total_filtered:,}")

    # Show example batches
    for i, batch in enumerate(plan.batches[:5]):
        bw = widths[batch.sample_indices]
        bh = heights[batch.sample_indices]
        ars = bw / bh
        logger.info(
            f"  Batch {i}: {len(batch.sample_indices)} samples, "
            f"resize=({batch.resize_height}x{batch.resize_width}), "
            f"AR range=[{ars.min():.2f}, {ars.max():.2f}]"
        )

    # --- Step 4: Worker split ---
    chunks = plan.split_for_workers(4)
    for i, chunk in enumerate(chunks):
        n_batches = len(chunk)
        n_samples = sum(len(b.sample_indices) for b in chunk)
        logger.info(f"  Worker {i}: {n_batches} batches, {n_samples} samples")

    # --- Summary ---
    logger.info("=== Summary ===")
    logger.info(f"Scan:  {scan_time:.1f}s for {total_scanned:,} samples ({total_scanned / scan_time:.0f} samples/s)")
    logger.info(f"Plan:  {plan_time:.3f}s for {plan.total_samples:,} samples")
    if MAX_SHARDS is not None:
        est_full = scan_time / MAX_SHARDS * 84911
        logger.info(f"Estimated full 85M scan time: {est_full / 60:.0f} min")


if __name__ == "__main__":
    main()
