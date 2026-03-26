#!/usr/bin/env python3
"""Compare batch plan I/O locality metrics.

Usage:
    # Analyse a single plan
    python scripts/compare_batch_plans.py \
        --manifest /path/to/manifest.parquet \
        --plan /path/to/plan.pt

    # Compare old vs new
    python scripts/compare_batch_plans.py \
        --manifest /path/to/manifest.parquet \
        --plan /path/to/old_plan.pt \
        --plan2 /path/to/new_plan.pt

    # Generate a locality plan and compare
    python scripts/compare_batch_plans.py \
        --manifest /path/to/manifest.parquet \
        --plan /path/to/old_plan.pt \
        --generate-locality \
        --tile-chunks 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


def load_plan(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def analyse_plan(plan, shard_paths, chunk_indices, num_workers=800, prefetch_window=32):
    """Compute I/O locality metrics for a batch plan."""
    splits = plan.split_for_workers(num_workers)
    rank0 = splits[0]

    if not rank0:
        print("  Rank 0 has no batches!")
        return

    n_batches = min(len(rank0), 200)
    shard_counts = []
    chunk_counts = []
    rows_per_chunk = []
    idx_spans = []

    for b in rank0[:n_batches]:
        idx = b.sample_indices
        chunk_map = {}
        shard_set = set()
        for si in idx:
            si = int(si)
            sp = shard_paths[si].as_py()
            ci = int(chunk_indices[si])
            shard_set.add(sp)
            key = (sp, ci)
            chunk_map[key] = chunk_map.get(key, 0) + 1

        shard_counts.append(len(shard_set))
        chunk_counts.append(len(chunk_map))
        idx_spans.append(int(idx.max()) - int(idx.min()))
        for count in chunk_map.values():
            rows_per_chunk.append(count)

    sc = np.array(shard_counts)
    cc = np.array(chunk_counts)
    rpc = np.array(rows_per_chunk)
    spans = np.array(idx_spans)

    print(f"  Rank 0: {len(rank0)} batches (analysing first {n_batches})")
    print(f"  Shards/batch:  mean={sc.mean():.1f}  p50={np.median(sc):.0f}  p95={np.percentile(sc, 95):.0f}  max={sc.max()}")
    print(f"  Chunks/batch:  mean={cc.mean():.1f}  p50={np.median(cc):.0f}  p95={np.percentile(cc, 95):.0f}  max={cc.max()}")
    print(f"  Rows/chunk:    mean={rpc.mean():.1f}  p50={np.median(rpc):.0f}  max={rpc.max()}")
    print(f"  Read amplif:   {cc.mean() * 100 / np.mean([len(b.sample_indices) for b in rank0[:n_batches]]):.1f}x (chunk=100 rows)")
    print(f"  Index span:    mean={spans.mean():.0f}  p50={np.median(spans):.0f}  p95={np.percentile(spans, 95):.0f}")

    # Concurrent chunks in prefetch windows
    concurrent = []
    for i in range(0, min(n_batches, 200), prefetch_window):
        window = rank0[i : i + prefetch_window]
        all_chunks = set()
        for b in window:
            for si in b.sample_indices:
                si = int(si)
                all_chunks.add((shard_paths[si].as_py(), int(chunk_indices[si])))
        concurrent.append(len(all_chunks))

    conc = np.array(concurrent)
    print(f"  Concurrent chunks ({prefetch_window}-batch window): mean={conc.mean():.0f}  max={conc.max()}")

    # Batch size distribution
    sizes = np.array([len(b.sample_indices) for b in rank0[:n_batches]])
    print(f"  Batch size:    mean={sizes.mean():.1f}  p50={np.median(sizes):.0f}  min={sizes.min()}  max={sizes.max()}")


def main():
    parser = argparse.ArgumentParser(description="Compare batch plan I/O locality")
    parser.add_argument("--manifest", required=True, help="Path to manifest parquet")
    parser.add_argument("--plan", required=True, help="Path to batch plan .pt")
    parser.add_argument("--plan2", help="Path to second plan .pt for comparison")
    parser.add_argument("--generate-locality", action="store_true",
                        help="Generate a locality plan from manifest and compare")
    parser.add_argument("--window-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=800)
    parser.add_argument("--prefetch-window", type=int, default=32)
    # Planning params (for --generate-locality)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--spatial-factor", type=int, default=16)
    parser.add_argument("--min-pixels", type=int, default=8192)
    parser.add_argument("--max-pixels", type=int, default=4194304)
    parser.add_argument("--resize-min-pixels", type=int, default=16384)
    parser.add_argument("--resize-max-pixels", type=int, default=1960000)
    parser.add_argument("--save-locality-plan", help="Save generated locality plan to this path")
    args = parser.parse_args()

    print(f"Loading manifest: {args.manifest}")
    manifest = pq.read_table(args.manifest, columns=["shard_path", "chunk_index"])
    shard_paths = manifest.column("shard_path")
    chunk_indices = manifest.column("chunk_index").to_numpy()
    print(f"  {len(manifest)} rows, {len(set(chunk_indices))} chunks")

    print(f"\n=== Plan 1: {args.plan} ===")
    plan1 = load_plan(args.plan)
    print(f"  {len(plan1.batches)} batches, {plan1.total_samples} samples, {plan1.total_filtered} filtered")
    analyse_plan(plan1, shard_paths, chunk_indices, args.num_workers, args.prefetch_window)

    plan2 = None
    plan2_label = None

    if args.plan2:
        plan2_label = args.plan2
        plan2 = load_plan(args.plan2)
    elif args.generate_locality:
        print(f"\nGenerating locality plan (window_size={args.window_size})...")
        from vision_tokenization.indexing.planning.batch_planner import plan_locality_batches
        plan2 = plan_locality_batches(
            manifest_path=args.manifest,
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
            spatial_factor=args.spatial_factor,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            resize_min_pixels=args.resize_min_pixels,
            resize_max_pixels=args.resize_max_pixels,
            window_size=args.window_size,
        )
        plan2_label = f"locality (window_size={args.window_size})"
        if args.save_locality_plan:
            Path(args.save_locality_plan).parent.mkdir(parents=True, exist_ok=True)
            torch.save(plan2, args.save_locality_plan)
            print(f"  Saved to {args.save_locality_plan}")

    if plan2 is not None:
        print(f"\n=== Plan 2: {plan2_label} ===")
        print(f"  {len(plan2.batches)} batches, {plan2.total_samples} samples, {plan2.total_filtered} filtered")
        analyse_plan(plan2, shard_paths, chunk_indices, args.num_workers, args.prefetch_window)


if __name__ == "__main__":
    main()
