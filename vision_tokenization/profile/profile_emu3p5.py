#!/usr/bin/env python3
"""Profile Emu3.5 vision tokenizer: kernel breakdown + throughput sweep.

Usage:
    # Submit as Slurm job (recommended — clean isolated GPU)
    sbatch vision_tokenization/profile/run_profile.slurm

    # Quick interactive run (stdout only)
    CUDA_VISIBLE_DEVICES=0 python -m vision_tokenization.profile.profile_emu3p5

    # Full nsys profile with GPU timeline
    CUDA_VISIBLE_DEVICES=0 nsys profile \
        --trace=cuda,nvtx --force-overwrite=true \
        -o vision_tokenization/profile/emu3p5_report \
        python -m vision_tokenization.profile.profile_emu3p5

    # Inspect results
    nsys stats vision_tokenization/profile/emu3p5_report.nsys-rep
"""

import gc
import sys
import time

import torch
import torch.cuda.nvtx as nvtx
from PIL import Image

sys.path.insert(0, ".")
sys.path.insert(0, "Tokenizer")

from vision_tokenization.vokenizers.emu import create_tokenizer

TOKENIZER_PATH = "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5"
MIN_PIXELS = 128 * 128
MAX_PIXELS = 1400 * 1400
SPATIAL_FACTOR = 16
WARMUP_ITERS = 3
BENCH_ITERS = 10


def make_batch(batch_size, h, w):
    return [Image.new("RGB", (w, h), (i % 256, 100, 50)) for i in range(batch_size)]


def profile_kernel_breakdown(tok, batch_size=16, h=512, w=512):
    """Profile individual stages of tokenize_images with NVTX markers."""
    print(f"\n{'='*70}")
    print(f"KERNEL BREAKDOWN: batch_size={batch_size}, resize={h}x{w}")
    print(f"{'='*70}")

    images = make_batch(batch_size, h, w)

    # Warmup
    for _ in range(WARMUP_ITERS):
        _ = tok.tokenize_images(images, resize_size=(h, w))
    torch.cuda.synchronize()

    timings = {}

    # Stage 1: preprocess_batch
    torch.cuda.synchronize()
    start = time.perf_counter()
    nvtx.range_push("preprocess_batch")
    img_tensors = tok.image_tokenizer.preprocess_batch(images, (h, w))
    torch.cuda.synchronize()
    nvtx.range_pop()
    timings["preprocess_batch"] = time.perf_counter() - start

    # Stage 2: encode
    torch.cuda.synchronize()
    start = time.perf_counter()
    nvtx.range_push("encode")
    with torch.inference_mode():
        indices, _ = tok.image_tokenizer.encode(img_tensors)
    torch.cuda.synchronize()
    nvtx.range_pop()
    timings["encode"] = time.perf_counter() - start

    # Stage 3: encapsulate_batch
    batch_size_actual, height, width = indices.shape
    image_indices = indices.flatten(start_dim=1)

    torch.cuda.synchronize()
    start = time.perf_counter()
    nvtx.range_push("encapsulate_batch")
    with torch.inference_mode():
        result = tok.encapsulate_batch(image_indices, height, width)
    torch.cuda.synchronize()
    nvtx.range_pop()
    timings["encapsulate_batch"] = time.perf_counter() - start

    total = sum(timings.values())
    print(f"\n{'Stage':<25} {'Time (ms)':>12} {'% of total':>12}")
    print("-" * 50)
    for stage, t in timings.items():
        print(f"{stage:<25} {t*1000:>12.2f} {t/total*100:>11.1f}%")
    print("-" * 50)
    print(f"{'TOTAL':<25} {total*1000:>12.2f} {'100.0%':>12}")

    tokens_per_img = (h // SPATIAL_FACTOR) * (w // SPATIAL_FACTOR)
    print(f"\nTokens/image: {tokens_per_img:,}  |  Total tokens: {tokens_per_img * batch_size_actual:,}")

    del img_tensors, indices, image_indices, result
    torch.cuda.empty_cache()
    gc.collect()

    return timings


def _run_timed(tok, images, resize_size, n_iters):
    """Run tokenize_images n_iters times, return list of per-iter seconds or None on OOM."""
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        try:
            result = tok.tokenize_images(images, resize_size=resize_size)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            del result
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            return None
    return times


def profile_throughput(tok):
    """Sweep batch sizes and resolutions, measure images/sec and tokens/sec."""
    print(f"\n{'='*70}")
    print("THROUGHPUT SWEEP")
    print(f"{'='*70}")

    # --- Sweep 1: fixed resolution, variable batch size ---
    h, w = 512, 512
    tpi = (h // SPATIAL_FACTOR) * (w // SPATIAL_FACTOR)

    print(f"\n--- Fixed resolution {h}x{w} ({tpi:,} tokens/img), variable batch size ---")
    print(f"{'batch_size':>10} {'total_tok':>12} "
          f"{'time_ms':>10} {'img/sec':>10} {'tok/sec':>12} {'peak_MB':>10}")
    print("-" * 70)

    for bs in [1, 2, 4, 8, 16, 32, 64]:
        images = make_batch(bs, h, w)

        # Warmup
        for _ in range(WARMUP_ITERS):
            _run_timed(tok, images, (h, w), 1)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        nvtx.range_push(f"sweep_bs{bs}_{h}x{w}")
        times = _run_timed(tok, images, (h, w), BENCH_ITERS)
        nvtx.range_pop()

        if times is None:
            print(f"{bs:>10} {bs*tpi:>12,} {'OOM':>10}")
            break

        avg_t = sum(times) / len(times)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"{bs:>10} {bs*tpi:>12,} "
              f"{avg_t*1000:>10.1f} {bs/avg_t:>10.1f} {bs*tpi/avg_t:>12,.0f} {peak_mb:>10,.0f}")

        torch.cuda.empty_cache()
        gc.collect()

    # --- Sweep 2: fixed batch size 16, variable resolution ---
    bs = 16
    print(f"\n--- Fixed batch_size={bs}, variable resolution ---")
    print(f"{'resolution':>12} {'tok/img':>10} {'total_tok':>12} "
          f"{'time_ms':>10} {'img/sec':>10} {'tok/sec':>12} {'peak_MB':>10}")
    print("-" * 82)

    for h, w in [(128, 128), (256, 256), (384, 384), (512, 512), (640, 640),
                  (768, 768), (1024, 1024), (1280, 1280), (1408, 1408)]:
        tpi = (h // SPATIAL_FACTOR) * (w // SPATIAL_FACTOR)
        images = make_batch(bs, h, w)

        for _ in range(WARMUP_ITERS):
            _run_timed(tok, images, (h, w), 1)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        nvtx.range_push(f"sweep_{h}x{w}_bs{bs}")
        times = _run_timed(tok, images, (h, w), BENCH_ITERS)
        nvtx.range_pop()

        if times is None:
            print(f"{h}x{w:>5} {tpi:>10,} {bs*tpi:>12,} {'OOM':>10}")
            break

        avg_t = sum(times) / len(times)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"{h}x{w:>5} {tpi:>10,} {bs*tpi:>12,} "
              f"{avg_t*1000:>10.1f} {bs/avg_t:>10.1f} {bs*tpi/avg_t:>12,.0f} {peak_mb:>10,.0f}")

        torch.cuda.empty_cache()
        gc.collect()


def main():
    print("Loading Emu3.5 tokenizer...")
    tok = create_tokenizer(
        mode="image_only",
        text_tokenizer_path=TOKENIZER_PATH,
        device="cuda",
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    print("Tokenizer loaded.\n")

    # Kernel breakdown at a few operating points
    profile_kernel_breakdown(tok, batch_size=8, h=256, w=256)
    profile_kernel_breakdown(tok, batch_size=16, h=512, w=512)
    profile_kernel_breakdown(tok, batch_size=32, h=512, w=512)

    # Throughput sweep
    profile_throughput(tok)

    print("\nDone.")


if __name__ == "__main__":
    main()
