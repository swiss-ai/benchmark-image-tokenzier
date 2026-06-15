"""Throughput + token-parity experiment for Emu3.5 IBQ encode precision modes.

Arms:
- fp32-strict: TF32 disabled for matmul + cudnn (pure FP32 CUDA cores).
- tf32:        TF32 enabled (the NGC container default == production baseline).
- bf16:        TF32 enabled + bf16 autocast around encode.

Parity is measured as exact position-wise token agreement against the tf32
arm, because production tokens were generated under the container's TF32
default. Run on a GH200 node::

    python -m vision_tokenization.profile.precision_parity
"""

import time

import numpy as np
import torch

MODEL_PATH = "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Emu3.5-VisionTokenizer"
MANIFEST = "/capstor/store/cscs/swissai/infra01/vision-datasets/manifest/docci/manifest.parquet"
INPUT_PATTERN = (
    "/capstor/store/cscs/swissai/infra01/vision-datasets/raw/stage2/"
    "hf___google___docci/docci-train.arrow"
)

# (resize_h, resize_w, images_per_encode, n_chunks) — ~4M px per encode call,
# matching production's max_encode_pixels chunking scale.
BUCKETS = [
    (512, 512, 16, 4),    # small latent: hw=1024/img, 16,384 positions/encode
    (1344, 1344, 2, 4),   # large latent: hw=7056/img, 14,112 positions/encode
]

TIMED_REPS = 6
WARMUP_REPS = 2


def set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def encode_arm(tok, tensors, arm):
    """Encode every chunk under one precision arm; return (indices, seconds, peak_gb)."""
    set_tf32(arm != "fp32-strict")
    torch.cuda.reset_peak_memory_stats()
    out = []
    # Parity pass (untimed)
    for t in tensors:
        if arm == "bf16":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ind, _ = tok.encode(t)
        else:
            ind, _ = tok.encode(t)
        out.append(ind.cpu())
    # Timed pass
    for _ in range(WARMUP_REPS):
        if arm == "bf16":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok.encode(tensors[0])
        else:
            tok.encode(tensors[0])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(TIMED_REPS):
        t = tensors[i % len(tensors)]
        if arm == "bf16":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok.encode(t)
        else:
            tok.encode(t)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    return out, elapsed, peak_gb


def main() -> None:
    from Tokenizer.Emu3_5_IBQ import Emu3_5_IBQ
    from vision_tokenization.pipeline.runtime.data import HFImageLoader

    print(f"torch {torch.__version__}, device: {torch.cuda.get_device_name(0)}")
    print(f"container TF32 defaults: matmul={torch.backends.cuda.matmul.allow_tf32} "
          f"cudnn={torch.backends.cudnn.allow_tf32}")

    loader = HFImageLoader(
        input_pattern=INPUT_PATTERN, manifest_path=MANIFEST, image_column="image",
    )
    n_needed = max(ipe * nc for _, _, ipe, nc in BUCKETS)
    images, _ = loader.load_batch(np.arange(n_needed, dtype=np.int64))
    loader.close()
    images = [im for im in images if im is not None]
    print(f"loaded {len(images)} docci images")

    tok = Emu3_5_IBQ(model_path=MODEL_PATH, verbose=False)

    for rh, rw, per_encode, n_chunks in BUCKETS:
        tensors = [
            tok.preprocess_batch(images[i * per_encode:(i + 1) * per_encode], (rh, rw))
            for i in range(n_chunks)
        ]
        tokens_per_encode = per_encode * (rh // 16) * (rw // 16)
        print(f"\n=== bucket {rh}x{rw} x{per_encode} imgs/encode "
              f"({tokens_per_encode:,} tokens/encode, {n_chunks} chunks) ===")

        results = {}
        for arm in ("tf32", "fp32-strict", "bf16"):
            inds, secs, peak = encode_arm(tok, tensors, arm)
            tps = tokens_per_encode * TIMED_REPS / secs
            results[arm] = inds
            print(f"{arm:>11}: {secs / TIMED_REPS * 1000:8.1f} ms/encode  "
                  f"{tps:12,.0f} tok/s  peak {peak:6.2f} GB")

        base = results["tf32"]
        total = sum(b.numel() for b in base)
        for arm in ("fp32-strict", "bf16"):
            mism = sum(
                int((a != b).sum()) for a, b in zip(results[arm], base)
            )
            print(f"parity {arm:>11} vs tf32: {mism:,}/{total:,} mismatches "
                  f"({mism / total * 100:.4f}%)")


if __name__ == "__main__":
    main()
