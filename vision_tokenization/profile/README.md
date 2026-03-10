# Emu3.5 Vision Tokenizer Profiling

Profiling results for the **Emu3.5 VQ encoder** (`Emu3_5_IBQ`, 455M params) on a single
NVIDIA GH200 120GB GPU. The encoder maps images to discrete tokens with `spatial_factor=16`,
producing `(H/16) x (W/16)` tokens per image. Codebook size is 131,072.

Profiling script: [`profile_emu3p5.py`](profile_emu3p5.py)
Slurm launcher: [`run_profile.slurm`](run_profile.slurm)
nsys reports: `reports/emu3p5_*.nsys-rep`

---

## Stage Breakdown

Three stages per batch: **preprocess** (PIL resize + normalize), **encode** (VQ forward pass),
**encapsulate** (wrap indices with BOI/EOI tokens).

| Config | Preprocess | Encode | Encapsulate | Total |
|--------|-----------|--------|-------------|-------|
| bs=8, 256x256 | 20.1 ms (8.1%) | 228.3 ms (91.8%) | 0.4 ms (0.2%) | 248.9 ms |
| bs=16, 512x512 | 57.9 ms (7.5%) | 708.3 ms (92.3%) | 1.3 ms (0.2%) | 767.5 ms |
| bs=32, 512x512 | 112.3 ms (8.2%) | 1257.8 ms (91.8%) | 0.5 ms (0.0%) | 1370.6 ms |

**Encode dominates at ~92%** of wall time. This is expected and desirable: the GPU is spending
its time on the VQ encoder forward pass (the actual useful work), not on data movement
or pre/post-processing overhead.

---

## Hottest GPU Kernels

From nsys `cuda_gpu_kern_sum`:

| Kernel | GPU Time % | Notes |
|--------|-----------|-------|
| `cunn_SpatialSoftMaxForward<float>` (int) | 14.5% | Float32 softmax, batch dim int |
| `cunn_SpatialSoftMaxForward<float>` (long) | 11.7% | Float32 softmax, batch dim long |
| **SpatialSoftMax total** | **~26%** | Architectural, runs in fp32 |
| `implicit_gemm_f32_tf32` (fprop, 256x128) | 16.0% | Conv2d forward via TF32 GEMM |
| `implicit_gemm_f32_tf32` (fprop, 256x128 v2) | 8.2% | Conv2d forward via TF32 GEMM |
| Elementwise / vectorized ops | ~25% | Activations, norms, etc. |
| Layout transforms (NCHW/NHWC) | ~9% | Data format conversion |

`SpatialSoftMax` is the single hottest kernel (~26% of GPU time). It runs entirely in float32
as required by the VQ architecture. This is not a bottleneck to optimize away: it is the
core quantization mechanism of Emu3.5.

---

## Throughput Sweep

### Batch Size Sweep (fixed 512x512, 1,024 tokens/image)

| Batch Size | Tokens | Time (ms) | img/sec | tok/sec | Peak VRAM (MB) |
|-----------|--------|-----------|---------|---------|----------------|
| 1 | 1,024 | 210.7 | 4.7 | 4,860 | 4,334 |
| 2 | 2,048 | 250.2 | 8.0 | 8,185 | 6,902 |
| 4 | 4,096 | 321.4 | 12.4 | 12,743 | 12,034 |
| 8 | 8,192 | 467.5 | 17.1 | 17,525 | 22,298 |
| 16 | 16,384 | 769.0 | 20.8 | 21,306 | 42,826 |
| 32 | 32,768 | 1377.8 | 23.2 | 23,783 | 83,882 |
| 64 | 65,536 | OOM | - | - | >97,871 |

Throughput plateaus around **bs=32** (~23.8K tok/sec). Going from bs=16 to bs=32 gives only
+11.6% throughput for +96% memory. bs=64 causes OOM.

### Resolution Sweep (fixed batch_size=16)

| Resolution | Tokens/img | Total Tokens | Time (ms) | img/sec | tok/sec | Peak VRAM (MB) |
|-----------|-----------|-------------|-----------|---------|---------|----------------|
| 128x128 | 64 | 1,024 | 80.7 | 198.3 | 12,689 | 4,336 |
| 256x256 | 256 | 4,096 | 333.3 | 48.0 | 12,291 | 12,034 |
| 384x384 | 576 | 9,216 | 519.9 | 30.8 | 17,726 | 24,864 |
| 512x512 | 1,024 | 16,384 | 766.3 | 20.9 | 21,382 | 42,826 |
| 640x640 | 1,600 | 25,600 | 1081.7 | 14.8 | 23,667 | 65,920 |
| 768x768 | 2,304 | 36,864 | OOM | - | - | >97,871 |

Token throughput increases with resolution up to 640x640, then OOMs at 768x768.
The encoder is more efficient with larger spatial inputs (more work per kernel launch).

---

## OOM Boundaries (GH200 120GB)

| Config | Total Tokens | Peak VRAM | Status |
|--------|-------------|-----------|--------|
| bs=32 @ 512x512 | 32,768 | 83,882 MB | OK (86% VRAM) |
| bs=64 @ 512x512 | 65,536 | - | OOM |
| bs=16 @ 640x640 | 25,600 | 65,920 MB | OK (67% VRAM) |
| bs=16 @ 768x768 | 36,864 | - | OOM |

---

## Recommended Batch Settings

Based on the profiling results, the production batch planner uses:

```yaml
max_batch_tokens: 25600    # token budget per batch
batch_size: 32             # max samples per batch
```

**Why 25,600 tokens?**
- Matches bs=16 @ 640x640 (25,600 tokens) which achieves **23,667 tok/sec** — within 0.5%
  of the peak throughput (23,783 tok/sec at bs=32 @ 512x512).
- Uses 65,920 MB peak VRAM = **67% of GH200 capacity**, leaving ~30% headroom for
  variable-resolution batches and mixed-mode (image+text) pipelines.
- Well below both OOM boundaries (32K tokens at bs=64, 36K tokens at bs=16@768).

**Why batch_size=32?**
- Throughput plateaus at bs=32 (only +11.6% over bs=16 at 512x512).
- Acts as a sample cap to prevent too many small images from filling a batch, which would
  cause high preprocessing overhead relative to encode time.
- Both constraints are enforced simultaneously: whichever is hit first limits the batch.

---

## Reproducing

```bash
# Submit profiling job
sbatch vision_tokenization/profile/run_profile.slurm

# Inspect nsys report
nsys stats vision_tokenization/profile/reports/emu3p5_<JOBID>.nsys-rep

# Open in Nsight Systems GUI (download .nsys-rep to local machine)
nsys-ui vision_tokenization/profile/reports/emu3p5_<JOBID>.nsys-rep
```
