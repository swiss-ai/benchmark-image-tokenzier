# Tokenizer Reconstruction Benchmarks

Compare discrete image tokenizers head-to-head on a fixed image set:
encode → decode → score (PSNR / SSIM / LPIPS) and inspect the
reconstructions visually side-by-side.

This directory holds the scripts, notebooks, and saved artefacts. The
distributed tokenization pipeline used to produce training data lives
in `../vision_tokenization/`; the qualitative VLM benchmarks live in
`../vision_tokenization/qualitative_benchmark/`.

## Layout

| Path | Purpose |
|---|---|
| `notebooks/` | Per-tokenizer exploration notebooks (Emu3, IBQ, FlowMo, FQGAN, OpenMAGViT2, Cosmos, TokenFlow, TiTok, UniTok, LlamaGen, Selftok, Seed, VQGAN — 14 total) |
| `reconstruction/` | Batched encode-decode runner: `batch_reconstruct.py`, `emu3_reconstruct_helper.py`, and a Slurm wrapper `run_batch_reconstruct.slurm` |
| `metrics/` | PSNR / SSIM / LPIPS / FID computation; results in `metrics_results.md` and `metrics_results.csv` |
| `inference/` | Conditional generation: `emu3_vllm_inferencer.py`, `test_conditional_generation.py` |
| `tiling/` | Tile-based encoding helpers for tokenizers with fixed input resolutions (`Tiler.py`, `molmo_tiler.py`) |
| `assets/` | Saved reconstructed images per tokenizer, organised by config and aspect ratio |

## Results at a glance

Hard numbers live in [`metrics/metrics_results.md`](metrics/metrics_results.md)
(63 tokenizer/config rows, sorted by LPIPS ascending). Top-3 by LPIPS:

| Folder | PSNR | SSIM | LPIPS | #Tokens |
|---|---:|---:|---:|---:|
| `Emu3VisionTokenizer` | 30.14 | 0.946 | **0.019** | 10,947 |
| `selftok_1024` | 28.19 | 0.957 | 0.023 | 21,299 |
| `unitok` | 26.83 | 0.948 | 0.026 | 5,325 |

## Running a reconstruction sweep

The Slurm wrapper picks one tokenizer config per job and writes
reconstructed images to `assets/<tokenizer>/`. Adapt and submit:

```bash
sbatch benchmarks/reconstruction/run_batch_reconstruct.slurm
```

Notebooks under `notebooks/` are scratchpads for iterating on a single
tokenizer — they read the same assets and metrics CSV, so you can
inspect failures visually without rerunning the whole sweep.

## Tokenizer comparison

A reference card for what each tokenizer does and where it sits in the
design space. **Resolution / token-count / codebook are the columns that
drive deployment choices**; the rest are background.

| Model | Approach | Token type | Train res. | Inf. res. | # Tokens / image | Codebook | Understanding | Generation | Pretraining data |
|---|---|---|---|---|---|---|:-:|:-:|---|
| **Open-MagVit2** | VQ-VAE + MLM | Spatial 2D | 256² | flexible | 16×16 compression | 262,144 | ✅ | ✅ | ImageNet-2012 |
| **Emu3-VisionTokenizer** | VQ-GAN (MoVQGAN) | Spatial 2D | ≥ 512² | flexible | 8×8 compression | 32,768 | ✅ | ✅ | [LAION-high-resolution](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion-high-resolution.md) |
| **Cosmos** | VQ-AE (discrete) | Spatial 2D | 256–4K | original | 16×16 or 8×8 compression | 64,000 | ✅ | ✅ | Video (driving, hand motion, navigation, …) |
| **FlowMo Hi** | Diffusion autoencoder (transformer) | Sequential 1D | 256² | 256² | 1,024 | 16,384 | — | ✅ | ImageNet-2012 |
| **TiTok** | 1D VQ-VAE (transformer) | Sequential 1D | 256² / 512² | same | 256 | 4,096 | — | ✅ | ImageNet |
| **Selftok** | Diffusion-based AR prior | Sequential AR | 256² | 256² | 512 / 1,024 / 1,536 | 32,768 | ✅ | ✅ | DataComp + LAION-2B + COYO-700M + in-house |
| **UniTok** | VQ-VAE | Sequential 1D | 256² | flexible | 8 × 256 (for 256²) | 8 × 16,000 | ✅ | ✅ | DataComp-1B |
| **DetailFlow** | Autoregressive (next-detail) | Sequential AR coarse→fine | 256² | 256² | 128 / 256 / 512 | 8,192 | — | ✅ | ImageNet-1K |
| **TokenFlow** | VQ-VAE (transformer, next-scale) | Spatial 2D | 256² / 384² | same | 16×16 / 27×27 | 32,768 | ✅ | ✅ | LAION + COYO-700M (no OCR) |
| **VILA-U** | RQ-VAE | Spatial 2D | 256² | 256² | 16×16×4 | 16,384 | ✅ | ✅ | COYO-700M |

"Training Data Augmented" is `Unknown` for every entry surveyed and was
omitted to reduce visual noise.
