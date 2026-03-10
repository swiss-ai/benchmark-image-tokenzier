# Benchmark: Discrete Image Tokenizers

Repo supports benchmarking and large-scale tokenization with discrete image tokenizers.

**Authors:** Yixuan Xu, Raphael Krest, Nicola Irmiger

## Repository Structure

```
.
├── vision_tokenization/    # Distributed tokenization pipeline (torch.distributed + Hydra)
│   ├── configs/            # Hydra config files
│   ├── pipelines/          # Distributed pipeline (core loop, data loading, writing)
│   ├── vokenizers/         # Tokenizer wrappers (EMU image-only, SFT, image-text-pair)
│   └── indexing/           # Manifest creation, batch planning, tar reading
├── Tokenizer/              # Vision tokenizer implementations (submodules + patches)
├── benchmarks/             # Benchmarking scripts and results
│   ├── notebooks/          # Tokenizer exploration notebooks
│   ├── reconstruction/     # Image encode-decode reconstruction
│   ├── inference/          # Conditional generation and vLLM inference
│   ├── metrics/            # Quality metrics (LPIPS, PSNR, SSIM, FID)
│   ├── tiling/             # Image tiling utilities
│   └── assets/             # Reconstructed images for metric comparison
├── conftest.py             # Pytest configuration
├── pyproject.toml          # Project configuration
└── format.sh               # Auto-formatting (black, isort, flake8)
```

## Setup Autoformatting

This repo supports auto-formatting using flake8, black and isort. Submodules are ignored by default.

Install dev requirements:
```bash
pip install -r requirements-dev.txt
```
Then run formatting any time using:
```bash
./format.sh
```

## Benchmarked Tokenizers

| Model | Approach | Token Type | Training Resolution | Inference Resolution | # Tokens per Image | Codebook Size | Training Data Augmented | Image Understanding | Image Generation | Pretraining Data |
|-------|----------|------------|---------------------|---------------------|-------------------|---------------|----------------------|---------------------|------------------|---------------------------|
| **Open-MagVit2** | VQ-VAE + MLM | Spatial (2D Grid) | 256×256 | Flexible (e.g., 256×256) | 16×16 Compression | 262,144 | Unknown | ✅ | ✅ |  Imagenet2012 |
| **Emu3-VisionTokenizer** | VQ-GAN (MoVQGAN) | Spatial (2D Grid) | ≥ 512×512 | Flexible (e.g., 512×512) | 8×8 Compression | 32,768 | Unknown | ✅ | ✅ | [laion-high-resolution](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion-high-resolution.md) |
| **Cosmos** | VQ-AE (Discrete) | Spatial (2D Grid) | Flexible (256px to 4K) | Original | 16×16 or 8×8 Compression | 64,000 | Unknown | ✅ | ✅ | VIDEO: <br> Driving (11%), <br> Hand motion and object manipulation (16%), <br>Human motion and activity (10%), <br> Spatial awareness and navigation (16%), <br> First person point-of-view (8%), <br> Nature dynamics (20%), <br> Dynamic camera movements (8%), <br> Synthetically rendered (4%), <br> Others (7%) |
| **FlowMo Hi** | Diffusion Autoencoder (Transformer-based) | Sequential (1D latent) | 256×256 | 256×256 | 1,024 | 16,384 | Unknown | — | ✅ | Imagenet2012 |
| **TiTok** | 1D VQ-VAE (Transformer-based) | Sequential (1D latent) | 256×256, 512×512 | 256×256, 512×512 | 256 | 4,096 | Unknown | — | ✅ | ImageNet |
| **Selftok** | Diffusion-based AR Prior | Sequential (Autoregressive Prior) | 256×256 | 256×256 | 512 / 1,024 / 1,536 | 32,768 | Unknown | ✅ | ✅ | DataComp: 25.45%, <br> LAION-2B En: 25.36%, <br> LAION-2B Multi: 24.26%, <br> COYO-700M: 12.96%, <br> In-house T2I: 7.98%, <br> In-house Text: 4.00% |
| **UniTok** | VQ-VAE | Sequential (1D latent) | 256×256 | flexible | flexible(8 x 256 for 256 x 256) | 8 x 16000 | Unknown | ✅ | ✅ | DataComp-1B |
| **DetailFlow** | Autoregressive | Sequential (AR coarse-to-fine, next-detail-prediction) | 256×256 | 256×256 | 128 / 256 / 512 | 8,192 | Unknown | - | ✅ | ImageNet-1K |
| **TokenFlow** | VQ-VAE (Transformer-based) | Spatial (2D latent, next-scale-prediction) | 256×256 / 384x384 | 256×256 / 384x384 |  16x16 / 27x27 | 32,768 | Unknown | ✅ | ✅ | LAION and COYO-700M (no ocr data!) |
| **VILA-U** | RQ-VAE | Spatial (2D latent) | 256×256 | 256×256 |  16x16x4 | 16,384 | Unknown | ✅ | ✅ | COYO-700M |
