# VLM Qualitative Benchmark

A toolkit for running qualitative benchmarks on Vision-Language Models (VLMs) and exploring results through an interactive web interface.

## Overview

This toolkit consists of two main components:

1. **`vlm_benchmark.py`**: Script to run VLM inference in three modes: VLM Q&A, image completion, and captioning
2. **Webapp**: Flask-based web interface to browse, filter, and compare benchmark results

## Table of Contents

- [Installation](#installation)
- [Choosing Models and Tokenizers](#choosing-models-and-tokenizers)
- [Benchmark Modes](#benchmark-modes)
  - [VLM Q&A](#vlm-qa-default)
  - [Image Completion](#image-completion)
  - [Captioning](#captioning)
- [Command-Line Reference](#command-line-reference)
- [Results Viewer Webapp](#results-viewer-webapp)
- [Data Format](#data-format)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch with CUDA support (for VLM inference)
- Flask 3.0.0 or higher (for webapp)

### Setup

```bash
cd vision_tokenization/qualitative_benchmark

# Core dependencies
pip install torch torchvision pillow tqdm

# Transformers: >=4.56 required for Apertus/EMU3.5 support, <5 to keep EMU3 compatible
pip install "transformers>=4.56,<5.0.0"

# Inference backend (install one or both)
pip install vllm          # Fast inference (recommended)
# HuggingFace backend uses transformers directly, no extra install needed

# Perceptual metrics (needed for image completion benchmarks)
pip install lpips scikit-image

# Webapp
pip install -r webapp/requirements.txt
```

---

## Choosing Models and Tokenizers

Running a benchmark requires three components:

1. **Model** (`--model_path`): The VLM (language model with vision tokens in its vocabulary)
2. **Text tokenizer** (`--tokenizer_path`): Tokenizer matching the model's vocabulary, including vision special tokens
3. **Vision tokenizer** (`--vision-tokenizer-type`): Encodes images into discrete tokens (EMU3 or EMU3.5)

The text tokenizer must contain the vision special tokens (`<|img_start|>`, `<|visual token XXXXXX|>`, etc.) that match the vision tokenizer. Vision token mapping is auto-detected from the text tokenizer at startup -- no `vision_token_mapping.json` file is needed.
Currently we calculate the vision token range by finding the id of first vision token `<|visual token 000000|>` and then adding the vision codebook size. 
Special tokens for vision are cached separately.

### EMU3 Baseline (BAAI/Emu3-Chat)

Uses the original EMU3 vision tokenizer (MoVQGAN, 8x8 compression, 32K codebook). Works with vLLM.

**Important:** BAAI/Emu3-Chat does not implement the HuggingFace `apply_chat_template` protocol. You **must** use `--prompt-builder emu3` so the prompt is formatted with the correct `You are a helpful assistant. USER: {image}{prompt} ASSISTANT:` pattern that Emu3 expects.

```bash
sbatch vision_tokenization/scripts/run_qualitative_benchmarks.sh 
    --experiment_name qa_emu3_baseline 
    --model_path BAAI/Emu3-Chat 
    --tokenizer_path BAAI/Emu3-Chat 
    --vision-tokenizer-type emu3 
    --prompt-builder emu3
```

Or with a locally-trained llama3+emu3 model (these typically have a proper chat template, so `--prompt-builder` is not needed):

```bash
sbatch vision_tokenization/scripts/run_qualitative_benchmarks.sh \
    --experiment_name qa_llama3_emu3 \
    --model_path /path/to/llama3-emu3-sft/HF \
    --tokenizer_path /capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3_tokenizer \
    --vision-tokenizer-type emu3
```

### Apertus with EMU3.5-IBQ

Uses the EMU3.5 IBQ vision tokenizer (16x compression with information bottleneck quantization). Requires `--inferencer-type hf` on older vLLM versions that don't support the Apertus architecture.

```bash
sbatch vision_tokenization/scripts/run_qualitative_benchmarks.sh \
    --experiment_name qa_apertus_emu3_5 \
    --model_path /path/to/apertus-8b-sft/HF \
    --tokenizer_path /capstor/store/cscs/swissai/infra01/MLLM/apertus_emu3.5_tokenizer \
    --vision-tokenizer-type emu3.5 \
    --inferencer-type hf
```

### Choosing the Inference Backend

| Backend | Flag | When to use |
|---------|------|------------|
| vLLM | `--inferencer-type vllm` (default) | Standard models, faster inference, supports tensor parallelism (`--tp-size`) |
| HuggingFace | `--inferencer-type hf` | Models not yet in vLLM (e.g., Apertus on older vLLM), debugging |

### Choosing the Vision Tokenizer

| Type | Flag | Codebook | Compression | Notes |
|------|------|----------|-------------|-------|
| EMU3 | `--vision-tokenizer-type emu3` (default) | 32K | 8x8 | BAAI/Emu3-VisionTokenizer, no extra path needed |
| EMU3.5-IBQ | `--vision-tokenizer-type emu3.5` | varies | 16x | Requires `--vision-tokenizer-path` or uses default CSCS path |

---

## Benchmark Modes

### VLM Q&A (default)

Runs VLM inference on image-prompt pairs matched by tags. Images and prompts are loaded from JSON config files and paired based on overlapping tags.

**When to use:** Evaluating how well a model answers questions or describes images.

```bash
# Via launch script
sbatch vision_tokenization/scripts/run_qualitative_benchmarks.sh \
    --experiment_name qa_apertus_emu3_5 \
    --model_path /path/to/apertus-8b-sft/HF \
    --tokenizer_path /capstor/store/cscs/swissai/infra01/MLLM/apertus_emu3.5_tokenizer \
    --vision-tokenizer-type emu3.5 \
    --inferencer-type hf \
    --chat-format to_apertus

# Direct invocation
python vlm_benchmark.py \
    --tokenizer_path /path/to/tokenizer \
    --model_path /path/to/model \
    --experiment_name my_qa_experiment \
    --image_list images.json \
    --prompt_list prompts.json \
    --chat-format to_apertus \
    --temperature 0.3 \
    --top_p 0.9
```

**Chat template handling:** By default the model's chat template is applied. Use `--chat-format` to select a format transform (`to_apertus`, `to_llama`) that restructures the prompt before template application. Use `--no_chat_template` to disable templating entirely and send raw `<image_tokens> + prompt` to the model. For models that don't implement HF's `apply_chat_template` (e.g., BAAI/Emu3-Chat), use `--prompt-builder` to apply a hardcoded prompt format instead — this takes highest priority and bypasses both `--chat-format` and `--no_chat_template`.

### Image Completion

Provides a percentage of an image's token rows as context and lets the model generate the remaining rows. Validates structural correctness (consistent row widths, proper end-of-row/frame/image tokens) and computes perceptual metrics (PSNR, SSIM, LPIPS) against a reference reconstruction.

**When to use:** Evaluating a model's ability to continue/complete visual content from partial context.

```bash
# EMU3 model, complete from 30%, 60%, 80%, 90% of rows
sbatch vision_tokenization/scripts/run_qualitative_benchmarks.sh \
    --experiment_name completion_llama3_emu3 \
    --model_path /path/to/llama3-emu3-sft/HF \
    --tokenizer_path /capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3_tokenizer \
    --vision-tokenizer-type emu3 \
    --image-completion \
    --completion-percentages 30,60,80,90

# Apertus EMU3.5, greedy decoding, strict row count validation
sbatch vision_tokenization/scripts/run_qualitative_benchmarks.sh \
    --experiment_name completion_apertus_emu3_5_greedy \
    --model_path /path/to/apertus-8b-sft/HF \
    --tokenizer_path /capstor/store/cscs/swissai/infra01/MLLM/apertus_emu3.5_tokenizer \
    --vision-tokenizer-type emu3.5 \
    --inferencer-type hf \
    --image-completion \
    --completion-percentages 30,60,80,90 \
    --strict-row-count \
    --greedy
```

**Key flags:**
- `--completion-percentages` (required): Comma-separated percentages of rows to provide as context, e.g. `30,60,80,90`
- `--strict-row-count`: Require the model to generate exactly the expected number of remaining rows for the completion to count as valid
- `--greedy`: Use greedy decoding (temperature=0, top_p=1.0) for deterministic output

**Output:** Results JSON + reconstructed images saved in `results/<experiment_name>/completion_images/` with a red boundary line marking where generation began.

### Captioning

Generates captions for each image. Chat template is disabled by default in this mode. An optional init phrase seeds the beginning of the caption.

**When to use:** Evaluating image captioning quality, or generating captions for a set of benchmark images.

```bash
# Basic captioning
sbatch vision_tokenization/scripts/run_qualitative_benchmarks.sh \
    --experiment_name captioning_apertus_emu3_5 \
    --model_path /path/to/apertus-8b-sft/HF \
    --tokenizer_path /capstor/store/cscs/swissai/infra01/MLLM/apertus_emu3.5_tokenizer \
    --vision-tokenizer-type emu3.5 \
    --inferencer-type hf \
    --captioning

# With an init phrase to seed the caption
sbatch vision_tokenization/scripts/run_qualitative_benchmarks.sh \
    --experiment_name captioning_apertus_emu3_5_seeded \
    --model_path /path/to/apertus-8b-sft/HF \
    --tokenizer_path /capstor/store/cscs/swissai/infra01/MLLM/apertus_emu3.5_tokenizer \
    --vision-tokenizer-type emu3.5 \
    --inferencer-type hf \
    --captioning \
    --caption-init-phrase "The image shows"
```

**Note:** In captioning mode, `--prompt_list` is ignored and the chat template is disabled by default.

---

## Command-Line Reference

### Core Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--tokenizer_path` | Yes | - | Path to text tokenizer (must contain vision special tokens) |
| `--model_path` | Yes | - | Path to VLM model |
| `--experiment_name` | Yes | - | Name of experiment (used for output filename) |
| `--results_folder` | No | `results/` | Directory for results |
| `--image_list` | No | `images.json` | Path to image configuration JSON |
| `--prompt_list` | No | `prompts.json` | Path to prompt configuration JSON (Q&A mode only) |
| `--overwrite` | No | `False` | Overwrite existing results file |
| `--debug` | No | `False` | Print token IDs and decoded text for first 3 samples |

### Vision Tokenizer Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--vision-tokenizer-type` | `emu3` | `emu3` or `emu3.5` / `emu3.5-ibq` |
| `--vision-tokenizer-path` | None | Path to vision tokenizer model weights (required for emu3.5 unless default CSCS path works) |
| `--min-tokenizer-pixels` | 65536 (256x256) | Minimum pixel count for image preprocessing |
| `--max-tokenizer-pixels` | 262144 (512x512) | Maximum pixel count for image preprocessing |

### Inference Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--inferencer-type` | `vllm` | `vllm` (faster) or `hf` (more compatible) |
| `--max-seq-len` | 8192 | Maximum sequence length for the model |
| `--tp-size` | 1 | Tensor parallel size (vLLM only) |
| `--no_chat_template` | `False` | Disable chat template application |
| `--chat-format` | None | Chat format transform: `to_apertus`, `to_llama` |
| `--prompt-builder` | None | Custom prompt builder bypassing `apply_chat_template` entirely: `emu3` (required for BAAI/Emu3-Chat) |
| `--temperature` | 0.3 | Sampling temperature |
| `--top_p` | 0.9 | Top-p sampling parameter |
| `--max_new_tokens` | 300 | Maximum tokens to generate |
| `--greedy` | `False` | Greedy decoding (sets temperature=0, top_p=1.0) |
| `--no-kv-cache` | `False` | Disable KV cache during inference (HF inferencer only) |

### Mode-Specific Arguments

| Argument | Mode | Description |
|----------|------|-------------|
| `--image-completion` | - | Enable image completion mode |
| `--completion-percentages` | Image completion | Comma-separated row percentages, e.g. `30,60,80,90` |
| `--strict-row-count` | Image completion | Require exact row count match for validity |
| `--captioning` | - | Enable captioning mode |
| `--caption-init-phrase` | Captioning | Seed phrase for caption generation |

---

## Results Viewer Webapp

### Overview

A Flask web application for browsing, filtering, and comparing VLM benchmark results.

### Features

- Browse multiple experiments with metadata
- View input images, prompts, and model outputs side-by-side
- Compare multiple experiments side-by-side
- Filter results by tags
- REST API for programmatic access

### Usage

```bash
cd webapp
pip install -r requirements.txt
python app.py
```

The webapp starts on `http://localhost:5000`.

| Argument | Default | Description |
|----------|---------|-------------|
| `--results_folder` | `results/` | Path to results directory |
| `--assets_folder` | `assets/` | Path to image assets |
| `--port` | `5000` | Port number |
| `--host` | `0.0.0.0` | Host address |
| `--debug` | `False` | Debug mode with auto-reload |

### REST API

| Endpoint | Description |
|----------|-------------|
| `GET /api/experiments` | List all experiments |
| `GET /api/experiment/<name>` | Get results for a specific experiment |
| `GET /assets/<filename>` | Serve image assets |

---

## Data Format

### Input Files

#### `images.json`

```json
[
    {
        "path": "assets/example1.jpg",
        "tags": ["general", "objects", "colors"]
    },
    {
        "path": "assets/chart.png",
        "tags": ["chart", "statistics"]
    }
]
```

#### `prompts.json` (Q&A mode only)

```json
[
    {
        "text": "Describe what you see in the image",
        "tags": ["general"],
        "require_all_tags": false
    },
    {
        "text": "Describe the data shown in this chart",
        "tags": ["chart", "statistics"],
        "require_all_tags": true
    }
]
```

**Tag matching:** If `require_all_tags` is false (default), at least one tag must overlap. If true, all prompt tags must be present in the image's tags.

### Output

Results are saved to `results/<experiment_name>.json`. Image completion mode also saves reconstructed images to `results/<experiment_name>/completion_images/`.

```json
{
    "timestamp": "2025-10-15T18:21:33.916674",
    "model_path": "/path/to/model",
    "tokenizer_path": "/path/to/tokenizer",
    "vision_tokenizer": "EMU3.5-IBQ",
    "inferencer": "HFInferencer",
    "inference_args": { "..." },
    "total_runs": 10,
    "runs": [
        {
            "image": { "path": "assets/example.jpg", "tags": ["general"] },
            "prompt": { "text": "Describe what you see", "tags": ["general"] },
            "output": "The image shows..."
        }
    ]
}
```

### Directory Structure

```
qualitative_benchmark/
├── assets/                    # Benchmark images
├── results/                   # Output (one JSON per experiment)
├── images.json                # Image configuration
├── prompts.json               # Prompt configuration (Q&A mode)
├── vlm_benchmark.py           # Main benchmark script
├── vlm.py                     # VLM orchestrator
├── metrics.py                 # PSNR, SSIM, LPIPS computation
├── emu3_reconstruct_helper.py # Token extraction and image reconstruction
├── inferencers/               # Inference backends
│   ├── base.py                # BaseInferencer ABC
│   ├── vllm_inferencer.py     # vLLM backend
│   └── hf_inferencer.py       # HuggingFace backend
├── v_tokenizers/              # Vision tokenizer wrappers
│   ├── base.py                # VLMVisionTokenizer ABC
│   ├── emu3.py                # EMU3 wrapper
│   └── emu3_5_ibq.py          # EMU3.5 IBQ wrapper
├── utils/                     # Prompt formatting utilities
├── README.md                  # This file
└── webapp/                    # Results viewer
    ├── app.py
    ├── requirements.txt
    ├── static/
    └── templates/
```

---

## Troubleshooting

### VLM Benchmark Issues

**CUDA out of memory:**
- Use `--max-tokenizer-pixels` to reduce image resolution
- Use `--tp-size 2` (or more) with vLLM to shard across GPUs
- Use `--max-seq-len` to limit context length

**Import errors / model loading failures:**
- Check `transformers` version: `pip install "transformers>=4.56,<5.0.0"`
- Apertus models need `>=4.56`; EMU3 VisionTokenizer breaks on `5.x`

**No results generated:**
- Check that image and prompt tags overlap (Q&A mode)
- Verify image paths in `images.json` are correct relative to the working directory
- Ensure model and tokenizer paths are valid

**Results file already exists:**
- Use `--overwrite` or choose a different experiment name

### Webapp Issues

**No experiments showing up:**
- Check `--results_folder` points to the directory containing JSON result files
- Ensure JSON files are valid

**Images not loading:**
- Check `--assets_folder` path
- Ensure image paths in results match actual file locations
