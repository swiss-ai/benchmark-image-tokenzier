# VLM Qualitative Benchmark

A comprehensive toolkit for running qualitative benchmarks on Vision-Language Models (VLMs) and exploring the results through an interactive web interface.

## Overview

This toolkit consists of two main components:

1. **`vlm_benchmark.py`**: Script to run VLM inference on image-prompt pairs and store results
2. **Webapp**: Flask-based web interface to browse, filter, and analyze benchmark results

## Table of Contents

- [Installation](#installation)
- [VLM Benchmark Script](#vlm-benchmark-script)
- [Results Viewer Webapp](#results-viewer-webapp)
- [Data Format](#data-format)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch with CUDA support (for VLM inference)
- Flask 3.0.0 or higher (for webapp)

### Setup

1. Navigate to the qualitative benchmark directory:
```bash
cd vision_tokenization/qualitative_benchmark
```

2. Install dependencies:
```bash
# For VLM benchmarking
pip install torch torchvision pillow tqdm vllm

# For the webapp
pip install -r webapp/requirements.txt
```

---

## VLM Benchmark Script

### Overview

`vlm_benchmark.py` runs VLM inference on combinations of images and prompts based on tag matching, storing results in JSON format.

### Usage

```bash
python vlm_benchmark.py \
    --tokenizer_path /path/to/tokenizer \
    --model_path /path/to/model \
    --experiment_name my_experiment \
    --image_list images.json \
    --prompt_list prompts.json \
    --results_folder results/
```

### Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--tokenizer_path` | Yes | - | Path to tokenizer supporting SFT and images |
| `--model_path` | Yes | - | Path to HuggingFace model |
| `--experiment_name` | Yes | - | Name of experiment (used for JSON filename) |
| `--results_folder` | No | `results/` | Path to results folder |
| `--image_list` | No | `images.json` | Path to image configuration JSON |
| `--prompt_list` | No | `prompts.json` | Path to prompt configuration JSON |
| `--overwrite` | No | `False` | Overwrite existing results file if it exists |

### Input Files

#### `images.json`

JSON file listing images with their paths and tags:

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

#### `prompts.json`

JSON file listing prompts with their text and tags:

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

**Tag Matching:**
- If `require_all_tags: false` (default): At least one tag must overlap between image and prompt
- If `require_all_tags: true`: All prompt tags must be present in the image tags

### Output

The script creates a JSON file `results/<experiment_name>.json` containing:
- Timestamp
- Model and tokenizer paths
- All inference results with images, prompts, and outputs

**Important:** The script will abort if a results file with the same experiment name already exists, unless you use the `--overwrite` flag.

Example output structure:
```
results/
├── experiment1.json
├── experiment2.json
└── baseline_eval.json
```

### Example Workflow

1. Prepare your images in the `assets/` folder
2. Create `images.json` and `prompts.json` configuration files
3. Run the benchmark:

```bash
python vlm_benchmark.py \
    --tokenizer_path /data/models/emu3-tokenizer \
    --model_path /data/models/emu3-3B \
    --experiment_name baseline_eval \
    --image_list images.json \
    --prompt_list prompts.json
```

4. Results will be saved to `results/baseline_eval.json`

**To overwrite existing results:**
```bash
python vlm_benchmark.py \
    --tokenizer_path /data/models/emu3-tokenizer \
    --model_path /data/models/emu3-3B \
    --experiment_name baseline_eval \
    --overwrite
```

---

## Results Viewer Webapp

### Overview

A Flask web application that provides an interactive interface to browse, filter, and analyze VLM benchmark results.

### Features

- **Browse Multiple Experiments**: View and compare results from different benchmark runs
- **Visual Result Display**: See input images, prompts, and model outputs side-by-side
- **Model Information**: View which model and tokenizer were used for each experiment
- **Tag-based Filtering**: Filter results by image tags or prompt tags
- **Responsive Design**: Clean, modern interface that works on all screen sizes
- **REST API**: JSON endpoints for programmatic access

### Usage

#### Quick Start

```bash
cd webapp
python app.py
```

The webapp will start on `http://localhost:5000` by default.

#### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--results_folder` | `results/` | Path to results folder containing experiment subdirectories |
| `--assets_folder` | `assets/` | Path to assets folder containing images (auto-detected if not specified) |
| `--port` | `5000` | Port to run the webapp on |
| `--host` | `0.0.0.0` | Host to run the webapp on |
| `--debug` | `False` | Run in debug mode (auto-reload, detailed errors) |

#### Examples

**Basic usage with defaults:**
```bash
cd webapp
python app.py
```

**Custom results folder:**
```bash
python app.py --results_folder /path/to/my/results
```

**Run on different port:**
```bash
python app.py --port 8080
```

**Debug mode for development:**
```bash
python app.py --debug
```

**Full configuration:**
```bash
python app.py \
    --results_folder /data/vlm_results \
    --assets_folder /data/benchmark_images \
    --port 8000 \
    --host 127.0.0.1 \
    --debug
```

### Using the Web Interface

1. **Home Page**: View all available experiments with metadata
2. **Experiment Page**: Click on an experiment to view its results
3. **Model Configuration**: See which model and tokenizer were used
4. **Filters**: Use dropdowns to filter by image or prompt tags
5. **Results**: Each result card displays:
   - Input image with tags
   - Prompt text with tags
   - Model output

### REST API Endpoints

The webapp provides JSON API endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /api/experiments` | List all experiments |
| `GET /api/experiment/<name>` | Get results for specific experiment |
| `GET /assets/<filename>` | Serve image assets |

**Example API Usage:**

```bash
# List all experiments
curl http://localhost:5000/api/experiments

# Get specific experiment results
curl http://localhost:5000/api/experiment/baseline_eval

# Get results with filtering (through browser)
http://localhost:5000/experiment/baseline_eval?image_tag=chart&prompt_tag=statistics
```

---

## Data Format

### Results JSON Structure

Generated by `vlm_benchmark.py` and consumed by the webapp:

```json
{
    "timestamp": "2025-10-15T18:21:33.916674",
    "model_path": "/path/to/model",
    "tokenizer_path": "/path/to/tokenizer",
    "total_runs": 10,
    "runs": [
        {
            "image": {
                "path": "assets/example.jpg",
                "tags": ["general", "objects"]
            },
            "prompt": {
                "text": "Describe what you see in the image",
                "tags": ["general"]
            },
            "output": "The image shows..."
        }
    ]
}
```

### Directory Structure

Expected layout for the qualitative benchmark:

```
qualitative_benchmark/
├── assets/                    # Images for benchmarking
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── results/                   # Benchmark results (one JSON file per experiment)
│   ├── experiment1.json
│   ├── experiment2.json
│   └── baseline_eval.json
├── images.json               # Image configuration
├── prompts.json              # Prompt configuration
├── vlm_benchmark.py          # Benchmark script
├── README.md                 # This file
└── webapp/                   # Results viewer
    ├── app.py
    ├── requirements.txt
    ├── static/
    │   └── style.css
    └── templates/
        ├── base.html
        ├── index.html
        └── experiment.html
```

---

## Troubleshooting

### VLM Benchmark Issues

**CUDA out of memory:**
- Reduce batch size in the inference configuration
- Use smaller images or reduce resolution
- Check GPU memory with `nvidia-smi`

**Import errors:**
- Ensure all paths in `vlm_benchmark.py` are correct
- Check that the EMU3 tokenizer and inferencer modules are available
- Verify CUDA and PyTorch installation

**No results generated:**
- Check that image tags and prompt tags have at least one overlap
- Verify image paths in `images.json` are correct
- Ensure the model and tokenizer paths are valid

**Results file already exists:**
- Use `--overwrite` flag to overwrite existing results
- Or choose a different experiment name

### Webapp Issues

**No experiments showing up:**
- Verify the `--results_folder` path points to the correct directory
- Check that the results folder contains JSON result files (not folders)
- Ensure JSON files are valid and properly formatted

**Images not loading:**
- Verify the `--assets_folder` path is correct
- Check that image paths in JSON results match actual file locations
- Ensure images are in a web-compatible format (JPG, PNG, etc.)

**Port already in use:**
```bash
# Use a different port
python app.py --port 8080
```

**Permission denied:**
```bash
# Run on port > 1024 or use sudo (not recommended)
python app.py --port 8080
```

### General Tips

- **Check file paths**: All paths should be relative to the `qualitative_benchmark` directory
- **Validate JSON**: Use a JSON validator to check configuration files
- **Check logs**: The webapp prints useful diagnostic information on startup
- **Browser cache**: Clear browser cache if changes don't appear

---

## Development

### Running in Debug Mode

For development, use debug mode for auto-reload and detailed error messages:

```bash
cd webapp
python app.py --debug
```

### Adding New Features

**VLM Benchmark:**
- Modify `VLMBenchmark` class to add new benchmark types
- Extend tag matching logic in `_tags_match()` method
- Add new inference arguments to `InferenceArgs` class

**Webapp:**
- Add new routes in `app.py`
- Create new templates in `templates/`
- Modify styling in `static/style.css`
- Extend filtering logic for additional filter types

---

## Support

This toolkit is part of the SwissAI benchmark image tokenizer project. For issues or questions, please refer to the main project documentation.

## Notes

- Currently only supports EMU3-based vision-language models
- Image tokenization uses EMU3 vision tokenizer with configurable resolution limits
- Results are stored in JSON format for easy parsing and analysis
- The webapp is designed for local/internal use; add authentication for production deployment