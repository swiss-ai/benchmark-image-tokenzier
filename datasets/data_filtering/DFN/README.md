# DFN (Data Filtering Networks) for Dataset Quality Scoring

This module implements Data Filtering Networks (DFN) for computing image-text alignment scores, enabling high-quality dataset filtering for vision-language models.

## Overview

DFN uses a CLIP-based model (`apple/DFN-public`) to compute alignment scores between images and their captions. Higher scores indicate better alignment, which typically correlates with higher quality training data.

## Installation

### 1. Setup Environment
```bash
# Create and activate conda environment
conda create -n dfn_filter python=3.10 -y
conda activate dfn_filter

# Install PyTorch with CUDA 12.6
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install required packages
pip install transformers pandas pyarrow webdataset tqdm accelerate open_clip_torch ftfy regex
```



## Quick Start

### 1. Test with Single Shard
```bash
python test_single_shard.py
```

### 2. Visualize Score Examples
```bash
# Quick visualization with default settings
python visualize_scores.py
```

### 3. Process Full Dataset
```bash
./run_full_processing.sh
```

## Main Scripts

### Core Modules

#### `dfn_filter.py`
Core DFN filtering module with the main `DFNFilter` class.

**Key Methods:**
- `compute_filtering_score(image, text)`: Compute score for single image-text pair
- `batch_compute_scores(pairs)`: Batch processing for multiple pairs
- `filter_by_threshold(scores, threshold)`: Filter by score threshold

**Example:**
```python
from dfn_filter import DFNFilter

model = DFNFilter(model_name='hf-hub:apple/DFN-public', device='cuda:0')
score = model.compute_filtering_score(image, caption)
```

#### `process_webdataset.py`
Multi-GPU parallel processing for webdataset format.

**Arguments:**
- `--dataset_path`: Path to webdataset directory
- `--output_path`: Output directory for scores
- `--batch_size`: Batch size per GPU (default: 32)
- `--num_workers`: Data loading workers (default: 4)

**Example:**
```bash
python process_webdataset.py \
    --dataset_path /path/to/webdataset \
    --output_path ./output \
    --batch_size 32
```

### Visualization Scripts

#### `visualize_custom.py`
Create individual images with scores and captions, categorized by percentiles.

**Arguments:**
- `--num_samples`: Number of samples to process (default: 500)
- `--samples_per_category`: Examples per percentile category (default: 5)
- `--output_dir`: Output directory (default: ./custom_visualization)


#### `visualize_scores.py`
Basic visualization with grid layout for different quality levels.

