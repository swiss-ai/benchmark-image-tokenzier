# DeQA-Score: Image Quality Assessment Pipeline

A high-performance, multi-GPU pipeline for assessing image quality using DeQA-Score model on webdataset format.

## 🚀 Quick Start

```bash
# Navigate to project directory
cd /iopsstor/scratch/cscs/rqu/benchmark-image-tokenzier/datasets/data_filtering/DeQA

# Activate environment
source /users/rqu/miniconda3_x86/etc/profile.d/conda.sh
conda activate deqa_score

# Run scoring on your dataset
python run_deqa_scoring.py --config configs/your_config.yaml
```

## Overview

DeQA-Score is a fine-tuned image quality assessment model based on mPLUG-Owl2 (7B parameters). This pipeline provides:

- ✅ Multi-GPU parallel processing
- ✅ WebDataset format support
- ✅ Batch processing with automatic checkpointing
- ✅ Quality score distribution analysis
- ✅ Sample extraction by quality percentile
- ✅ Comprehensive statistics generation

### Quality Score Range
- **1.0-2.0**: Very Poor quality
- **2.0-3.0**: Poor to Below Average
- **3.0-4.0**: Average to Good
- **4.0-4.5**: Very Good
- **4.5-5.0**: Excellent

## Installation

Required packages are pre-installed in the `deqa_score` conda environment:
```
pytorch==2.0.1
transformers==4.36.1
webdataset
pyarrow
pandas
numpy
pillow
tqdm
```

## Usage Guide

### 1. Score Your Dataset


#### Run Scoring

```bash
# Score entire dataset
python run_deqa_scoring.py --config configs/your_dataset_config.yaml

# Example: Score COCO2017 dataset
python run_deqa_scoring.py --config configs/coco2017_config.yaml

# Example: Score LLaVA-Pretrain dataset
python run_deqa_scoring.py --config configs/config.yaml
```

### 2. Analyze Score Distribution

Determine optimal filtering thresholds for your dataset:

```bash
# Analyze score distribution
python analyze_score_distribution.py \
  --score-dir /path/to/scores \
  --output-json analysis.json \
  --no-plot

# Example: Analyze COCO2017 scores
python analyze_score_distribution.py \
  --score-dir /capstor/store/cscs/swissai/infra01/vision-datasets/coco2017/webdataset_permissive/DeQA_scores \
  --output-json coco2017_permissive_analysis.json \
  --no-plot

python analyze_score_distribution.py \
  --score-dir /capstor/store/cscs/swissai/infra01/vision-datasets/coco2017/webdataset/DeQA_scores \
  --output-json coco2017_full_analysis.json \
  --no-plot
```

This will show:
- Score distribution across quality intervals
- Percentile-based thresholds
- Filtering recommendations
- Dataset statistics

#### Recommended Filtering Thresholds

| Threshold | Strategy | Data Kept | Use Case |
|-----------|----------|-----------|----------|
| ≥ 2.5 | Minimal | ~98% | Remove only worst quality |
| ≥ 3.0 | Conservative | ~94% | General filtering |
| ≥ 3.5 | Moderate | ~80% | Quality-focused training |
| ≥ 4.0 | Aggressive | ~35-58% | Maximum quality only |

### 3. Filter Dataset by Quality

Filter your webdataset to keep only high-quality samples while preserving the original format:

```bash
# Filter webdataset by quality threshold
python filter_webdataset_by_quality.py \
  --input-dir /path/to/original/webdataset \
  --output-dir /path/to/filtered/webdataset \
  --score-dir /path/to/deqa/scores \
  --threshold 3.5 \
  --pattern "*.tar"

# Example: Filter COCO2017 dataset (keep quality >= 3.5)
python filter_webdataset_by_quality.py \
  --input-dir /capstor/store/cscs/swissai/infra01/vision-datasets/coco2017/webdataset_permissive \
  --output-dir /capstor/store/cscs/swissai/infra01/vision-datasets/coco2017/webdataset_permissive_highQ \
  --score-dir /capstor/store/cscs/swissai/infra01/vision-datasets/coco2017/webdataset_permissive/DeQA_scores \
  --threshold 3.5 \
  --pattern "coco_train2017_permissive*.tar"
```

#### Filtering Parameters
- `--threshold`: Quality score threshold (keep samples >= threshold)
- `--pattern`: Pattern to match tar files (default: "*.tar")
- `--dry-run`: Preview filtering statistics without processing files

#### Filtering Results Example
With threshold 3.5 on COCO2017:
- **Input**: 30,371 samples (4.2 GB)
- **Output**: 25,450 samples (3.6 GB)
- **Kept**: 83.8% of samples
- **Quality improvement**: Mean score 3.96 → 4.15

The filtered dataset maintains the exact webdataset structure and can be used with the same data loaders.

### 4. Extract Sample Images

Extract images from each quality percentile for visual inspection:

```bash
# Extract sample images by quality percentile
python extract_samples_by_percentile.py \
  --score-dir /path/to/scores \
  --input-dir /path/to/webdataset \
  --output-dir extracted_samples \
  --samples-per-percentile 50 \
  --n-percentiles 10 \
  --create-grid

# Example: Extract COCO2017 samples
python extract_samples_by_percentile.py \
  --score-dir coco2017_scores \
  --input-dir /capstor/store/cscs/swissai/infra01/vision-datasets/coco2017/webdataset_output \
  --output-dir coco2017_samples \
  --samples-per-percentile 50 \
  --n-percentiles 10 \
  --create-grid

# Example: Extract LLaVA samples
python extract_samples_by_percentile.py \
  --score-dir output_3shards \
  --input-dir /capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain/webdataset_output \
  --output-dir llava_samples \
  --samples-per-percentile 60 \
  --n-percentiles 10
```

#### Extraction Parameters
- `--samples-per-percentile`: Number of samples per quality tier (default: 10)
- `--n-percentiles`: Number of quality tiers (10 = deciles, 4 = quartiles)
- `--create-grid`: Generate visualization grid
- `--seed`: Random seed for reproducible sampling
