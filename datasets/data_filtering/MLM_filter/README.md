# MLM Filter Pipeline for WebDataset

Multi-modal Language Model (MLM) filter for assessing image-caption quality in webdataset format. Based on the [MLM-Filter](https://github.com/weizhiwang/MLM-Filter) implementation using Qwen2.5-1.5B.

## Installation

```bash
# Create conda environment
conda create -n mlm_filter python=3.10
conda activate mlm_filter

# Install dependencies
pip install torch torchvision transformers==4.44.2
pip install webdataset pandas pyarrow matplotlib seaborn
pip install pillow peft accelerate

# Clone and setup LLaVA (required for model architecture)
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .
cd ..
```

## Quick Start

### Basic Usage

```bash
# Test mode - processes first TAR file only (quick validation)
python mlm_filter_pipeline_multigpu.py --test_mode --num_gpus 1

# Production run with default settings (4 GPUs)
python mlm_filter_pipeline_multigpu.py

# Specify number of GPUs
python mlm_filter_pipeline_multigpu.py --num_gpus 4

# Custom configuration file
python mlm_filter_pipeline_multigpu.py --config config_test.yaml --num_gpus 4
```

### Analyze Results

```bash
# Generate statistical analysis report
python analyze_mlm_results.py --input mlm_scores_output_test/mlm_scores_final.parquet --output_dir ./mlm_analysis_test2

# Extract sample images with score annotations (per metric)
python save_sample_images.py --parquet mlm_scores_output_test/mlm_scores_final.parquet --output_dir ./mlm_analysis_test2

# Extract top/bottom 10% based on mean score
python save_sample_images_mean.py --parquet mlm_scores_output_test/mlm_scores_final.parquet --output_dir ./mlm_analysis_test2
```

#### Mean Score Visualization

The `save_sample_images_mean.py` script provides a focused view of quality extremes:

```bash
# Basic usage - saves top/bottom 10% based on mean of 4 scores
python save_sample_images_mean.py --parquet mlm_scores_output/mlm_scores_final.parquet

# Custom number of samples and percentile
python save_sample_images_mean.py \
  --parquet mlm_scores_output/mlm_scores_final.parquet \
  --num_samples 30 \
  --percentile 5  # Top/bottom 5%
```
