# Image Information Filter for WebDataset

Fast, parallel computation of image quality metrics for large-scale WebDataset archives with resolution-independent evaluation, percentile-based filtering and visualization.

## Quick Start

```bash
# Setup
conda activate info_metric

# Process dataset with optimal configuration (128-core server)
python run_webdataset_metrics.py --config new_config.yaml

# Analyze and extract samples
python run_analysis.py \
    --metrics-path ./densefusion_metrics \
    --dataset-path /capstor/store/cscs/swissai/infra01/vision-datasets/DenseFusion/DenseFusion-1M \
    --output ./analysis_results \
    --samples-per-percentile 30
```

## Metrics

| Metric | Description | Range | Quality Indicators |
|--------|-------------|-------|-------------------|
| **Luminance Entropy** | Tonal diversity (Shannon entropy) | 0-1 | Low (<0.3): Uniform, High (>0.7): Rich textures |
| **Spatial Information** | ITU-T P.910 spatial complexity | 0-250+ | Low (<50): Smooth, High (>100): Detailed |
| **Edge Density** | Fraction of edge pixels (Canny) | 0-1 | Low (<0.02): Few edges, High (>0.1): Edge-rich |
| **Variance of Laplacian** | Focus/sharpness measure | 0-30000+ | Low (<500): Blurry, High (>3000): Sharp |
| **Brenner Focus** | Gradient-based sharpness (2-pixel step) | 0-1000+ | Low (<100): Blurry, High (>400): Sharp |

## Processing Commands

### 1. Compute Metrics

```bash
# Full dataset (optimal config)
python run_webdataset_metrics.py --config new_config.yaml

# Specific shards only
python run_webdataset_metrics.py --config new_config.yaml --shards 0 1 2

# Custom settings (override config)
python run_webdataset_metrics.py \
    --input /capstor/store/cscs/swissai/infra01/vision-datasets/DenseF
           - usion/DenseFusion-1M \
    --output ./3shard_metrics_output_resolution_normalized \
    --shards 0 1 2 \
    --num-workers 40 \
    --metrics luminance_entropy spatial_information edge_density brenner_focus

# Test mode with limited samples
python run_webdataset_metrics.py --test --max-samples 100 --num-shards 2

# All options:
# --config FILE: Configuration YAML (default: webdataset_config.yaml)
# --input PATH: Override input dataset path
# --output PATH: Override output directory
# --shards N1 N2: Process specific shard indices
# --num-shards N: Process first N shards
# --test: Test mode with limited samples
# --max-samples N: Max samples per shard (for testing)
# --num-workers N: Override parallel workers
# --metrics M1 M2: Select specific metrics
# --no-preserve-sharding: Save as single parquet
# --summary-only: Only compute statistics
# --verbose/-v: Verbose output
# --quiet/-q: Suppress output
```

### 2. Analyze & Extract Samples

```bash
# Basic analysis with default settings
python run_analysis.py \
    --metrics-path ./densefusion_metrics \
    --dataset-path /path/to/dataset \
    --output ./analysis_output

# Custom percentiles (focus on extremes)
python run_analysis.py \
    --metrics-path ./densefusion_metrics \
    --dataset-path /path/to/dataset \
    --output ./extreme_analysis \
    --percentile-ranges 0 2 3 5 10 90 95 98 100 \
    --samples-per-percentile 30

# Select specific metrics for scoring
python run_analysis.py \
    --metrics-path ./densefusion_metrics \
    --dataset-path /path/to/dataset \
    --use-metrics luminance_entropy spatial_information edge_density brenner_focus \
    --samples-per-percentile 20

# Show all metrics in images (not just selected)
python run_analysis.py \
    --metrics-path ./densefusion_metrics \
    --dataset-path /path/to/dataset \
    --use-metrics brenner_focus \
    --show-all-metrics

# Statistics only (no image extraction)
python run_analysis.py \
    --metrics-path ./densefusion_metrics \
    --dataset-path /path/to/dataset \
    --no-images --no-plots
```

#### Analysis Options

| Option | Description | Default |
|--------|-------------|---------|
| `--metrics-path` | Path to metrics parquet files | Required |
| `--dataset-path` | Path to WebDataset tar files | Required |
| `--output` | Output directory | `./analysis_output` |
| `--n-percentiles` | Number of percentile bins | `10` |
| `--samples-per-percentile` | Images per quality range | `5` |
| `--percentile-ranges` | Custom percentile boundaries | Auto (0-100) |
| `--use-metrics` | Select metrics for scoring | All metrics |
| `--show-all-metrics` | Show all metrics in images | False |
| `--skip-missing-images` | Continue if images can't load | False |
| `--no-images` | Skip image extraction | False |
| `--no-plots` | Skip generating plots | False |


## Dataset Filtering

### Filter WebDataset Based on Quality Metrics

After computing metrics, filter the dataset while preserving WebDataset format:

```bash
# Percentile-based: keep middle 60% quality
python run_webdataset_filter.py \
    --input-path /path/to/dataset \
    --metrics-path /path/to/metrics \
    --output-path /path/to/filtered \
    --mode percentile \
    --percentile-range 20 80 \
    --metrics luminance_entropy spatial_information edge_density brenner_focus

# Percentile-based: keep top 70% quality
python run_webdataset_filter.py \
    --output-path /path/to/filtered \
    --mode percentile \
    --percentile-min 30 \
    --metrics luminance_entropy spatial_information edge_density brenner_focus

# Threshold-based filtering
python run_webdataset_filter.py \
    --output-path /path/to/filtered \
    --mode threshold \
    --min-avg-score 0.4 \
    --max-avg-score 0.8 \
    --metrics luminance_entropy spatial_information edge_density brenner_focus

# Per-metric thresholds
python run_webdataset_filter.py \
    --output-path /path/to/filtered \
    --mode threshold \
    --min-spatial 70.0 \
    --min-edge 0.03 \
    --min-laplacian 1000.0

# Using config file
python run_webdataset_filter.py \
    --config filter_configs/filter_config.yaml \
    --output-path /path/to/filtered \
    --num-workers 40
```

#### Available Metrics for Filtering

Default metrics for average score computation (recommended):
- `luminance_entropy` - Tonal diversity measure (default)
- `spatial_information` - Spatial complexity, ITU-T P.910 (default)
- `edge_density` - Edge pixel fraction (default)
- `brenner_focus` - Focus/sharpness measure, gradient-based (default)

Optional metric:
- `variance_of_laplacian` - Focus/sharpness measure, Laplacian-based (similar to brenner_focus)

Example combinations:
```bash
# Default 4 metrics (recommended for quality filtering)
--metrics luminance_entropy spatial_information edge_density brenner_focus

# All 5 metrics (includes variance_of_laplacian - not recommended as it's redundant)
--metrics luminance_entropy spatial_information edge_density variance_of_laplacian brenner_focus
```

#### Filtering Options

| Option | Description | Default |
|--------|-------------|---------|
| **Paths** | | |
| `--input-path` | Path to original WebDataset | DenseFusion path |
| `--metrics-path` | Path to computed metrics | DenseFusion scores |
| `--output-path` | Output path for filtered dataset | Required |
| `--config` | Config file (JSON/YAML) | None |
| **Mode** | | |
| `--mode` | Filter mode: `percentile` or `threshold` | `threshold` |
| **Percentile Mode** | | |
| `--percentile-range` | Keep samples in range (e.g., `10 90`) | None |
| `--percentile-min` | Keep above percentile (e.g., `30`) | None |
| `--percentile-score` | Score for percentile filter | `avg_score` |
| **Threshold Mode** | | |
| `--min-avg-score` | Minimum average score | None |
| `--max-avg-score` | Maximum average score | None |
| `--min-luminance` | Min luminance entropy | None |
| `--max-luminance` | Max luminance entropy | None |
| `--min-spatial` | Min spatial information | None |
| `--max-spatial` | Max spatial information | None |
| `--min-edge` | Min edge density | None |
| `--max-edge` | Max edge density | None |
| `--min-laplacian` | Min variance of Laplacian | None |
| `--max-laplacian` | Max variance of Laplacian | None |
| **Processing** | | |
| `--metrics` | Metrics for average score | Default: luminance_entropy, spatial_information, edge_density, brenner_focus |
| `--shards` | Specific shard indices | All shards |
| `--num-workers` | Parallel workers | `40` |
| `--compression` | Output: `gz`, `bz2`, `xz` | None |

## Complete Workflow Example

```bash
# 1. Activate environment
conda activate info_metric

# 2. Process dataset to compute metrics
python run_webdataset_metrics.py --config new_config.yaml

# 3. Analyze metrics and extract samples

python run_analysis.py \
          --metrics-path /capstor/store/cscs/swissai/infra01/vision-datasets/DenseFusion/4_img_info_filtered_DenseFusion-1M \
          --dataset-path /capstor/store/cscs/swissai/infra01/vision-datasets/DenseFusion/DenseFusion-1M \
          --output ./3shard_viz \
          --samples-per-percentile 30 \
          --percentile-ranges 0 2 3 4 5 10 90 100 \
          --use-metrics luminance_entropy spatial_information edge_density brenner_focus

# 4. Filter dataset based on analysis (keep top 70%)
python run_webdataset_filter.py \
    --input-path /capstor/store/cscs/swissai/infra01/vision-datasets/DenseFusion/DenseFusion-1M \
    --metrics-path /capstor/store/cscs/swissai/infra01/vision-datasets/DenseFusion/4_img_info_filtered_DenseFusion-1M/scores \
    --output-path /capstor/store/cscs/swissai/infra01/vision-datasets/DenseFusion/4_img_info_filtered_DenseFusion-1M \
    --mode percentile \
    --percentile-min 5 \
    --metrics luminance_entropy spatial_information edge_density brenner_focus

# 5. Visualize filtered results
python visualize_samples.py \
    --analysis-dir ./analysis_results \
    --output-dir ./visualizations
```
