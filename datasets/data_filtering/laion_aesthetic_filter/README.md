# LAION Aesthetic Score Filter

Multi-GPU pipeline for filtering large-scale webdataset formatted datasets using LAION aesthetic scores.

## Quick Start

```bash
# Setup
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vlm-data1

# Full pipeline (558K samples, ~22 minutes total)
python compute_embeddings_parallel.py --config config.yaml              # ~12 min
python compute_aesthetic_scores_parallel.py --config config.yaml --partition-by-shard  # ~5 min
python apply_filters.py --config config.yaml                            # ~5 min
```

## Commands and Explanations

### 1. Environment Setup
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vlm-data1
```
Activates the conda environment with PyTorch, CLIP, and required dependencies.

### 2. Extract CLIP Embeddings (Multi-GPU)
```bash
python compute_embeddings_parallel.py --config config.yaml
```
**What it does:**
- Extracts CLIP ViT-L/14 embeddings from 558K images using 4 GPUs in parallel
- Processes webdataset tar files from configured input path
- Saves L2-normalized embeddings as compressed numpy files (.npz)
- Uses batch_size=32 per GPU (optimal for GH200 hardware)
- Takes ~12 minutes for full dataset

**Test options:**
```bash
# Test with subset (10 shards, 2 GPUs)
python compute_embeddings_parallel.py --config config.yaml --max-shards 10 --num-gpus 2

# Single GPU fallback
python compute_embeddings_parallel.py --config config.yaml --num-gpus 1
```

### 3. Compute Aesthetic Scores (Multi-GPU)
```bash
python compute_aesthetic_scores_parallel.py --config config.yaml --partition-by-shard
```
**What it does:**
- Computes LAION aesthetic scores (0-10 scale) using pre-computed embeddings
- Uses 4 GPUs with score_batch_size=1024 for fast processing
- Saves scores as partitioned parquet files for efficient querying
- Takes ~5 minutes for full dataset

**Options:**
```bash
# Single file output (not recommended for large datasets)
python compute_aesthetic_scores_parallel.py --config config.yaml

# Test with subset
python compute_aesthetic_scores_parallel.py --config config.yaml --max-files 100
```

### 4. Apply Filters and Create Dataset
```bash
python apply_filters.py --config config.yaml
```
**What it does:**
- Loads aesthetic scores and applies filter threshold (default ≥4.5)
- Creates new webdataset with only passing samples (~63% pass rate)
- Generates detailed filter report with before/after statistics
- Outputs filtered dataset to configured output directory
- Takes ~5 minutes

**Options:**
```bash
# Dry run to see statistics without creating dataset
python apply_filters.py --config config.yaml --dry-run

# Custom samples per tar file
python apply_filters.py --config config.yaml --samples-per-tar 500
```

### 5. Visualize and Analyze Results
```bash
# Extract sample images by score percentiles (default: 15 samples per category)
python visualize_filtered_samples.py --filter-type aesthetic --num-samples 20 --create-grid

# Extract specific score categories
python visualize_filtered_samples.py --filter-type aesthetic --categories top_10_percent,bottom_10_percent

# Extract more samples for detailed analysis
python visualize_filtered_samples.py --filter-type aesthetic --num-samples 50 --create-grid
```
**What it does:**
- Extracts sample images from different aesthetic score ranges
- Creates visual grids showing quality differences across percentiles
- Saves organized samples to `./filtered_samples/aesthetic/` directory
- Generates summary statistics and score distributions

**Pre-generated samples available:**
The `filtered_samples/aesthetic/` directory already contains extracted samples showing:
- **Bottom 10%** (scores 0.56-3.4): Low-quality, blurry, poor composition
- **Low 25%** (scores 3.4-4.1): Below average quality
- **Median range** (scores 4.1-5.7): Average quality images
- **High 75%** (scores 5.7-6.3): Above average quality
- **Top 10%** (scores 6.3-9.1): High-quality, sharp, well-composed

Each category includes:
- Sample images (*.jpg)
- Metadata with captions and scores (metadata.json)
- Visual grid comparing samples (*_grid.png)

**How to access filtered dataset:**
After running `apply_filters.py`, the filtered webdataset will be created based on the output path in config.yaml.
- Check `filtered_output/filter_report.json` for statistics (if local output)
- View `filtered_output/filtered_samples.txt` for list of passing sample IDs
- Use the filtered webdataset directly for training

## Configuration

Edit `config.yaml` for paths and thresholds:

```yaml
dataset:
  input_path: "/path/to/webdataset_output"
  output_path: "./filtered_output"

metadata:
  base_path: "/path/to/metadata"
  embeddings_path: "/path/to/metadata/embeddings"
  scores_path: "/path/to/metadata/scores"

processing:
  batch_size: 32           # Optimal for GH200
  num_workers: 0           # 0 recommended for webdataset
  num_gpus: 4              # Use all available GPUs
  score_batch_size: 1024   # For aesthetic scoring

filters:
  aesthetic_score:
    enabled: true
    min_score: 4.5         # LAION threshold
    max_score: 10.0
```
