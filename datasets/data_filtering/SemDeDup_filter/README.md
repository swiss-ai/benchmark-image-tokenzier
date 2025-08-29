# SemDeDup for WebDataset

Semantic deduplication pipeline for WebDataset format using CLIP embeddings and K-means clustering to identify and remove near-duplicate images.


---

## 1. SemDeDup Pipeline (Optimized)

**File:** `semdedup_webdataset_optimized.py`

### Basic Usage
```bash
python semdedup_webdataset_optimized.py \
  --dataset-path /path/to/webdataset \
  --output-path ./results \
  --epsilon 0.01
```

### All Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset-path` | Required | Path to WebDataset directory containing .tar files |
| `--output-path` | `./semdedup_output` | Output directory for results |
| `--max-shards` | None | Limit number of shards to process |
| `--max-samples` | None | Limit total samples to process |
| `--clip-model` | `ViT-L/14` | CLIP model variant (ViT-B/32, ViT-L/14, etc.) |
| `--batch-size` | 256 | Batch size for GPU processing |
| `--num-workers` | 16 | Data loading workers |
| `--num-clusters` | 100 | K-means clusters (use ~sqrt(N) for N samples) |
| `--epsilon` | `0.001 0.005 0.01 0.02 0.05` | Deduplication thresholds (space-separated) |
| `--no-multi-gpu` | False | Disable multi-GPU processing |
| `--no-mixed-precision` | False | Disable FP16 mixed precision |
| `--verbose` | False | Enable detailed logging |
| `--seed` | 42 | Random seed |

### Example Commands
```bash
# Test on small dataset
python semdedup_webdataset_optimized.py \
  --dataset-path /data/webdataset \
  --max-samples 5000 \
  --num-clusters 70 \
  --epsilon 0.01

# Production run with multiple thresholds
python semdedup_webdataset_optimized.py \
  --dataset-path /data/large_dataset \
  --output-path ./production_results \
  --epsilon 0.001 0.005 0.01 0.02 0.05 \
  --num-clusters 500 \
  --batch-size 512

# Limited shards for testing
python semdedup_webdataset_optimized.py \
  --dataset-path /data/webdataset \
  --max-shards 5 \
  --max-samples 10000 \
  --verbose
```

---

## 2. Filter WebDataset

**File:** `filter_webdataset.py`

### Basic Usage
```bash
python filter_webdataset.py \
  --input-path /path/to/webdataset \
  --output-path ./filtered_dataset \
  --filter-file ./results/kept_samples_eps_0.01.txt
```

### All Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-path` | Required | Input WebDataset path |
| `--output-path` | Required | Output directory for filtered dataset |
| `--filter-file` | Required | Text file with kept sample IDs |
| `--filter-parquet` | None | Alternative: parquet file with filtering metadata |
| `--num-workers` | 4 | Number of parallel workers |
| `--verbose` | False | Enable verbose logging |

### Example Commands
```bash
# Filter using text file
python filter_webdataset.py \
  --input-path /data/original \
  --output-path ./filtered_0.01 \
  --filter-file ./results/kept_samples_eps_0.01.txt

# Filter using parquet metadata
python filter_webdataset.py \
  --input-path /data/original \
  --output-path ./filtered_0.05 \
  --filter-parquet ./results/filter_eps_0.05.parquet \
  --verbose
```

---

## 3. Visualize Duplicates

**File:** `visualize_duplicates.py`

### Basic Usage
```bash
python visualize_duplicates.py \
  --semdedup-output ./results \
  --dataset-path /path/to/webdataset \
  --epsilon 0.01
```

### All Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--semdedup-output` | `./semdedup_results` | SemDeDup output directory |
| `--dataset-path` | Required | Path to WebDataset |
| `--output-dir` | `./duplicate_visualizations` | Output directory |
| `--epsilon` | 0.01 | Epsilon value to analyze |
| `--max-pairs` | 100 | Maximum duplicate pairs to visualize |
| `--max-groups` | 10 | Maximum duplicate groups to show |
| `--max-scan` | 20000 | Maximum samples to scan when extracting images (reduce for speed) |

### Example Commands
```bash
# Visualize top duplicates
python visualize_duplicates.py \
  --semdedup-output ./results \
  --dataset-path /data/webdataset \
  --epsilon 0.005 \
  --max-pairs 50

# Quick analysis with few examples
python visualize_duplicates.py \
  --semdedup-output ./test_results \
  --dataset-path /data/webdataset \
  --output-dir ./test_viz \
  --max-pairs 10 \
  --max-groups 5

# Fast visualization with limited scanning
python visualize_duplicates.py \
  --semdedup-output ./results \
  --dataset-path /data/webdataset \
  --max-scan 5000 \
  --max-pairs 20
```

### Output Files
```
output_dir/
├── duplicate_pair_*.png         # Side-by-side duplicate comparisons
├── duplicate_group_*.png        # Groups of similar images
├── duplicate_pairs.csv          # Duplicate pairs data
├── duplicate_summary.json       # Analysis statistics
└── top_duplicate_groups.json   # Duplicate group information
```

**Note:** Visualization can be slow for large datasets as it needs to scan through tar files to find specific images. Use `--max-scan` to limit scanning time at the cost of potentially missing some images.

---

## Complete Workflow Example

```bash
# 1. Run SemDeDup pipeline
python semdedup_webdataset_optimized.py \
  --dataset-path /capstor/store/cscs/swissai/infra01/vision-datasets/coco2017/webdataset_permissive \
  --output-path ./coco_semdedup_full_1 \
  --epsilon 0.01 0.03 0.05 \
  --num-clusters 100 \
  --verbose

# 2. Visualize duplicates (generates PNG images)
python visualize_duplicates.py \
  --semdedup-output ./coco_semdedup \
  --dataset-path /capstor/store/cscs/swissai/infra01/vision-datasets/coco2017/webdataset_permissive \
  --epsilon 0.05 \
  --output-dir ./coco_semdedup/viz1 \
  --max-pairs 50 \
  --max-groups 50 \
  --max-scan 10000

# 3. Create filtered dataset
python filter_webdataset.py \
  --input-path /capstor/store/cscs/swissai/infra01/vision-datasets/coco2017/webdataset_permissive \
  --output-path ./coco_filtered \
```
---



