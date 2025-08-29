# Q-Align Scoring Pipeline

Visual quality and aesthetic scoring for WebDataset using Q-Align model.

## Setup

Activate the environment:
```bash
source ~/.bashrc
source /users/rqu/miniconda3/bin/activate q_align
cd /iopsstor/scratch/cscs/rqu/benchmark-image-tokenzier/datasets/data_filtering/Q-Align
```

## Files

- `score_dataset.py` - Main scoring script
- `filter_by_scores.py` - Filter samples by score thresholds
- `check_scoring_progress.py` - Monitor scoring progress
- `config_*.yaml` - Configuration files

## Usage

### 1. Score Dataset

**Test run (10 shards):**
```bash
python score_dataset.py --config config_test.yaml
```

**Production run (full dataset):**
```bash
python score_dataset.py --config config_production.yaml
```

### 2. Monitor Progress

```bash
python check_scoring_progress.py
```

For specific output directory:
```bash
python check_scoring_progress.py --output_dir /path/to/output
```

### 3. Filter Samples

Filter samples by quality scores:
```bash
python filter_by_scores.py --config config_filter.yaml
```




## Output Structure

### Scoring Output
```
output_dir/
├── shard_idx=0/           # Partitioned parquet files
│   └── *.parquet
├── q_align_statistics.json  # Score statistics
└── checkpoint.json        # Resume checkpoint
```

### Filtering Output
```
filtered_samples/
├── filtered_top_10_quality/
│   ├── samples.csv        # Filtered sample IDs
│   └── statistics.json    # Statistics for subset
└── ...
```

## Score Interpretation

| Score Type | Low | Medium | High | Very High |
|------------|-----|--------|------|-----------|
| Quality | <0.4 | 0.4-0.6 | 0.6-0.8 | >0.8 |
| Aesthetic | <0.3 | 0.3-0.5 | 0.5-0.7 | >0.7 |


## Example Workflow

```bash
# 1. Activate environment
conda activate q_align

# 2. Test on small subset
python score_dataset.py --config config_test.yaml

# 3. Check results
python check_scoring_progress.py --test

# 4. Run full dataset
python score_dataset.py --config config_production.yaml

# 5. Filter high-quality samples
python filter_by_scores.py --config config_filter.yaml
```