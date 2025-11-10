# HuggingFace Dataset Downloader

Pre-download HuggingFace datasets to cache for efficient use with the vision tokenization pipeline.

## Quick Start

```bash
# Download FineVision subset
sbatch download_dataset_hf.slurm laion_gpt4v

# Monitor progress
tail -f /iopsstor/scratch/cscs/$USER/benchmark-image-tokenizer/download_dataset/logs/hf-*.out
```

## Parameters

Configure via environment variables:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATASET_NAME` | `HuggingFaceM4/FineVision` | HF dataset repository |
| `SUBSET_NAME` | `$1` (required) | Dataset subset/config (use `""` if none) |
| `SPLIT` | `train` | Dataset split |
| `CACHE_DIR` | `/capstor/.../hf_cache` | Cache location |
| `NUM_PROC` | auto-detect CPUs | Number of download processes |
| `FORCE_REDOWNLOAD` | `false` | Re-download if cached |

## Usage Examples

### FineVision Subsets

```bash
sbatch download_dataset_hf.slurm laion_gpt4v
sbatch download_dataset_hf.slurm lvis_instruct4v
sbatch download_dataset_hf.slurm ocrvqa
sbatch download_dataset_hf.slurm allava_instruct_laion4v
```

### Other Datasets

```bash
# DOCCI (no subset needed)
DATASET_NAME="google/docci" sbatch download_dataset_hf.slurm ""

# Custom cache location
CACHE_DIR="/custom/cache" sbatch download_dataset_hf.slurm laion_gpt4v

# Force re-download
FORCE_REDOWNLOAD="true" sbatch download_dataset_hf.slurm ocrvqa

# Custom number of processes
NUM_PROC=64 sbatch download_dataset_hf.slurm lvis_instruct4v
```

## Integration with Tokenization

Add `--cache-dir <path>` to tokenization commands to use cached datasets.

### tokenize.py

```bash
python vision_tokenization/tokenize.py hf \
    --dataset-name "HuggingFaceM4/FineVision" \
    --config-name "laion_gpt4v" \
    --cache-dir "/capstor/store/cscs/swissai/infra01/vision-datasets/hf_cache" \
    --tokenizer-path "/path/to/tokenizer" \
    --output-dir "/path/to/output" \
    --num-gpus 4 \
    --device cuda \
    --mode image_only
```

### tokenize_hf_datasets.py

```bash
python vision_tokenization/tokenize_hf_datasets.py \
    --dataset-name "HuggingFaceM4/FineVision" \
    --config-name "laion_gpt4v" \
    --cache-dir "/capstor/store/cscs/swissai/infra01/vision-datasets/hf_cache" \
    --tokenizer-path "/path/to/tokenizer" \
    --output-dir "/path/to/output" \
    --num-gpus 4 \
    --batch-size 1000
```

### tokenize_hf_datasets_sft.py

```bash
python vision_tokenization/tokenize_hf_datasets_sft.py \
    --dataset-name "HuggingFaceM4/FineVision" \
    --config-name "lvis_instruct4v" \
    --cache-dir "/capstor/store/cscs/swissai/infra01/vision-datasets/hf_cache" \
    --tokenizer-path "/path/to/tokenizer" \
    --output-dir "/path/to/output" \
    --num-gpus 4 \
    --min-image-res "256*256" \
    --max-image-res "696*696"
```

## Troubleshooting

### Authentication Error

```bash
huggingface-cli login
# or
export HF_TOKEN="your_token_here"
```

### Dataset Not Found During Tokenization

Ensure `--cache-dir` matches the download cache location exactly.

### Slow Downloads

- Script auto-uses `hf_transfer` for faster downloads
- Increase time limit: `#SBATCH --time=24:00:00`

### Check if Dataset is Cached

```bash
python -c "
from datasets import load_dataset
ds = load_dataset(
    'HuggingFaceM4/FineVision',
    name='laion_gpt4v',
    split='train',
    cache_dir='/capstor/store/cscs/swissai/infra01/vision-datasets/hf_cache'
)
print(f'Loaded: {len(ds)} samples')
"
```

## Cache Structure

```
/capstor/store/cscs/swissai/infra01/vision-datasets/hf_cache/
├── downloads/                           # Raw files
├── HuggingFaceM4___fine_vision/        # Processed cache
│   ├── laion_gpt4v/train/
│   ├── lvis_instruct4v/train/
│   └── ...
└── google___docci/default/train/
```

Cache is shared across tokenization runs - no manual management needed.

## Batch Download

```bash
#!/bin/bash
SUBSETS=("laion_gpt4v" "lvis_instruct4v" "ocrvqa")

for subset in "${SUBSETS[@]}"; do
    sbatch download_dataset_hf.slurm "$subset"
    sleep 5
done
```

## Related Documentation

- **vision_tokenization/**: Tokenization pipeline docs
- **HuggingFace Datasets**: https://huggingface.co/docs/datasets/