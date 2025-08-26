# WebDataset Dataloader for Multi-Modal Training

## Quick Start

### Installation
```bash
conda create -n multimodal_training python=3.10
conda activate multimodal_training
pip install torch torchvision webdataset pillow tqdm numpy
```

### Available Datasets

| Dataset | Path | Shards | Size | Type |
|---------|------|--------|------|------|
| COCO 2017 (Permissive) | `/capstor/store/cscs/swissai/infra01/vision-datasets/coco2017/webdataset_permissive_highQ` | 31 | 26K images | `coco` |
| COCO 2017 (Full) | `/capstor/store/cscs/swissai/infra01/vision-datasets/coco2017/webdataset_output` | 118 | 118K images | `coco` |
| LLaVA-558K | `/capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain/webdataset_output` | 559 | 559K images | `llava` |
| DenseFusion-1M | `/capstor/store/cscs/swissai/infra01/vision-datasets/DenseFusion/DenseFusion-1M` | 53 | 262K images | `generic` |

## Core Files

### 1. `generalized_webdataset_loader.py`
**Main dataloader for all datasets**
```python
from generalized_webdataset_loader import GeneralizedWebDatasetLoader

loader = GeneralizedWebDatasetLoader(
    dataset_path="/path/to/dataset",
    dataset_type="coco",  # or "llava" or "generic"
    batch_size=32,
    num_workers=4,
    shuffle=True
)
dataset = loader.create_dataset(decode_images=True)
dataloader = loader.create_dataloader(dataset)
```

### 2. `multi_gpu_training_test.py`
**Multi-GPU training example**
```bash
python multi_gpu_training_test.py \
    --dataset_path /path/to/dataset \
    --dataset_type coco \
    --batch_size 16 \
    --num_workers 4 \
    --num_epochs 1 \
    --world_size 2  # number of GPUs
```

### 3. `convert_coco_to_wds_v2.py`
**Convert COCO to WebDataset format**
```bash
python convert_coco_to_wds_v2.py \
    --data_root /path/to/coco2017 \
    --output_dir /path/to/output \
    --samples_per_shard 1000 \
    --compression_quality 95 \
    --num_workers 8
```

### 4. `extract_permissive_coco.py`
**Extract permissive licensed COCO images**
```bash
python extract_permissive_coco.py \
    --input_dir /path/to/coco_webdataset \
    --output_dir /path/to/permissive_output \
    --samples_per_shard 1000
```




## Common Parameters

### GeneralizedWebDatasetLoader
- `dataset_path`: Path to WebDataset directory
- `dataset_type`: "coco", "llava", or "generic"
- `batch_size`: Batch size per GPU (default: 32)
- `num_workers`: Data loading workers (default: 4)
- `shuffle`: Shuffle data (default: True)
- `prefetch_factor`: Batches to prefetch (default: 2)
- `persistent_workers`: Keep workers alive (default: True)

### create_dataset()
- `decode_images`: Decode images (default: True)
- `image_format`: "PIL" or "torch" (default: "PIL")
- `return_metadata`: Include resolution info (default: False)
- `transform`: Optional image transforms
- `filter_fn`: Optional filter function
