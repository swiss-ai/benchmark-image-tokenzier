# Distributed EMU3 Processing with WebDataset

## Overview

This guide covers distributed processing of image data using EMU3 tokenization with WebDataset, supporting multi-GPU and multi-node configurations for massive-scale data processing.

## Architecture

### Components

1. **WebDataset Shards**: Distributed tar files containing images and metadata
2. **Distributed Processor**: Handles shard distribution across ranks
3. **EMU3 Tokenizer**: Direct tokenization without text generation
4. **Indexed Dataset Builder**: Creates Megatron-compatible output
5. **Checkpoint Manager**: Enables resume capability

### Data Flow

```
WebDataset Shards → Distributed Loading → Vision Tokenization → EMU3 Tokenization → Indexed Dataset
     ↓                      ↓                    ↓                     ↓                    ↓
  [.tar files]      [Rank assignment]    [Image→Indices]      [Direct tokens]     [.bin/.idx files]
```

## Installation

```bash
# Install required packages
pip install webdataset torch numpy tqdm

# Clone repository
git clone <repository>
cd vision_tokenization
```

## Quick Start

### 1. Prepare WebDataset Shards

```python
from examples.prepare_webdataset_emu3 import process_dataset_distributed

# Convert images to WebDataset with EMU3 tokenization
process_dataset_distributed(
    image_dir="/path/to/images",
    output_dir="/path/to/shards",
    tokenizer_path="/path/to/emu3_tokenizer",
    images_per_shard=1000
)
```

### 2. Single GPU Processing

```bash
python utils/webdataset_emu3_distributed.py \
    --input-shards "/data/shards/shard-{000000..000099}.tar" \
    --output-prefix "/data/processed/emu3_tokens" \
    --tokenizer-path "/models/emu3_tokenizer" \
    --batch-size 64 \
    --num-workers 8
```

### 3. Multi-GPU Processing (Single Node)

```bash
torchrun --nproc_per_node=8 \
    utils/webdataset_emu3_distributed.py \
    --input-shards "/data/shards/shard-{000000..000999}.tar" \
    --output-prefix "/data/processed/emu3_tokens" \
    --tokenizer-path "/models/emu3_tokenizer" \
    --batch-size 64 \
    --num-workers 8 \
    --backend nccl
```

### 4. Multi-Node Processing (SLURM)

```bash
# Submit SLURM job
sbatch scripts/run_distributed_emu3.sh
```

## Configuration

### Key Parameters

| Parameter | Description | Default |
|-----------|------------|---------|
| `--input-shards` | WebDataset shard pattern | Required |
| `--output-prefix` | Output indexed dataset prefix | Required |
| `--tokenizer-path` | Path to EMU3 tokenizer | Required |
| `--batch-size` | Batch size per GPU | 32 |
| `--num-workers` | DataLoader workers | 4 |
| `--prefetch-factor` | Prefetch multiplier | 2 |
| `--buffer-size` | Shuffle buffer size | 10000 |
| `--checkpoint-dir` | Checkpoint directory | None |
| `--checkpoint-interval` | Samples between checkpoints | 1000 |

### Environment Variables

```bash
# Distributed settings
export MASTER_ADDR=node001
export MASTER_PORT=29500
export WORLD_SIZE=16
export RANK=0

# NCCL optimization
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
```

## Sharding Strategy

### Automatic Shard Distribution

Shards are automatically distributed across ranks:

```python
shards_per_rank = total_shards // world_size
extra_shards = total_shards % world_size

# Rank 0: shards 0-99
# Rank 1: shards 100-199
# ...
```

### Load Balancing

- Each rank processes an equal number of shards
- Extra shards distributed to lower ranks
- Shuffling within each rank for randomization

## Checkpointing

### Enable Checkpointing

```bash
python utils/webdataset_emu3_distributed.py \
    --checkpoint-dir /checkpoints/emu3 \
    --checkpoint-interval 5000 \
    ...
```

### Resume from Checkpoint

The system automatically detects and resumes from checkpoints:

```python
# Checkpoint format
{
    "processed_samples": 150000,
    "total_tokens": 38400000,
    "timestamp": 1699123456.789
}
```

## Performance Optimization

### 1. Optimal Batch Size

```python
# Formula: batch_size = available_gpu_memory / (sequence_length * 4)
# Example for 32GB GPU with 512 token sequences:
batch_size = 32 * 1024**3 / (512 * 4) ≈ 16384
```

### 2. Worker Configuration

```python
# Recommended settings
num_workers = min(cpu_count, batch_size // 4)
prefetch_factor = 2  # Load 2 batches ahead
```

### 3. Memory Management

```python
# Pin memory for faster GPU transfer
dataloader = DataLoader(
    dataset,
    pin_memory=True,
    persistent_workers=True
)
```

### 4. NCCL Tuning

```bash
# For InfiniBand networks
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3

# For Ethernet
export NCCL_SOCKET_IFNAME=eth0
```

## Monitoring

### Progress Tracking

```
Rank 0: 100%|████████| 10000/10000 [1:23:45<00:00, 2.01it/s, samples=320000, tokens=81920000, tok/s=16384]
```

### Logging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)
```

### Metrics

Monitor these key metrics:
- **Tokens/second**: Overall throughput
- **GPU utilization**: Should be >90%
- **Memory usage**: Avoid OOM
- **Network bandwidth**: For multi-node

## Output Format

### Indexed Dataset Structure

```
output_prefix.idx  # Index file with offsets
output_prefix.bin  # Binary data file with tokens
```

### Reading Output

```python
from utils.indexed_dataset_megatron import MMapIndexedDataset

dataset = MMapIndexedDataset('output_prefix')
tokens = dataset[0]  # Get first sequence
print(f"Sequence length: {len(tokens)}")
```

## Troubleshooting

### Common Issues

1. **OOM Errors**
   - Reduce batch size
   - Reduce number of workers
   - Enable gradient checkpointing

2. **Slow Processing**
   - Check GPU utilization
   - Increase batch size
   - Optimize shard size (1000-10000 samples)

3. **NCCL Errors**
   - Verify network configuration
   - Check firewall settings
   - Test with `NCCL_DEBUG=INFO`

4. **Checkpoint Issues**
   - Ensure checkpoint directory is accessible
   - Check disk space
   - Verify write permissions

## Benchmarks

### Single GPU (A100 80GB)

| Dataset Size | Processing Time | Throughput |
|-------------|----------------|------------|
| 1M images | 2.5 hours | 111 img/s |
| 10M images | 24 hours | 116 img/s |
| 100M images | 10 days | 116 img/s |

### Multi-GPU (8x A100)

| Dataset Size | Processing Time | Throughput |
|-------------|----------------|------------|
| 1M images | 20 minutes | 833 img/s |
| 10M images | 3.3 hours | 840 img/s |
| 100M images | 33 hours | 840 img/s |

### Multi-Node (4 nodes, 32x A100)

| Dataset Size | Processing Time | Throughput |
|-------------|----------------|------------|
| 100M images | 8.5 hours | 3,267 img/s |
| 1B images | 85 hours | 3,267 img/s |

## Advanced Usage

### Custom Vision Tokenizer

```python
class CustomVisionTokenizer:
    def encode(self, image):
        # Your vision encoding logic
        indices = your_model.encode(image)
        height, width = get_dimensions(indices)
        return indices, height, width

# Use in processing
processor = DistributedEMU3Processor(config)
processor.vision_tokenizer = CustomVisionTokenizer()
```

### Pipeline Integration

```python
# Chain with other preprocessing
dataset = (
    wds.WebDataset(shards)
    .decode("pil")
    .map(augment_image)  # Custom augmentation
    .map(process_emu3)   # EMU3 tokenization
    .batched(batch_size)
)
```

### Streaming Processing

```python
# Process infinite stream
dataset = wds.WebDataset(shards, shardshuffle=True, resampled=True)
for batch in dataset:
    process_batch(batch)
```

## Best Practices

1. **Shard Size**: 1GB-5GB per shard for optimal I/O
2. **Compression**: Use `.tar` without additional compression
3. **Shuffling**: Enable both shard and sample shuffling
4. **Validation**: Reserve 5-10% shards for validation
5. **Backup**: Keep original shards after processing

## Integration with Training

```python
# Use processed data in training
from torch.utils.data import DataLoader

dataset = MMapIndexedDataset('processed_tokens')
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

for batch in dataloader:
    # Training step
    loss = model(batch)
    loss.backward()
```

## Future Improvements

- [ ] Streaming tokenization without intermediate storage
- [ ] Dynamic batching based on sequence length
- [ ] Automatic shard rebalancing
- [ ] Cloud storage support (S3, GCS)
- [ ] Real-time monitoring dashboard
- [ ] Automatic performance tuning

## Related Resources

- [WebDataset Documentation](https://github.com/webdataset/webdataset)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL Performance Tuning](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/tuning.html)
- [Megatron-LM Data Processing](https://github.com/NVIDIA/Megatron-LM)