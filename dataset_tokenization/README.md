# Dataset Tokenization Pipeline

This directory contains a scalable pipeline for tokenizing large-scale image datasets using vision tokenizers and saving them in Megatron-LM's IndexedDataset format.

## Overview

The pipeline consists of three main components:

1. **WebDataset Sharding**: Converts directories of images into efficient WebDataset tar files
2. **Vision Tokenization**: Uses models like Emu3VisionTokenizer to convert images to discrete tokens
3. **IndexedDataset Creation**: Saves tokens in Megatron-LM compatible format for efficient training

## Directory Structure

```
dataset_tokenization/
├── configs/              # Dataset-specific configuration files
│   └── llava_config.yaml
├── scripts/              # Utility scripts (future)
├── utils/                # Utility modules
│   └── indexed_dataset.py
├── tokenize_images.py    # Main tokenization script
└── README.md
```

## Usage

### Basic Usage

#### Pure Vision Tokenizer (Emu3 only)
```bash
python tokenize_images.py \
    --dataset llava \
    --input-path /capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain \
    --device cuda \
    --batch-size 16
```

#### Multimodal Tokenizer (Text + Image tokens)
First, detect your text tokenizer's vocabulary size:
```bash
python utils/detect_vocab_size.py alehc/swissai-tokenizer
```

Then tokenize with proper vocabulary offset:
```bash
python tokenize_images.py \
    --dataset llava \
    --input-path /capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain \
    --text-vocab-size 50000 \
    --image-vocab-size 131072 \
    --device cuda \
    --batch-size 16
```

### Advanced Options

```bash
python tokenize_images.py \
    --dataset my_dataset \
    --input-path /path/to/images \
    --output-dir /custom/output/path \
    --tokenizer emu3 \
    --batch-size 32 \
    --num-shards 20 \
    --device cuda:1 \
    --create-shards  # Force recreation of shards
```

## Output Format

The pipeline creates the following structure:

```
output_dir/
└── dataset_name/
    ├── shards/           # WebDataset tar files
    │   ├── shard-000000.tar
    │   ├── shard-000001.tar
    │   └── ...
    ├── tokens/           # Tokenized data
    │   ├── dataset_name_emu3.bin      # Binary token data
    │   ├── dataset_name_emu3.idx      # Index for random access
    │   └── dataset_name_emu3.meta.json # Metadata
    └── logs/             # Processing logs
        └── tokenization_stats.json
```

## IndexedDataset Format

The output follows Megatron-LM's IndexedDataset format:

- `.bin`: Raw token data (int32 values)
- `.idx`: Index file with offsets and sizes
- `.meta.json`: Metadata including statistics

### Reading the Dataset

```python
from utils.indexed_dataset import IndexedDatasetReader

# Load the dataset
reader = IndexedDatasetReader("/path/to/dataset_name_emu3")

# Get number of sequences
print(f"Number of sequences: {len(reader)}")

# Get tokens for a specific image
tokens = reader[0]  # Returns numpy array of token indices
```

## Multimodal Tokenizer Support

### Vocabulary Structure

For multimodal models, the vocabulary is structured as:
```
[0, text_vocab_size)        - Text tokens
[text_vocab_size, total)    - Image tokens
```

For example, with `alehc/swissai-tokenizer` (text) + Emu3 (image):
- Text tokens: [0, 50000)
- Image tokens: [50000, 181072)
- Total vocabulary: 181,072 tokens

### Token Offset Process

1. **Emu3 tokenizer** outputs image tokens in range [0, 131072)
2. **Pipeline** adds offset: `image_token + text_vocab_size`
3. **Result**: Image tokens in range [50000, 181072)

This ensures no collision between text and image tokens in the multimodal vocabulary.

### Detecting Vocabulary Size

Use the provided utility to detect your text tokenizer's vocabulary:
```bash
python utils/detect_vocab_size.py alehc/swissai-tokenizer
```

This will output the exact vocabulary size and provide the correct command line arguments.

## Pipeline Details

### 1. WebDataset Creation

The pipeline first converts your image directory into WebDataset shards:

- Scans for images (jpg, png, webp)
- Splits into balanced shards
- Each sample contains:
  - Image data
  - Metadata (original path, index, etc.)

### 2. Tokenization Process

For each image:

1. **Load**: Image loaded from WebDataset as PIL
2. **Preprocess**: Convert to tensor, normalize (via `tokenizer.preprocess()`)
3. **Encode**: Generate discrete tokens (via `tokenizer.encode()`)
4. **Flatten**: Convert spatial tokens [H, W] to flat array
5. **Save**: Add to IndexedDataset

### 3. Statistics Tracking

The pipeline tracks:
- Total images processed
- Token statistics (min, max, average)
- Processing times
- Failed images

## Performance Considerations

### Single Node Optimization

- **Batch Processing**: Process multiple images at once
- **GPU Memory**: Clears cache periodically
- **I/O Efficiency**: WebDataset enables streaming from disk

### Future Multi-Node Support

The pipeline is designed for future distributed processing:

```bash
# Future usage (not implemented yet)
torchrun --nproc_per_node=8 tokenize_images.py \
    --dataset llava \
    --input-path /path/to/images \
    --distributed
```

## Adding New Tokenizers

To add a new vision tokenizer:

1. Implement the tokenizer following the base interface
2. Add to tokenizer choices in `tokenize_images.py`
3. Update the initialization logic

Example:
```python
if args.tokenizer == "emu3":
    self.tokenizer = Emu3VisionTokenizer(device=self.device)
elif args.tokenizer == "new_tokenizer":
    self.tokenizer = NewVisionTokenizer(device=self.device)
```

## Troubleshooting

### Out of Memory

- Reduce `--batch-size`
- Use CPU with `--device cpu`
- Enable gradient checkpointing in tokenizer (if supported)

### Slow Processing

- Increase `--batch-size` if memory allows
- Use faster storage for shards
- Ensure GPU is being utilized

### Failed Images

Check `logs/tokenization_stats.json` for failed image count. Common causes:
- Corrupted image files
- Unsupported formats
- Permission issues

## References

- WebDataset: https://github.com/webdataset/webdataset
- Megatron-LM IndexedDataset: https://github.com/NVIDIA/Megatron-LM
- Emu3 Vision Tokenizer: https://huggingface.co/BAAI/Emu3-VisionTokenizer