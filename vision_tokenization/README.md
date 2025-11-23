# Vision Tokenization Pipeline

Unified pipeline for tokenizing large-scale vision datasets with support for multiple modes and vision tokenizers.

**Important:** This pipeline requires an omni-tokenizer (extended text tokenizer with other modality vocabularies). The omni-tokenizer wraps token IDs from modality-specific tokenizers into the unified vocabulary. For example, vision token ID 100 becomes `<|visual token 00100|>` in the omni-tokenizer's vocabulary, with corresponding entries in the model's input and output embeddings. However, during our actual tokenization pipeline, we use the modality-specific tokenizer (e.g., Emu3VisionTokenizer) to get raw token indices, then apply a simple offset to map them to the omni-tokenizer's ID space.

**Why this matters:** During inference, the model can predict tokens from any modality (text, vision, audio, etc.). The omni-tokenizer handles the unified token space, while modality-specific tokenizers (vision, audio) decode their respective tokens back to images, audio, etc. See [`utils/omni_tokenizer/README.md`](utils/omni_tokenizer/README.md) for creating omni-tokenizers.

## Token Structure Format

Images are tokenized into a structured sequence with special tokens marking boundaries and rows:

```
[BOS]
<|img_start|>
"32*32"                    # Image dimensions as text
<|img_token_start|>
<|visual token 00100|>     # Row 1, column 1
<|visual token 00523|>     # Row 1, column 2
...
<|img_end_of_row|>         # End of row 1
<|visual token 01244|>     # Row 2, column 1
...
<|img_end_of_row|>         # End of row 2
...                        # All H rows
<|img_end_of_frame|>       # End of entire image
<|img_end|>
[EOS]
```

This structure is created by the `encapsulate_image()` method in [`vokenizers/emu/image_only.py`](vokenizers/emu/image_only.py), which wraps raw vision indices with boundary markers and spatial information. The complete pipeline is orchestrated by `tokenize_image()` in the same file.

**Tokenizer Class Hierarchy:**
```
EMUImageOnlyTokenizer (base class)
├── tokenize_image()      # Core method for image tokenization
├── encapsulate_image()   # Wraps vision indices with structure tokens
│
├─→ EMUImageTextPairTokenizer
│   └── tokenize_image_text_pair()
│       ├── Calls self.tokenize_image() for image (GPU, parallel)
│       ├── Tokenizes text separately (CPU, parallel)
│       └── Concatenates based on mode (image2text or text2image)
│
└─→ EMUSftTokenizer
    └── tokenize_conversation()
        ├── Calls self.tokenize_image() for image (GPU, parallel)
        ├── Applies chat template and tokenizes text (CPU, parallel, includes <|image|> placeholder)
        └── Replaces <|image|> placeholder token with actual vision tokens
```

## Quick Start

### Image-Only Tokenization
```bash
python tokenize.py hf \
    --mode image_only \
    --dataset-name HuggingFaceM4/FineVision \
    --config-name CoSyn_400k_chart \
    --dataset-split train \
    --tokenizer-path /path/to/omni/tokenizer \
    --output-dir ./output \
    --num-gpus 4 \
    --num-shards 100 \
    --device cuda
```

### Resume from Checkpoint
```bash
# If processing was interrupted, simply add --resume to skip completed shards
python tokenize.py hf \
    --config config.json \
    --resume
```

### SFT Tokenization (Conversations with Images)
```bash
python tokenize.py hf \
    --mode sft \
    --dataset-name HuggingFaceM4/FineVision \
    --config-name CoSyn_400k_chart \
    --dataset-split train \
    --tokenizer-path /path/to/omni/instruct/tokenizer \
    --output-dir ./output \
    --num-gpus 4 \
    --num-shards 100 \
    --device cuda
```

## Supported Modes

**Note**: Currently supports **single image** per sample. Multi-image interleaving is not yet supported.

- **`image_only`** - Tokenize single images only (for pretraining)
- **`image2text`** - Single image followed by text caption
- **`text2image`** - Text prompt followed by single image
- **`sft`** - Supervised fine-tuning with conversations (single image + text)

## Configuration Options

### Common Arguments

- `--tokenizer-path` - Path to omni-tokenizer (base for pretraining, instruct for sft). The vision tokenizer type and path are automatically loaded from the omni-tokenizer's config.
- `--output-dir` - Output directory for tokenized data
- `--num-gpus` - Number of GPUs for distributed processing
- `--device` - Device to use (cuda or cpu)

### Image Resolution Control

**Tokenizer Resolution** (controls vision tokenizer behavior):
- `--min-tokenizer-pixels` - Minimum pixels for tokenizer (e.g., "512*512" or "262144")
- `--max-tokenizer-pixels` - Maximum pixels for tokenizer (e.g., "1024*1024" or "1048576")

**Dataset Filtering** (filters which images to process):
- `--min-image-pixels` - Filter out images smaller than this
- `--max-image-pixels` - Filter out images larger than this

### Dataset Options

- `--dataset-name` - HuggingFace dataset name
- `--config-name` - Dataset configuration/subset
- `--dataset-split` - Dataset split (train/validation/test)
- `--cache-dir` - Cache directory for datasets
- `--max-samples` - Maximum samples to process (for testing)

### Processing Options

- `--num-shards` - Number of shards for distributed processing and checkpointing (required)
- `--num-proc` - Number of processes for dataset loading
- `--image-field` - Name of image field in dataset (default: "images")
- `--text-field` - Name of text field in dataset (default: "texts")
- `--resume` - Resume from existing checkpoint, skipping completed shards

## Configuration Files

You can use JSON configuration files instead of CLI arguments:

```json
{
  "dataset_name": "HuggingFaceM4/FineVision",
  "config_name": "CoSyn_400k_chart",
  "dataset_split": "train",
  "mode": "sft",
  "tokenizer_path": "/path/to/tokenizer",
  "output_dir": "./output",
  "num_gpus": 4,
  "num_shards": 100,
  "device": "cuda",
  "min_tokenizer_pixels": "512*512",
  "max_tokenizer_pixels": "1024*1024"
}
```

### Important: Shards in HuggingFace vs WebDataset

**Note:** The concept of "shards" differs between HuggingFace and WebDataset formats:

- **WebDataset shards**: Physical `.tar` files on disk. Each shard is a separate file containing a subset of data.
- **HuggingFace shards**: Logical views of the dataset created using `.shard()` method. The entire dataset stays as one unit, but we create efficient strided views for parallel processing.

When using `--num-shards` with HuggingFace datasets:
- It doesn't create physical shard files for input data
- It creates logical shards for processing, which results in separate output files (`rank_XXX_shard_YYYYY.bin/idx`)
- Each logical shard is processed atomically for checkpointing
- `num_shards` should be a multiple of `num_gpus` (workers) for optimal load balancing
  - Example: 8 workers → use 8, 16, 24, 32, 40, 48... shards
  - The pipeline will automatically adjust if `num_shards < num_gpus` to ensure each worker has work

## Processing Large Datasets via Config

For large datasets, simply modify the `dataset_split` field to process subsets:

```json
{
  "dataset_name": "massive_dataset",
  "config_name": "some_config",
  "dataset_split": "train[:10000000]",  // Process first 10M samples
  "mode": "image_only",
  "tokenizer_path": "/path/to/tokenizer",
  "output_dir": "./output",
  "num_gpus": 8,
  "num_shards": 1000,  // Enable shard-based checkpointing
  "device": "cuda"
}
```

**Dataset split examples:**
- `"dataset_split": "train[:10000000]"` - First 10M samples
- `"dataset_split": "train[10000000:20000000]"` - Samples 10M to 20M
- `"dataset_split": "train[:50%]"` - First half of dataset
- `"dataset_split": "train[-5000000:]"` - Last 5M samples

**Note:** Vision tokenizer type and path are automatically loaded from the omni-tokenizer's `tokenizer_config.json`.

Use with:
```bash
python tokenize.py hf --config config.json
```

CLI arguments override config file values.

## Output Format

The pipeline creates Megatron-LM IndexedDataset format with shard-based files:

```
output_dir/
└── config_name/
    ├── rank_0_shard_3_32.bin   # Binary token data (worker 0, shard 3 of 32 total)
    ├── rank_0_shard_3_32.idx   # Index for random access
    ├── rank_0_shard_4_32.bin   # Binary token data (worker 0, shard 4 of 32 total)
    ├── rank_0_shard_4_32.idx
    ├── rank_1_shard_8_32.bin   # Binary token data (worker 1, shard 8 of 32 total)
    ├── rank_1_shard_8_32.idx
    ├── rank_2_shard_16_32.bin  # Binary token data (worker 2, shard 16 of 32 total)
    ├── rank_2_shard_16_32.idx
    └── dataset_info.json       # Processing metadata (written after completion)
```

The filename format `rank_{worker}_shard_{id}_{total}` enables:
- **Atomic checkpointing**: Each shard is saved independently
- **Easy resume**: The total shard count is embedded in filenames
- **Clear tracking**: You can see progress at a glance

## Checkpoint and Resume

### How It Works

1. Each shard saves to `rank_{worker}_shard_{id}_{total}.bin` and `.idx`
2. The `.idx` file marks completion (written last)
3. Resume detects completed shards by checking for `.idx` files
4. Only uncompleted shards are reprocessed

### Resume Usage

```bash
# Initial run
python tokenize.py hf --config config.json

# Resume after interruption
python tokenize.py hf --config config.json --resume
```

### Edge Cases

**Incomplete shards** (`.bin` without `.idx`):
- Automatically overwritten when reprocessed

**Inconsistent shard counts**:
- Program stops with error if existing files have different total shards
- Example error:
  ```
  ERROR: Inconsistent total shard counts found: [32, 64]
    32 total shards: 20 files
    64 total shards: 10 files
  Clean the output directory or use a different output path
  ```

**Mismatched resume**:
- If resuming with different `num_shards`, program stops:
  ```
  ERROR: No existing shards match expected count (100). Found shard counts: [32]
  To resume, use --num-shards 32 or start fresh without --resume
  ```


## Examples

### Example 1: Image-Only Pretraining Dataset

```bash
python tokenize.py hf \
    --mode image_only \
    --dataset-name laion/laion-high-resolution \
    --dataset-split train \
    --tokenizer-path ./llama3_emu3_base \
    --output-dir ./tokenized_data \
    --num-gpus 8 \
    --num-shards 800 \
    --device cuda \
    --min-tokenizer-pixels "512*512" \
    --max-tokenizer-pixels "1024*1024" \
    --min-image-pixels "256*256" \
    --max-image-pixels "2048*2048"
```

### Example 2: SFT with FineVision

```bash
python tokenize.py hf \
    --mode sft \
    --dataset-name HuggingFaceM4/FineVision \
    --config-name CoSyn_400k_chart \
    --dataset-split train \
    --tokenizer-path ./llama3_emu3_instruct \
    --output-dir ./sft_data \
    --num-gpus 4 \
    --num-shards 100 \
    --device cuda \
    --max-samples 1000  # For testing
```

### Example 3: Using Emu3.5

```bash
# Just use an Emu3.5 omni-tokenizer - vision tokenizer is auto-loaded!
python tokenize.py hf \
    --mode image_only \
    --dataset-name ... \
    --tokenizer-path ./llama3_emu3.5_base \
    --output-dir ./emu3.5_data \
    --num-gpus 4 \
    --num-shards 100 \
    --device cuda
```

## Performance Tips

1. **Shard Count**: Choose appropriate `--num-shards` based on dataset size (see recommendations above)
2. **GPU Count**: More GPUs = faster processing with Ray's work-stealing
3. **Resolution**: Lower max pixels = faster tokenization but lower quality
4. **Filtering**: Use min/max image pixels to skip unwanted images early
5. **Load Balancing**: Set `num_shards` as a multiple of `num_gpus` for optimal distribution

## Troubleshooting

### Out of Memory
- Reduce `--max-tokenizer-pixels`
- Increase `--num-shards` to reduce samples per shard
- Use fewer GPUs

### Slow Processing
- Ensure `--num-shards` is a multiple of `--num-gpus`
- Use more GPUs with `--num-gpus`
- Check GPU utilization with `nvidia-smi`

### Missing Images or Text
- Check `--image-field` and `--text-field` match your dataset
- Use `--max-samples 10` to test on small subset first

## Architecture

The pipeline uses:
- **Ray** for distributed GPU processing with dynamic work-stealing
- **HuggingFace Datasets** for efficient data loading via `.shard()` method
- **Megatron-LM IndexedDataset** for training-optimized output format
- **Shard-based processing** for atomic checkpointing and resume capability

Workers pull shards dynamically from a shared queue, ensuring optimal GPU utilization. Each shard is processed independently and saved as a separate file pair (`rank_XXX_shard_YYYYY.bin/idx`), enabling robust checkpointing and recovery.

## Related Tools

- **Omni-Tokenizer Creation**: See `utils/omni_tokenizer/README.md`
- **Old WebDataset Pipeline**: See `README_OLD_WEBDATASET.md` (legacy)
