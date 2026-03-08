# Vision Tokenization Pipeline

Converts image datasets into Megatron-LM IndexedDataset format (`.bin`/`.idx`) for training. Supports HuggingFace and WebDataset sources, multiple tokenization modes, batching strategies, and multi-image samples.

## Quick Start

```bash
# HuggingFace dataset
python -m vision_tokenization.tokenize hf --config path/to/config.json

# WebDataset (tar files)
python -m vision_tokenization.tokenize wds --config path/to/config.json

# Resume from checkpoint
python -m vision_tokenization.tokenize hf --config path/to/config.json --resume
```

## Pipelines

### HuggingFace Pipeline (`hf`)

Processes HuggingFace datasets with explicit sharding (`--num-shards`).

**Required config fields:** `dataset_name`, `dataset_split`, `mode`, `num_shards`

**Dataset loading methods** (`dataset_load_method`):

| Method | When to use |
|--------|-------------|
| `default` | Standard `load_dataset()`. Requires HF Hub access. |
| `builder_load` | Uses `load_dataset_builder().as_dataset()`. For restricted environments without HF Hub cache. Dataset must be pre-prepared. |
| `disk_load` | Uses `load_from_disk()` for datasets saved with `dataset.save_to_disk()`. `dataset_name` is a local path. |

Additional HF options: `config_name` (subset), `cache_dir`, `num_proc`, `max_samples`, `dataset_streamed`, `data_files`.

### WebDataset Pipeline (`wds`)

Processes tar-based datasets. Shards are auto-discovered from `input_pattern` (glob or braceexpand).

**Required config fields:** `input_pattern`, `mode`

**Key differences from HF:**
- No `num_shards` — each tar file is a shard
- Image/text fields use semicolon-separated extension keys: `"jpg;png;jpeg;webp"`
- Supports braceexpand patterns: `"data_{000..100}.tar"`
- `--skip-sample-count` skips slow tar header scanning for large collections

## Tokenization Modes

| Mode | Description | Token structure |
|------|-------------|-----------------|
| `image_only` | Images only, no text | `[BOS][img_struct][EOS]` |
| `image2text` | Image-text pairs (captioning) | `[BOS][img_struct][text][EOS]` |
| `text2image` | Text-image pairs (generation) | `[BOS][text][img_struct][EOS]` |
| `sft` | Supervised fine-tuning with chat template | `[BOS][chat_template_with_images][EOS]` |

Where `img_struct` = `img_start + dims + img_token_start + vision_tokens + EOLs + EOF + img_end`.

## Single Image vs Multi-Image

### Single Image (default)

Each sample has one image. Specified via `image_field` (column name for HF, semicolon-separated extension keys for WDS).

### Multi-Image

Enabled by setting `image_field_pattern` (e.g., `"img"`). Auto-discovers all matching fields per sample, sorted alphabetically.

- **HF**: Matches column names starting with pattern (`img1`, `img2`, `img3`)
- **WDS**: Matches decoded sample keys starting with pattern (`img1.png`, `img2.png`)

**Token structure** (paired modes):
```
image2text: [BOS][img1_struct][img2_struct]...[text][EOS]
text2image: [BOS][text][img1_struct][img2_struct]...[EOS]
```

**SFT mode**: Images are matched to `<|image|>` placeholders in conversation text in order. Conversations must contain the correct number of placeholders.

**Constraints:**
- Incompatible with `batch_mode` (raises `ValueError` at startup)
- Incompatible with `image_only` mode
- At least 1 image must match the pattern; samples with 0 matches are skipped
- ALL images must pass resolution filtering; if any fails, the entire sample is skipped

**Example config:**
```json
{
  "image_field_pattern": "img",
  "text_field": "txt",
  "mode": "image2text",
  "batch_mode": null
}
```

CLI: `--image-field-pattern "img"`

## Batching

Batching groups images by size to reduce padding waste during GPU tokenization. Configured via `batch_mode` and `batch_size`.

| Strategy | Description |
|----------|-------------|
| `None` | No batching. Processes one sample at a time. Required for multi-image. |
| `simple` | Groups contiguous samples into fixed-size batches. |
| `sorted` | Sorts by aspect ratio before batching. Default for HF/WDS pipelines. Reduces padding. |
| `clustered` | K-means clustering on aspect ratio + log area. Best padding efficiency. Requires `fastkmeans`. |

**Additional batching options:**
- `batch_size`: Number of images per batch (default: 1)
- `buffer_size`: Buffer size for sorted/clustered batchers (default: `batch_size * 20`)
- `resize_size`: How to resize images in a batch — `"avg"`, `"max"`, or explicit `(H, W)` tuple

## Resolution Filtering

Two independent resolution controls:

### Tokenizer resolution (`min_tokenizer_pixels`, `max_tokenizer_pixels`)

Passed to the EMU3 tokenizer's image processor. Controls the pixel range for tokenizer preprocessing (resizing/padding). Does **not** skip samples.

### Image resolution filter (`min_image_pixels`, `max_image_pixels`)

Pre-tokenization filter. Images outside this range are **skipped** (not resized). Uses `image.size` (O(1), no decode needed for WDS).

Format: `"384*384"` (parsed as product = 147456 pixels) or raw integer.

## Filtering and Failure Modes

Every sample is checked before tokenization. The outcome is one of:

| Status | Cause | Behavior |
|--------|-------|----------|
| `OK` | All checks pass | Tokenized normally |
| `DATA_SKIP` | Missing image, missing text (when required), or empty multi-image list | Skipped, counted in `stats["skipped"]` |
| `RESOLUTION_SKIP_MIN` | Image below `min_image_pixels` | Skipped, counted in `stats["resolution_skipped_min"]` |
| `RESOLUTION_SKIP_MAX` | Image above `max_image_pixels` | Skipped, counted in `stats["resolution_skipped_max"]` |

### Multi-image specific filtering

- **No matching images** (0 keys match pattern): `DATA_SKIP`
- **Any image is None**: `DATA_SKIP`
- **Any image fails resolution check**: Entire sample skipped with the corresponding resolution status
- **Missing text** (for `image2text`/`text2image`/`sft`): `DATA_SKIP`

### Runtime errors

| Error | Handling |
|-------|----------|
| CUDA OOM | Caught, logged, sample skipped. Counted in `stats["cuda_oom_errors"]` |
| Transform failure | Sample skipped. Counted in `stats["transform_errors"]` |
| Corrupt/undecodable image | Sample skipped. Counted in `stats["errors"]` |
| Tokenization error | Sample skipped. Counted in `stats["errors"]` |

All errors are non-fatal — the pipeline continues processing remaining samples.

## Data Transforms

Optional transforms applied after data extraction, before tokenization.

```json
{
  "image_transforms": "convert_rgb,resize_max",
  "text_transforms": "remove_string",
  "transform_params": {
    "resize_max": {"max_size": 1024},
    "remove_string": {"strings": ["<unwanted>"]}
  }
}
```

For multi-image samples, image transforms are applied to each image individually.

Built-in: `convert_rgb`, `resize_max`, `random_augment`, `remove_string`, `strip_whitespace`.

## Resume and Checkpointing

Pass `--resume` to skip already-completed shards. The pipeline checks for existing output files per shard and skips them.

Each shard produces its own `.bin`/`.idx` files named `rank_{worker_id}_shard_{shard_id}_{num_shards}`.

## Progress and SLURM

- **TTY detected**: tqdm progress bars
- **Non-TTY (piped output)**: Line-based logging at configurable `log_interval` (default: 1000 samples)
- **SLURM**: Auto-detects time limit from environment. Override with `--slurm-time-limit "12:00:00"`. Warns when approaching timeout.

## Common Configuration Fields

| Field | Default | Description |
|-------|---------|-------------|
| `tokenizer_path` | required | Path to EMU3 tokenizer |
| `output_dir` | required | Output directory |
| `num_gpus` | required | GPUs per node |
| `device` | required | `"cuda"` or `"cpu"` |
| `mode` | required | `image_only`, `image2text`, `text2image`, `sft` |
| `batch_mode` | `"sorted"` | `null`, `"simple"`, `"sorted"`, `"clustered"` |
| `batch_size` | 1 | Images per batch |
| `min_tokenizer_pixels` | — | Min pixels for tokenizer preprocessing |
| `max_tokenizer_pixels` | — | Max pixels for tokenizer preprocessing |
| `min_image_pixels` | — | Min pixels filter (skip below) |
| `max_image_pixels` | — | Max pixels filter (skip above) |
| `image_field_pattern` | `null` | Multi-image pattern prefix |
| `image_transforms` | — | Comma-separated transform names |
| `text_transforms` | — | Comma-separated transform names |
| `log_interval` | 1000 | Samples between progress logs |
| `conversation_transform` | — | SFT conversation format transform |

See `vision_tokenization/configs/` for complete example configurations.
