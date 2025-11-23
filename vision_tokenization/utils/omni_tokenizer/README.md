# Omni-Tokenizer

Create omnimodal tokenizers by adding vision tokens to text tokenizers with automatic codebook size detection.

## Quick Start

**Base Tokenizer (for pretraining):**
```bash
python -m vision_tokenization.utils.omni_tokenizer.create_base \
    --text-tokenizer-path meta-llama/Llama-3.2-3B \
    --vision-tokenizer-path BAAI/Emu3-VisionTokenizer \
    --vision-tokenizer Emu3 \
    --output-path ./llama3_emu3_tokenizer
```

**Instruct Tokenizer (for SFT/chat):**
```bash
python -m vision_tokenization.utils.omni_tokenizer.create_instruct \
    --text-tokenizer-path meta-llama/Llama-3.2-3B-Instruct \
    --vision-tokenizer-path BAAI/Emu3-VisionTokenizer \
    --vision-tokenizer Emu3 \
    --output-path ./llama3_emu3_instruct_tokenizer
```

## Design

The omni-tokenizer extends text tokenizers with vision capabilities while maintaining future extensibility:

**Token Allocation:**
1. Original text tokens (e.g., 128,256 for LLaMA-3.2-3B)
2. 200 RESERVED_OMNI tokens for multimodal expansion
3. Visual tokens (auto-detected from vision tokenizer)

**Image Structure Tokens (auto-renamed from RESERVED_OMNI):**
- `<|RESERVED_OMNI_001|>` → `<|img_start|>` - Image sequence boundary
- `<|RESERVED_OMNI_002|>` → `<|img_end|>` - Image sequence end
- `<|RESERVED_OMNI_003|>` → `<|img_token_start|>` - Individual token marker
- `<|RESERVED_OMNI_004|>` → `<|img_end_of_row|>` - Row delimiter
- `<|RESERVED_OMNI_005|>` → `<|img_end_of_frame|>` - Frame delimiter
- `<|RESERVED_OMNI_006|>` → `<|img_generation_start|>` - Generation mode marker
- `<|RESERVED_OMNI_007|>` → `<|image|>` - Image placeholder

**Visual Tokens:**
Codebook size is automatically detected from the vision tokenizer:
- Emu3VisionTokenizer: 32,768 tokens → format `<|visual token XXXXX|>` (5 digits)
- Emu3_5_IBQ: 131,072 tokens → format `<|visual token XXXXXX|>` (6 digits)

The number of digits adjusts automatically based on the codebook size.

## Supported Vision Tokenizers

- **Emu3VisionTokenizer** - 32,768 visual tokens
- **Emu3_5_IBQ** - 131,072 visual tokens (Emu3.5 with IBQ)

## Architecture

**Two tokenizer types:**

1. **Base Tokenizer** (`create_base.py`):
   - Extends base text tokenizer (e.g., `meta-llama/Llama-3.2-3B`)
   - Adds RESERVED_OMNI tokens + vision tokens
   - Used for pretraining multimodal models

2. **Instruct Tokenizer** (`create_instruct.py`):
   - Extends instruct text tokenizer (e.g., `meta-llama/Llama-3.2-3B-Instruct`)
   - Adds RESERVED_OMNI tokens + vision tokens (same as base)
   - Preserves chat template from instruct text tokenizer
   - Adds SFT sequences for Megatron-LM training
   - Used for supervised fine-tuning and chat applications

**Why separate architectures?**
- Base: Clean starting point for pretraining
- Instruct: Inherits chat template + any instruct-specific tokens automatically
- Mirrors Meta's approach: Vision-Instruct extends from text instruct tokenizer

**Important notes about base vs instruct text tokenizers:**
- Base and instruct text tokenizers typically have the **same vocabulary size**
- However, instruct tokenizers may repurpose some `reserved_special_token` slots for chat functionality
- Example: LLaMA-3.2-Vision-Instruct replaces `<|reserved_special_token_2|>` with `<|step_id|>`
- Our omni-tokenizer adds vision tokens **after** the text vocabulary, so these differences are preserved

## Examples

### Base Tokenizer

**Emu3:**
```bash
python -m vision_tokenization.utils.omni_tokenizer.create_base \
    --text-tokenizer-path meta-llama/Llama-3.2-3B \
    --vision-tokenizer-path BAAI/Emu3-VisionTokenizer \
    --vision-tokenizer Emu3 \
    --output-path ./llama3_emu3_base
```

**Emu3.5 IBQ:**
```bash
python -m vision_tokenization.utils.omni_tokenizer.create_base \
    --text-tokenizer-path meta-llama/Llama-3.2-3B \
    --vision-tokenizer-path /path/to/Emu3.5-VisionTokenizer \
    --vision-tokenizer Emu3.5 \
    --output-path ./llama3_emu3.5_base
```

### Instruct Tokenizer

**Emu3:**
```bash
python -m vision_tokenization.utils.omni_tokenizer.create_instruct \
    --text-tokenizer-path meta-llama/Llama-3.2-3B-Instruct \
    --vision-tokenizer-path BAAI/Emu3-VisionTokenizer \
    --vision-tokenizer Emu3 \
    --output-path ./llama3_emu3_instruct
```

**Emu3.5 IBQ:**
```bash
python -m vision_tokenization.utils.omni_tokenizer.create_instruct \
    --text-tokenizer-path meta-llama/Llama-3.2-3B-Instruct \
    --vision-tokenizer-path /path/to/Emu3.5-VisionTokenizer \
    --vision-tokenizer Emu3.5 \
    --output-path ./llama3_emu3.5_instruct
```

## Options

### create_base.py

**Required:**
- `--text-tokenizer-path` - Path or HuggingFace model ID for base text tokenizer
- `--vision-tokenizer-path` - Path to vision tokenizer model directory
- `--vision-tokenizer` - Vision tokenizer type (Emu3 or Emu3.5)
- `--output-path` - Output directory for the omni-tokenizer

**Optional:**
- `--num-reserved-tokens` - Number of RESERVED_OMNI tokens (default: 200)

### create_instruct.py

**Required:**
- `--text-tokenizer-path` - Path or HuggingFace model ID for instruct text tokenizer
- `--vision-tokenizer-path` - Path to vision tokenizer model directory
- `--vision-tokenizer` - Vision tokenizer type (Emu3 or Emu3.5)
- `--output-path` - Output directory for the omni-tokenizer

**Note:** Instruct tokenizer automatically inherits the chat template from the text tokenizer.

## Output

The created tokenizer includes:
- Standard tokenizer files (tokenizer.json, tokenizer_config.json, etc.)
- `vision_token_mapping.json` - Metadata about vision token ranges and mappings

Example output for Emu3.5:
```
Original vocabulary size: 128,256
RESERVED_OMNI tokens:     200
Visual tokens added:      131,072
Final vocabulary size:    259,528
```
