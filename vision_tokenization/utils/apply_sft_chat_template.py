#!/usr/bin/env python3
"""
Apply chat template to SFT dataset and prepare for Megatron tokenization.
Supports both parquet and jsonl output formats.

Example usage:
    # Apply chat template to FineVision text dataset (save as parquet)
    # BOS/EOS tokens are added by default if missing
    python3 -m vision_tokenization.utils.apply_sft_chat_template \
        --dataset HuggingFaceM4/FineVision \
        --config text_numinamath_cot \
        --output-dir /capstor/store/cscs/swissai/infra01/vision-datasets \
        --tokenizer-path /capstor/store/cscs/swissai/infra01/MLLM/llama3_vision_instruct_emu3_tokenizer \
        --num-proc 64

    # Process multiple configs (comma-separated) - creates subfolders
    python3 -m vision_tokenization.utils.apply_sft_chat_template \
        --dataset HuggingFaceM4/FineVision \
        --config lrv_chart,text_numinamath_cot,ocr_data \
        --output-dir /output \
        --tokenizer-path /path/to/tokenizer

    # Process multiple configs from a file (one per line) - creates subfolders
    python3 -m vision_tokenization.utils.apply_sft_chat_template \
        --dataset HuggingFaceM4/FineVision \
        --config @configs.txt \
        --output-dir /output \
        --tokenizer-path /path/to/tokenizer

    # Use conversation transform with debug output
    python3 -m vision_tokenization.utils.apply_sft_chat_template \
        --dataset HuggingFaceM4/FineVision \
        --config lrv_chart \
        --output-dir /output \
        --tokenizer-path /path/to/tokenizer \
        --conversation-transform finevision_to_llama \
        --debug-samples 3

    # Apply chat template to dataset (save as jsonl)
    python3 -m vision_tokenization.utils.apply_sft_chat_template \
        --dataset HuggingFaceH4/ultrachat_200k \
        --output-dir /capstor/store/cscs/swissai/infra01/vision-datasets \
        --output-format jsonl \
        --tokenizer-path /capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3_tokenizer \
        --conversation-transform normalize_format \
        --debug-samples 5 \
        --max-samples 100000

    # Disable BOS/EOS addition if needed
    python3 -m vision_tokenization.utils.apply_sft_chat_template \
        --dataset HuggingFaceM4/FineVision \
        --config lrv_chart \
        --output-dir /output \
        --tokenizer-path /path/to/tokenizer \
        --no-add-bos --no-add-eos

    # Use llava_to_apertus in text-only mode (no image placeholder)
    python3 -m vision_tokenization.utils.apply_sft_chat_template \
        --dataset mvp-lab/LLaVA-OneVision-1.5-Instruct-Data \
        --output-dir /output \
        --tokenizer-path /path/to/tokenizer \
        --conversation-transform llava_to_apertus \
        --transform-params '{"llava_to_apertus": {"add_image_placeholder": false}}'

    # Or load transform params from a file
    python3 -m vision_tokenization.utils.apply_sft_chat_template \
        --dataset mvp-lab/LLaVA-OneVision-1.5-Instruct-Data \
        --output-dir /output \
        --tokenizer-path /path/to/tokenizer \
        --conversation-transform llava_to_apertus \
        --transform-params @my_params.json

    # Use pre-prepared dataset on cluster (offline, builder_load method)
    python3 -m vision_tokenization.utils.apply_sft_chat_template \
        --dataset HuggingFaceM4/FineVision \
        --config lrv_chart \
        --output-dir /output \
        --tokenizer-path /path/to/tokenizer \
        --cache-dir /path/to/hf_cache \
        --dataset-load-method builder_load \
        --offline

    # Load a dataset saved with dataset.save_to_disk() (disk_load method)
    python3 -m vision_tokenization.utils.apply_sft_chat_template \
        --dataset /path/to/saved_dataset \
        --output-dir /output \
        --tokenizer-path /path/to/tokenizer \
        --dataset-load-method disk_load

    # Then tokenize with datatrove (works with both parquet and jsonl)
    # IMPORTANT: Use --no-add-special-tokens to avoid double BOS token
    # (chat template already added BOS/EOS)
    python3 /iopsstor/scratch/cscs/xyixuan/data-pipeline-pretrain/examples/tokenize_megatron/preprocess_megatron.py \
        --tokenizer-name-or-path /capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3_tokenizer \
        --output-folder datasets/openmathinstruct_tokenized \
        --n-tasks 16 \
        --no-add-special-tokens \
        jsonl \
        --dataset text_OpenMathInstruct-2_formatted.parquet \
        --column text
"""

import argparse
import json
import os
from typing import Any, Dict

from transformers import AutoTokenizer

from vision_tokenization.pipelines.hf.dataset_loader import load_hf_dataset

from vision_tokenization.vokenizers.conversation_transforms import (
    ConversationTransformRegistry,
)


def parse_transform_params(params_arg: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse transform params from JSON string or file.

    Args:
        params_arg: JSON string or @file path with transform parameters.
            Format: '{"transform_name": {"param": value}}' or '@params.json'

    Returns:
        Dict mapping transform names to their parameters
    """
    if params_arg is None:
        return {}

    if params_arg.startswith("@"):
        file_path = params_arg[1:]
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Transform params file not found: {file_path}")
        with open(file_path, "r") as f:
            return json.load(f)

    return json.loads(params_arg)


def parse_configs(config_arg: str) -> list:
    """
    Parse config argument to list of config names.

    Supports:
        - Single config: "lrv_chart"
        - Comma-separated: "lrv_chart,text_numinamath_cot,ocr_data"
        - File path (prefixed with @): "@configs.txt"
        - File path (if file exists): "/path/to/configs.txt"

    Args:
        config_arg: Config argument string

    Returns:
        List of config names
    """
    if config_arg is None:
        return [None]

    # Check if it's a file reference (starts with @)
    if config_arg.startswith("@"):
        file_path = config_arg[1:]
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")
        with open(file_path, "r") as f:
            configs = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        return configs

    # Check if it's a file path (exists as file)
    if os.path.isfile(config_arg):
        with open(config_arg, "r") as f:
            configs = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        return configs

    # Otherwise treat as comma-separated list
    configs = [c.strip() for c in config_arg.split(",") if c.strip()]
    return configs


def normalize_messages(messages):
    """
    Normalize messages to standard format efficiently.

    Handles two formats:
    1. Standard: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    2. Conversation: [{"user": "...", "assistant": "..."}]

    Returns standard format.
    """
    if not messages or not isinstance(messages, list):
        return messages

    first = messages[0]
    if not isinstance(first, dict):
        return messages

    # Check format and convert if needed (single pass)
    if "role" in first:  # Already correct format
        return messages

    if "user" in first:  # Conversation format - convert
        return [
            {"role": role, "content": conv[role]} for conv in messages for role in ("user", "assistant") if role in conv
        ]

    return messages  # Unknown format, return as-is


def print_debug(raw, transformed, formatted, idx):
    """Print debug information for a sample."""
    print(f"\n{'='*60}")
    print(f"DEBUG SAMPLE {idx + 1}")
    print(f"{'='*60}")
    print("\n--- RAW INPUT ---")
    raw_str = json.dumps(raw, indent=2, default=str, ensure_ascii=False)
    print(raw_str[:2000] + ("..." if len(raw_str) > 2000 else ""))
    print("\n--- AFTER TRANSFORM ---")
    transformed_str = json.dumps(transformed, indent=2, default=str, ensure_ascii=False)
    print(transformed_str[:2000] + ("..." if len(transformed_str) > 2000 else ""))
    print("\n--- FORMATTED TEXT ---")
    print(formatted[:2000] + ("..." if len(formatted) > 2000 else ""))
    print(f"{'='*60}\n")


def process_single_config(
    dataset_name: str,
    config: str,
    split: str,
    output_dir: str,
    output_format: str,
    tokenizer,
    message_column: str,
    add_bos: bool,
    add_eos: bool,
    num_proc: int,
    conversation_transform,
    debug_samples: int,
    max_samples: int,
    use_subfolder: bool,
    cache_dir: str = None,
    dataset_load_method: str = "default",
) -> dict:
    """
    Process a single dataset config.

    Returns:
        Dict with processing results (count, errors, output_file, bos_added_by_template)
    """
    # Determine output path
    if use_subfolder and config:
        config_output_dir = os.path.join(output_dir, config)
        os.makedirs(config_output_dir, exist_ok=True)
        filename = f"{config}_formatted.{output_format}"
        output_file = os.path.join(config_output_dir, filename)
    else:
        os.makedirs(output_dir, exist_ok=True)
        if config:
            filename = f"{config}_formatted.{output_format}"
        else:
            # Use last part of dataset name (e.g., HuggingFaceH4/ultrachat_200k -> ultrachat_200k)
            ds_name = dataset_name.split("/")[-1]
            filename = f"{ds_name}_formatted.{output_format}"
        output_file = os.path.join(output_dir, filename)

    print(f"\n{'#'*60}")
    print(f"Processing config: {config or 'default'}")
    print(f"{'#'*60}")
    print(f"Loading dataset: {dataset_name}")
    if config:
        print(f"Config: {config}")
    print(f"Split: {split}")
    print(f"Message column: {message_column}")

    # Load dataset
    dataset = load_hf_dataset(
        dataset_name=dataset_name,
        config_name=config,
        split=split,
        cache_dir=cache_dir,
        num_proc=num_proc,
        method=dataset_load_method,
    )

    # Select subset if max_samples is specified
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Processing {len(dataset)} samples")

    # Get BOS/EOS tokens from tokenizer
    bos_token = tokenizer.bos_token if add_bos else None
    eos_token = tokenizer.eos_token if add_eos else None

    # Counter for debug samples
    debug_counter = {"count": 0}

    # Define processing function
    def apply_chat_template_to_sample(example):
        """Apply chat template to a single sample."""
        try:
            # Get raw messages from the specified column
            raw_messages = example[message_column]

            # Apply conversation transform or fallback to normalize_messages
            if conversation_transform:
                messages = conversation_transform.transform(raw_messages)
            else:
                messages = normalize_messages(raw_messages)

            # Apply chat template
            # LLAMA3.2 Vision Instruct Tokenizer hardcode the BOS token
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            # Add BOS/EOS if requested
            if add_bos and bos_token and not formatted_text.startswith(bos_token):
                formatted_text = bos_token + formatted_text
            if add_eos and eos_token and not formatted_text.endswith(eos_token):
                formatted_text = formatted_text + eos_token

            # Debug output (only for first N samples, single-process to avoid race)
            if debug_samples > 0 and debug_counter["count"] < debug_samples:
                print_debug(raw_messages, messages, formatted_text, debug_counter["count"])
                debug_counter["count"] += 1

            # Add new 'text' field
            example["text"] = formatted_text
            return example
        except Exception as e:
            print(f"Error processing sample: {e}")
            example["text"] = ""  # Add empty text for failed samples
            return example

    # Process dataset - use single process if debug mode to ensure proper output ordering
    effective_num_proc = 1 if debug_samples > 0 else (num_proc if num_proc else 1)
    if debug_samples > 0:
        print(f"\nDebug mode enabled: printing {debug_samples} sample(s) (using single process)")

    print(f"\nApplying chat template to dataset...")
    processed_dataset = dataset.map(
        apply_chat_template_to_sample, num_proc=effective_num_proc, desc=f"Applying chat template ({config or 'default'})"
    )

    # Save to appropriate format
    print(f"\nSaving to {output_file}...")
    if output_format == "parquet":
        processed_dataset.to_parquet(output_file)
    elif output_format == "jsonl":
        processed_dataset.to_json(output_file, orient="records", lines=True)

    count = len(processed_dataset)
    errors = sum(1 for sample in processed_dataset if sample.get("text", "") == "")

    # Check if chat template added BOS token (verify first few samples)
    sample_texts = [processed_dataset[i]["text"] for i in range(min(10, count)) if processed_dataset[i].get("text")]
    bos_added_by_template = (
        all(text.startswith(tokenizer.bos_token) for text in sample_texts) if tokenizer.bos_token else False
    )

    print(f"\nConfig '{config or 'default'}' complete!")
    print(f"  Samples processed: {count:,}")
    print(f"  Errors: {errors:,}")
    print(f"  Output: {output_file}")

    return {
        "config": config,
        "count": count,
        "errors": errors,
        "output_file": output_file,
        "bos_added_by_template": bos_added_by_template,
    }


def main():
    # Handle --list-transforms early before full argument parsing
    import sys

    if "--list-transforms" in sys.argv:
        print("Available conversation transforms:")
        for name in ConversationTransformRegistry.list_transforms():
            transform_cls = ConversationTransformRegistry.get_transform(name)
            doc = transform_cls.__doc__ or "No description"
            # Get first line of docstring
            first_line = doc.strip().split("\n")[0]
            print(f"  {name}: {first_line}")
        return

    parser = argparse.ArgumentParser(description="Apply chat template to SFT dataset")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name or path")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Dataset configuration(s). Supports: single config name, comma-separated list "
            "(e.g., 'config1,config2'), path to txt file with @prefix (e.g., '@configs.txt'), "
            "or direct path to txt file. When multiple configs are given, outputs are placed in subfolders."
        ),
    )
    parser.add_argument("--split", default="train", help="Dataset split to use")
    parser.add_argument("--output-dir", required=True, help="Output directory for formatted files")
    parser.add_argument(
        "--output-format", default="parquet", choices=["parquet", "jsonl"], help="Output format (default: parquet)"
    )
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer for chat template")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to process per config")
    parser.add_argument("--message-column", default="texts", help="Column containing messages")
    parser.add_argument("--add-bos", action=argparse.BooleanOptionalAction, default=True,
                        help="Add BOS token if missing (default: True, use --no-add-bos to disable)")
    parser.add_argument("--add-eos", action=argparse.BooleanOptionalAction, default=True,
                        help="Add EOS token if missing (default: True, use --no-add-eos to disable)")
    parser.add_argument("--num-proc", type=int, default=64, help="Number of processes for parallel processing")
    parser.add_argument(
        "--conversation-transform",
        default=None,
        help=(
            "Name of conversation transform to apply (e.g., 'finevision_to_llama', 'normalize_format', "
            "'llava_to_apertus'). Use --list-transforms to see available transforms."
        ),
    )
    parser.add_argument(
        "--list-transforms",
        action="store_true",
        help="List available conversation transforms and exit",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=0,
        help="Number of samples to print debug info for (shows raw input, transformed, and formatted output)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace datasets cache directory",
    )
    parser.add_argument(
        "--dataset-load-method",
        type=str,
        choices=["default", "builder_load", "disk_load"],
        default="default",
        help=(
            "Dataset loading method. 'default' uses load_dataset(), "
            "'builder_load' uses load_dataset_builder().as_dataset() "
            "(useful on clusters with pre-prepared datasets), "
            "'disk_load' uses load_from_disk() for datasets saved with "
            "dataset.save_to_disk() (dataset name is a local path)"
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Enable HF datasets offline mode (sets HF_DATASETS_OFFLINE=1)",
    )
    parser.add_argument(
        "--transform-params",
        type=str,
        default=None,
        help=(
            "JSON string or @file path with transform parameters. "
            "Format: '{\"transform_name\": {\"param\": value}}' or '@params.json'. "
            "Example: '{\"llava_to_apertus\": {\"add_image_placeholder\": false}}'"
        ),
    )

    args = parser.parse_args()

    # Set offline mode before any dataset loading
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        print("HF Datasets Offline Mode: ENABLED")

    # Parse configs
    configs = parse_configs(args.config)
    use_subfolder = len(configs) > 1

    print(f"Dataset: {args.dataset}")
    print(f"Configs to process: {configs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output format: {args.output_format}")
    if use_subfolder:
        print(f"Using subfolders: Yes (multiple configs)")

    print(f"\nLoading tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Verify tokens exist
    if args.add_bos:
        assert tokenizer.bos_token is not None, "Tokenizer does not have a BOS token defined"
        print(f"BOS token: Enabled ({tokenizer.bos_token}, ID: {tokenizer.bos_token_id})")
    else:
        print(f"BOS token: Disabled")

    if args.add_eos:
        assert tokenizer.eos_token is not None, "Tokenizer does not have an EOS token defined"
        print(f"EOS token: Enabled ({tokenizer.eos_token}, ID: {tokenizer.eos_token_id})")
    else:
        print(f"EOS token: Disabled")

    # Parse transform params and create conversation transform if specified
    transform_params = parse_transform_params(args.transform_params)
    conversation_transform = None
    if args.conversation_transform:
        transform_cls = ConversationTransformRegistry.get_transform(args.conversation_transform)
        params = transform_params.get(args.conversation_transform, {})
        conversation_transform = transform_cls(**params)
        if params:
            print(f"Conversation transform: {args.conversation_transform} with params: {params}")
        else:
            print(f"Conversation transform: {args.conversation_transform}")
    else:
        print("Conversation transform: None (using default normalize_messages)")

    # Process each config
    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Processing config: {config or 'default'}")
        result = process_single_config(
            dataset_name=args.dataset,
            config=config,
            split=args.split,
            output_dir=args.output_dir,
            output_format=args.output_format,
            tokenizer=tokenizer,
            message_column=args.message_column,
            add_bos=args.add_bos,
            add_eos=args.add_eos,
            num_proc=args.num_proc,
            conversation_transform=conversation_transform,
            debug_samples=args.debug_samples,
            max_samples=args.max_samples,
            use_subfolder=use_subfolder,
            cache_dir=args.cache_dir,
            dataset_load_method=args.dataset_load_method,
        )
        results.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE - SUMMARY")
    print(f"{'='*60}")

    total_count = sum(r["count"] for r in results)
    total_errors = sum(r["errors"] for r in results)

    print(f"\nConfigs processed: {len(results)}")
    print(f"Total samples: {total_count:,}")
    print(f"Total errors: {total_errors:,}")

    print(f"\nOutput files:")
    for r in results:
        print(f"  {r['config'] or 'default'}: {r['output_file']}")

    # Token verification (use first result)
    if results:
        bos_added_by_template = results[0]["bos_added_by_template"]
        print(f"\nToken verification:")
        if bos_added_by_template:
            print(f"  BOS token: Added by chat template (hardcoded in template)")
        else:
            print(f"  BOS token: NOT added by chat template")
            if not args.add_bos:
                print(f"  Warning: BOS token not present - consider removing --no-add-bos")

        if args.add_eos:
            print(f"  EOS token: Added by script")
        else:
            print(f"  EOS token: NOT added (--no-add-eos) - datatrove tokenizer may add it")

    print(f"\n{'='*60}")
    print(f"Next step - tokenize with datatrove:")
    print(
        f"python3 /iopsstor/scratch/cscs/$(USER)/data-pipeline-pretrain/examples/tokenize_megatron/preprocess_megatron.py \\"
    )
    print(f"    --tokenizer-name-or-path {args.tokenizer_path} \\")
    print(f"    --output-folder <output_folder> \\")
    print(f"    --n-tasks 16 \\")
    if results and results[0]["bos_added_by_template"]:
        print(f"    --no-add-special-tokens \\  # Chat template already added BOS/EOS")
    print(f"    jsonl \\")
    print(f"    --dataset <output_file> \\")
    print(f"    --column text")


if __name__ == "__main__":
    main()
