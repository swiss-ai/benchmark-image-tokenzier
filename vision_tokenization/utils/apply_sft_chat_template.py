#!/usr/bin/env python3
"""
Apply chat template to SFT dataset and prepare for Megatron tokenization.
Supports both parquet and jsonl output formats.

Example usage:
    # Apply chat template to FineVision text dataset (save as parquet)
    python3 apply_sft_chat_template.py \
        --dataset HuggingFaceM4/FineVision \
        --config text_numinamath_cot \
        --output-dir /capstor/store/cscs/swissai/infra01/vision-datasets \
        --tokenizer-path /capstor/store/cscs/swissai/infra01/MLLM/llama3_vision_instruct_emu3_tokenizer \
        --add-eos \
        --num-proc 64

    # Apply chat template to dataset (save as jsonl)
    python3 apply_sft_chat_template.py \
        --dataset HuggingFaceH4/ultrachat_200k \
        --output-dir /capstor/store/cscs/swissai/infra01/vision-datasets \
        --output-format jsonl \
        --tokenizer-path /capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3_tokenizer \
        --add-eos \
        --max-samples 100000

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

from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Apply chat template to SFT dataset")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name or path")
    parser.add_argument("--config", default=None, help="Dataset configuration (e.g., for FineVision)")
    parser.add_argument("--split", default="train", help="Dataset split to use")
    parser.add_argument("--output-dir", required=True, help="Output directory for formatted files")
    parser.add_argument(
        "--output-format", default="parquet", choices=["parquet", "jsonl"], help="Output format (default: parquet)"
    )
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer for chat template")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--message-column", default="texts", help="Column containing messages")
    parser.add_argument("--add-bos", action="store_true", help="Add BOS token to the beginning")
    parser.add_argument("--add-eos", action="store_true", help="Add EOS token to the end")
    parser.add_argument("--num-proc", type=int, default=64, help="Number of processes for parallel processing")

    args = parser.parse_args()

    # Generate output filename from dataset and config
    import os

    os.makedirs(args.output_dir, exist_ok=True)

    # Create filename: config_formatted.{format} or dataset_formatted.{format}
    if args.config:
        filename = f"{args.config}_formatted.{args.output_format}"
    else:
        # Use last part of dataset name (e.g., HuggingFaceH4/ultrachat_200k -> ultrachat_200k)
        dataset_name = args.dataset.split("/")[-1]
        filename = f"{dataset_name}_formatted.{args.output_format}"

    output_file = os.path.join(args.output_dir, filename)

    print(f"Loading dataset: {args.dataset}")
    if args.config:
        print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    print(f"Message column: {args.message_column}")

    # Load dataset (download full dataset for faster processing)
    if args.config:
        dataset = load_dataset(
            args.dataset, args.config, split=args.split, num_proc=args.num_proc  # Use parallel processing if specified
        )
    else:
        dataset = load_dataset(
            args.dataset, split=args.split, num_proc=args.num_proc  # Use parallel processing if specified
        )

    print(f"Loading tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Get BOS/EOS tokens from tokenizer
    bos_token = tokenizer.bos_token if args.add_bos else None
    eos_token = tokenizer.eos_token if args.add_eos else None

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

    # Select subset if max_samples is specified
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"Processing {len(dataset)} samples")

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
                {"role": role, "content": conv[role]}
                for conv in messages
                for role in ("user", "assistant")
                if role in conv
            ]

        return messages  # Unknown format, return as-is

    # Define processing function
    def apply_chat_template_to_sample(example):
        """Apply chat template to a single sample."""
        try:
            # Get messages from the specified column and normalize format
            messages = normalize_messages(example[args.message_column])

            # Apply chat template
            # LLAMA3.2 Vision Instruct Tokenizer hardcode the BOS token
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            # Add BOS/EOS if requested
            if args.add_bos and bos_token and not formatted_text.startswith(bos_token):
                formatted_text = bos_token + formatted_text
            if args.add_eos and eos_token and not formatted_text.endswith(eos_token):
                formatted_text = formatted_text + eos_token

            # Add new 'text' field
            example["text"] = formatted_text
            return example
        except Exception as e:
            print(f"Error processing sample: {e}")
            example["text"] = ""  # Add empty text for failed samples
            return example

    # Process dataset in parallel using map
    print(f"\nApplying chat template to dataset...")
    processed_dataset = dataset.map(
        apply_chat_template_to_sample, num_proc=args.num_proc if args.num_proc else 1, desc="Applying chat template"
    )

    # Save to appropriate format
    print(f"\nSaving to {output_file}...")
    if args.output_format == "parquet":
        processed_dataset.to_parquet(output_file)
    elif args.output_format == "jsonl":
        processed_dataset.to_json(output_file, orient="records", lines=True)

    count = len(processed_dataset)
    errors = sum(1 for sample in processed_dataset if sample.get("text", "") == "")

    # Check if chat template added BOS token (verify first few samples)
    sample_texts = [processed_dataset[i]["text"] for i in range(min(10, count)) if processed_dataset[i].get("text")]
    # Always check against tokenizer.bos_token (actual string), not bos_token variable (which may be None)
    bos_added_by_template = (
        all(text.startswith(tokenizer.bos_token) for text in sample_texts) if tokenizer.bos_token else False
    )

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total samples processed: {count:,}")
    print(f"Errors encountered: {errors:,}")
    print(f"Output saved to: {output_file}")

    # Verification and warnings
    print(f"\nToken verification:")
    if bos_added_by_template:
        print(f"✅ BOS token: Added by chat template (hardcoded in template)")
    else:
        print(f"⚠️  BOS token: NOT added by chat template")
        if not args.add_bos:
            print(f"   Warning: You may need --add-bos flag")

    if args.add_eos:
        print(f"✅ EOS token: Added by script (--add-eos)")
    else:
        print(f"⚠️  EOS token: NOT added - datatrove tokenizer may add it")

    print(f"\n{'='*60}")
    print(f"Next step - tokenize with datatrove:")
    print(
        f"python3 /iopsstor/scratch/cscs/xyixuan/data-pipeline-pretrain/examples/tokenize_megatron/preprocess_megatron.py \\"
    )
    print(f"    --tokenizer-name-or-path {args.tokenizer_path} \\")
    print(f"    --output-folder <output_folder> \\")
    print(f"    --n-tasks 16 \\")
    if bos_added_by_template:
        print(f"    --no-add-special-tokens \\  # Chat template already added BOS/EOS")
    print(f"    jsonl \\")
    print(f"    --dataset {output_file} \\")
    print(f"    --column text")


if __name__ == "__main__":
    main()
