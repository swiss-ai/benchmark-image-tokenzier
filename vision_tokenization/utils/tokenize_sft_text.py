#!/usr/bin/env python3
"""
Apply chat template to SFT dataset and prepare for Megatron tokenization.
Supports both parquet and jsonl output formats.

Example usage:
    # Apply chat template to FineVision text dataset (save as parquet)
    python3 tokenize_sft_text.py \
        --dataset HuggingFaceM4/FineVision \
        --config text_OpenMathInstruct-2 \
        --output-file /capstor/store/cscs/swissai/infra01/vision-datasets/text_OpenMathInstruct-2_formatted.parquet \
        --tokenizer-path /capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3_tokenizer \
        --add-eos \
        --num-proc 16

    # Apply chat template to dataset (save as jsonl)
    python3 tokenize_sft_text.py \
        --dataset HuggingFaceH4/ultrachat_200k \
        --output-file ultrachat_formatted.jsonl \
        --tokenizer-path /capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3_tokenizer \
        --add-eos \
        --max-samples 100000

    # Then tokenize with datatrove (works with both parquet and jsonl)
    python3 /iopsstor/scratch/cscs/xyixuan/data-pipeline-pretrain/examples/tokenize_megatron/preprocess_megatron.py \
        --tokenizer-name-or-path /capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3_tokenizer \
        --output-folder datasets/openmathinstruct_tokenized \
        --n-tasks 16 \
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
    parser.add_argument("--output-file", required=True, help="Output JSONL file")
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer for chat template")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--message-column", default="messages", help="Column containing messages")
    parser.add_argument("--add-bos", action="store_true", help="Add BOS token to the beginning")
    parser.add_argument("--add-eos", action="store_true", help="Add EOS token to the end")
    parser.add_argument("--num-proc", type=int, default=None, help="Number of processes for parallel processing")

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    if args.config:
        print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    print(f"Message column: {args.message_column}")

    # Load dataset (download full dataset for faster processing)
    if args.config:
        dataset = load_dataset(
            args.dataset,
            args.config,
            split=args.split,
            num_proc=args.num_proc  # Use parallel processing if specified
        )
    else:
        dataset = load_dataset(
            args.dataset,
            split=args.split,
            num_proc=args.num_proc  # Use parallel processing if specified
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

    # Define processing function
    def apply_chat_template_to_sample(example):
        """Apply chat template to a single sample."""
        try:
            # Get messages from the specified column
            messages = example[args.message_column]

            # Apply chat template
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

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
        apply_chat_template_to_sample,
        num_proc=args.num_proc if args.num_proc else 1,
        desc="Applying chat template"
    )

    # Save to appropriate format based on file extension
    print(f"\nSaving to {args.output_file}...")
    if args.output_file.endswith('.parquet'):
        processed_dataset.to_parquet(args.output_file)
    elif args.output_file.endswith('.jsonl') or args.output_file.endswith('.json'):
        processed_dataset.to_json(args.output_file, orient="records", lines=True)
    else:
        # Default to parquet for preprocess_megatron compatibility
        print("No extension specified, saving as parquet")
        processed_dataset.to_parquet(args.output_file + ".parquet")

    count = len(processed_dataset)
    errors = sum(1 for sample in processed_dataset if sample.get("text", "") == "")

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total samples processed: {count:,}")
    print(f"Errors encountered: {errors:,}")
    print(f"Output saved to: {args.output_file}")

    if args.add_eos and not args.add_bos:
        print(f"\nNote: Only EOS token was added. Since MegatronDocumentTokenizer's encode_batch")
        print(f"adds BOS automatically, this is typically the correct configuration.")

    print(f"\n{'='*60}")
    print(f"Next step - tokenize with datatrove:")
    print(f"python3 /iopsstor/scratch/cscs/xyixuan/data-pipeline-pretrain/examples/tokenize_megatron/preprocess_megatron.py \\")
    print(f"    --tokenizer-name-or-path {args.tokenizer_path} \\")
    print(f"    --output-folder <output_folder> \\")
    print(f"    --n-tasks 16 \\")
    print(f"    jsonl \\")
    print(f"    --dataset {args.output_file} \\")
    print(f"    --column text")


if __name__ == "__main__":
    main()