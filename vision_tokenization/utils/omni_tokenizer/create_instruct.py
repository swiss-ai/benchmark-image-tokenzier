#!/usr/bin/env python3
"""
Create Instruct Omni-Tokenizer

Adds chat template and SFT sequences to a base omni-tokenizer (created by create_base.py).

Usage:
    # Step 1: Create base tokenizer first (see create_base.py)

    # Step 2: Add chat template + SFT sequences
    python create_instruct.py \
        --base-tokenizer-path /path/to/omni_base \
        --instruct-tokenizer-path meta-llama/Llama-3.2-3B-Instruct \
        --output-path /path/to/omni_instruct
"""

import argparse
import json
import os
import shutil

from transformers import AutoTokenizer


def create_instruct_tokenizer(base_tokenizer_path: str, instruct_tokenizer_path: str, output_path: str):
    """
    Add chat template and SFT sequences to a base omni-tokenizer.

    This function:
    1. Loads the base omni-tokenizer (which already has vision tokens)
    2. Copies chat template from instruct text tokenizer
    3. Adds pre-tokenized SFT sequences to tokenizer config for Megatron-LM

    Args:
        base_tokenizer_path: Path to base omni-tokenizer (created by create_base.py)
        instruct_tokenizer_path: Path to instruct text tokenizer (for chat template)
        output_path: Path to save instruct omni-tokenizer

    Returns:
        Tuple of (tokenizer object, stats dict)
    """

    print("=" * 60)
    print("CREATING OMNI-TOKENIZER (INSTRUCT)")
    print("=" * 60)

    # Step 1: Load base omni-tokenizer
    print(f"\nStep 1: Loading base omni-tokenizer from {base_tokenizer_path}...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path)
    print(f"  ✓ Loaded tokenizer with vocab size: {len(tokenizer):,}")

    # Verify it's an omni-tokenizer (has vision tokens)
    if "<|img_start|>" not in tokenizer.get_vocab():
        raise ValueError(
            f"Base tokenizer at {base_tokenizer_path} does not have <|img_start|> token. "
            f"Please create it first using create_base.py"
        )
    print("  ✓ Verified: base tokenizer has vision tokens")

    # Step 2: Load instruct tokenizer for chat template
    print(f"\nStep 2: Loading instruct tokenizer from {instruct_tokenizer_path}...")
    print("=" * 60)

    instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_tokenizer_path)
    chat_template = instruct_tokenizer.chat_template
    if not chat_template:
        raise ValueError(f"Instruct tokenizer at {instruct_tokenizer_path} does not have a chat template.")
    print("  ✓ Loaded chat template from instruct tokenizer")

    # Copy base tokenizer to output path if different
    if os.path.abspath(base_tokenizer_path) != os.path.abspath(output_path):
        print(f"\n  Copying base tokenizer to {output_path}...")
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        shutil.copytree(base_tokenizer_path, output_path)
        print("  ✓ Copied base tokenizer")

    # Load config
    config_path = os.path.join(output_path, "tokenizer_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Add chat template to config
    config["chat_template"] = chat_template

    # Build stats
    stats = {
        "base_tokenizer": base_tokenizer_path,
        "instruct_tokenizer": instruct_tokenizer_path,
        "tokenizer_type": "instruct",
        "vocab_size": config.get("vocab_size", len(tokenizer)),
        "sft_sequences_added": False,
    }

    # Step 3: Add SFT sequences for Megatron-LM
    print("\n" + "=" * 60)
    print("Step 3: Adding SFT sequences for Megatron-LM")
    print("=" * 60)

    # Detect chat template style
    if "<|start_header_id|>" in chat_template:
        # LLaMA-3 style
        user_header = "<|start_header_id|>user<|end_header_id|>"
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>"
        eot_token = "<|eot_id|>"
        print("  ✓ Detected LLaMA-3 style chat template")
    elif "<|user_start|>" in chat_template:
        # Apertus style (ChatML-like)
        user_header = "<|user_start|>"
        assistant_header = "<|assistant_start|>"
        eot_token = "<|assistant_end|>"
        print("  ✓ Detected Apertus style chat template")
    else:
        raise ValueError(
            f"Unsupported chat template format. Supported styles are:\n"
            f"- LLaMA-3 (with '<|start_header_id|>')\n"
            f"- Apertus (with '<|user_start|>')\n"
            f"Found chat template does not match any known format."
        )

    # Add pre-tokenized SFT sequences (using base omni-tokenizer which has correct single-token IDs)
    config["sft_user_begin_sequence"] = tokenizer.encode(user_header, add_special_tokens=False)
    config["sft_assistant_begin_sequence"] = tokenizer.encode(assistant_header, add_special_tokens=False)
    config["sft_eot_token"] = tokenizer.encode(eot_token, add_special_tokens=False)
    config["img_begin_token"] = tokenizer.encode("<|img_start|>", add_special_tokens=False)
    config["img_end_token"] = tokenizer.encode("<|img_end|>", add_special_tokens=False)

    print(f"  ✓ sft_user_begin_sequence: {config['sft_user_begin_sequence']} ({user_header})")
    print(f"  ✓ sft_assistant_begin_sequence: {config['sft_assistant_begin_sequence']} ({assistant_header})")
    print(f"  ✓ sft_eot_token: {config['sft_eot_token']} ({eot_token})")
    print(f"  ✓ img_begin_token: {config['img_begin_token']} (<|img_start|>)")
    print(f"  ✓ img_end_token: {config['img_end_token']} (<|img_end|>)")
    stats["sft_sequences_added"] = True

    # Save updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Update vision_token_mapping.json with instruct type
    mapping_path = os.path.join(output_path, "vision_token_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            mapping = json.load(f)

        mapping["tokenizer_type"] = "instruct"

        with open(mapping_path, "w") as f:
            json.dump(mapping, f, indent=2)

    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION - Chat Template & SFT Tokens")
    print("=" * 60)

    print("  ✓ Chat template added from instruct tokenizer")

    # Check chat template tokens
    print("\nChat template tokens:")
    for token in ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]:
        if token in tokenizer.get_vocab():
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"  {token}: ID {token_id}")

    # Check image tokens
    print("\nImage structure tokens:")
    for token in ["<|image|>", "<|img_start|>", "<|img_end|>"]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            print(f"  {token}: ID {token_id}")

    print("\n" + "=" * 60)

    return tokenizer, stats


def main():
    parser = argparse.ArgumentParser(
        description="Add chat template and SFT sequences to a base omni-tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_instruct.py \\
      --base-tokenizer-path /path/to/llama3_emu3_base \\
      --instruct-tokenizer-path meta-llama/Llama-3.2-3B-Instruct \\
      --output-path /path/to/llama3_emu3_instruct
        """,
    )

    parser.add_argument(
        "--base-tokenizer-path", type=str, required=True, help="Path to base omni-tokenizer (created by create_base.py)"
    )
    parser.add_argument(
        "--instruct-tokenizer-path",
        type=str,
        required=True,
        help="Path or HuggingFace model ID for instruct text tokenizer (for chat template)",
    )
    parser.add_argument("--output-path", type=str, required=True, help="Path to save instruct omni-tokenizer")

    args = parser.parse_args()

    # Create instruct omni-tokenizer
    tokenizer, stats = create_instruct_tokenizer(
        base_tokenizer_path=args.base_tokenizer_path,
        instruct_tokenizer_path=args.instruct_tokenizer_path,
        output_path=args.output_path,
    )

    print("\n" + "=" * 60)
    print("INSTRUCT OMNI-TOKENIZER CREATION SUMMARY")
    print("=" * 60)
    print(f"Base tokenizer:      {stats['base_tokenizer']}")
    print(f"Instruct tokenizer:  {stats['instruct_tokenizer']}")
    print(f"Tokenizer type:      {stats['tokenizer_type']}")
    print(f"Vocabulary size:     {stats['vocab_size']:,}")
    print(f"SFT sequences added: {'✓' if stats['sft_sequences_added'] else '✗'}")
    print("=" * 60)
    print("\n✅ Instruct omni-tokenizer created successfully!")
    print(f"   Saved to: {args.output_path}")


if __name__ == "__main__":
    main()
