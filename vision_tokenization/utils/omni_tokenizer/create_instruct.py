#!/usr/bin/env python3
"""
Create Instruct Omni-Tokenizer

Creates an instruct omnimodal tokenizer by adding vision tokens to an instruct text tokenizer.
The instruct text tokenizer already contains the chat template, so this script:
1. Adds RESERVED_OMNI tokens + vision tokens (same as create_base.py)
2. Adds pre-tokenized SFT sequences for Megatron-LM

Usage:
    # For Emu3
    python create_instruct.py \
        --text-tokenizer-path meta-llama/Llama-3.2-3B-Instruct \
        --vision-tokenizer-path BAAI/Emu3-VisionTokenizer \
        --vision-tokenizer Emu3 \
        --output-path ./llama3_emu3_omni_instruct

    # For Emu3.5
    python create_instruct.py \
        --text-tokenizer-path meta-llama/Llama-3.2-3B-Instruct \
        --vision-tokenizer-path /capstor/store/cscs/swissai/infra01/MLLM/Emu3.5-VisionTokenizer \
        --vision-tokenizer Emu3.5 \
        --output-path ./llama3_emu3.5_omni_instruct
"""

import argparse
import json
import os

from .core import create_base_tokenizer


def create_instruct_tokenizer(
    text_tokenizer_path: str, vision_tokenizer_path: str, vision_tokenizer: str, output_path: str
):
    """
    Create an instruct omni-tokenizer from an instruct text tokenizer.

    This function:
    1. Calls create_base_tokenizer to add RESERVED_OMNI + vision tokens
    2. Adds pre-tokenized SFT sequences to tokenizer config for Megatron-LM
    3. Preserves the chat template from the instruct text tokenizer

    Args:
        text_tokenizer_path: Path to instruct text tokenizer (e.g., meta-llama/Llama-3.2-3B-Instruct)
        vision_tokenizer_path: Path to vision tokenizer model
        vision_tokenizer: Vision tokenizer name (e.g., "Emu3", "Emu3.5")
        output_path: Path to save instruct omni-tokenizer

    Returns:
        Tuple of (tokenizer object, stats dict)
    """

    print("=" * 60)
    print("CREATING OMNI-TOKENIZER (INSTRUCT)")
    print("=" * 60)

    # Step 1: Create base tokenizer (adds RESERVED_OMNI + vision tokens)
    print("\nStep 1: Adding vision tokens to instruct text tokenizer...")
    print("=" * 60)

    tokenizer, base_stats = create_base_tokenizer(
        text_tokenizer_path=text_tokenizer_path,
        vision_tokenizer_path=vision_tokenizer_path,
        vision_tokenizer=vision_tokenizer,
        output_path=output_path,
    )

    # Update stats for instruct type
    stats = base_stats.copy()
    stats["tokenizer_type"] = "instruct"
    stats["sft_sequences_added"] = False

    # Step 2: Add SFT sequences for Megatron-LM
    print("\n" + "=" * 60)
    print("Step 2: Adding SFT sequences for Megatron-LM")
    print("=" * 60)

    config_path = os.path.join(output_path, "tokenizer_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Auto-detect chat template format from the tokenizer
    chat_template = config.get("chat_template", "")

    # Detect LLaMA-3 vs Mistral style
    if "<|start_header_id|>" in chat_template:
        # LLaMA-3 style
        user_header = "<|start_header_id|>user<|end_header_id|>"
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>"
        eot_token = "<|eot_id|>"
        print("  ✓ Detected LLaMA-3 style chat template")
    elif "[INST]" in chat_template:
        # Mistral/Mixtral style
        user_header = "[INST]"
        assistant_header = "[/INST]"
        eot_token = "</s>"
        print("  ✓ Detected Mistral/Mixtral style chat template")
    else:
        # Raise error for unsupported formats
        raise ValueError(
            f"Unsupported chat template format. Only LLaMA-3 and Mistral/Mixtral styles are supported.\n"
            f"Expected '<|start_header_id|>' (LLaMA-3) or '[INST]' (Mistral) in chat template."
        )

    # Add pre-tokenized SFT sequences
    config["sft_user_begin_sequence"] = tokenizer.encode(user_header, add_special_tokens=False)
    config["sft_assistant_begin_sequence"] = tokenizer.encode(assistant_header, add_special_tokens=False)
    config["sft_eot_token"] = tokenizer.encode(eot_token, add_special_tokens=False)
    config["img_begin_token"] = tokenizer.encode("<|img_start|>", add_special_tokens=False)
    config["img_end_token"] = tokenizer.encode("<|img_end|>", add_special_tokens=False)

    print(f"  ✓ sft_user_begin_sequence: {config['sft_user_begin_sequence']} ({user_header})")
    print(f"  ✓ sft_assistant_begin_sequence: {config['sft_assistant_begin_sequence']} ({assistant_header})")
    print(f"  ✓ sft_eot_token: {config['sft_eot_token']} ({eot_token})")
    print(f"  ✓ img_begin_token: {config['img_begin_token']}")
    print(f"  ✓ img_end_token: {config['img_end_token']}")
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

    # Verify chat template exists
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        print("  ✓ Chat template preserved from instruct text tokenizer")
    else:
        print("  ⚠️  No chat template found")

    # Check chat template tokens
    print("\nChat template tokens:")
    for token in ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]:
        if token in tokenizer.get_vocab():
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"  {token}: ID {token_id}")

    # Check image token
    print("\nImage structure tokens:")
    image_token_id = tokenizer.convert_tokens_to_ids("<|image|>")
    if image_token_id != tokenizer.unk_token_id:
        print(f"  <|image|>: ID {image_token_id}")

    img_start_id = tokenizer.convert_tokens_to_ids("<|img_start|>")
    if img_start_id != tokenizer.unk_token_id:
        print(f"  <|img_start|>: ID {img_start_id}")

    img_end_id = tokenizer.convert_tokens_to_ids("<|img_end|>")
    if img_end_id != tokenizer.unk_token_id:
        print(f"  <|img_end|>: ID {img_end_id}")

    print("\n" + "=" * 60)

    return tokenizer, stats


def main():
    parser = argparse.ArgumentParser(
        description="Create instruct omni-tokenizer from instruct text tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create instruct tokenizer with Emu3
  python create_instruct.py \\
      --text-tokenizer-path meta-llama/Llama-3.2-3B-Instruct \\
      --vision-tokenizer-path BAAI/Emu3-VisionTokenizer \\
      --vision-tokenizer Emu3 \\
      --output-path ./llama3_emu3_omni_instruct

  # Create instruct tokenizer with Emu3.5
  python create_instruct.py \\
      --text-tokenizer-path meta-llama/Llama-3.2-3B-Instruct \\
      --vision-tokenizer-path /path/to/Emu3.5-VisionTokenizer \\
      --vision-tokenizer Emu3.5 \\
      --output-path ./llama3_emu3.5_omni_instruct

  # Use Mixtral instruct tokenizer
  python create_instruct.py \\
      --text-tokenizer-path mistralai/Mixtral-8x7B-Instruct-v0.1 \\
      --vision-tokenizer-path BAAI/Emu3-VisionTokenizer \\
      --vision-tokenizer Emu3 \\
      --output-path ./mixtral_emu3_omni_instruct
        """,
    )

    parser.add_argument(
        "--text-tokenizer-path",
        type=str,
        required=True,
        help="Path to instruct text tokenizer (e.g., meta-llama/Llama-3.2-3B-Instruct)",
    )
    parser.add_argument("--vision-tokenizer-path", type=str, required=True, help="Path to vision tokenizer model")
    parser.add_argument(
        "--vision-tokenizer",
        type=str,
        required=True,
        choices=["Emu3", "Emu3.5"],
        help="Vision tokenizer type (Emu3 or Emu3.5)",
    )
    parser.add_argument("--output-path", type=str, required=True, help="Path to save instruct omni-tokenizer")

    args = parser.parse_args()

    # Create instruct omni-tokenizer
    tokenizer, stats = create_instruct_tokenizer(
        text_tokenizer_path=args.text_tokenizer_path,
        vision_tokenizer_path=args.vision_tokenizer_path,
        vision_tokenizer=args.vision_tokenizer,
        output_path=args.output_path,
    )

    print("\n" + "=" * 60)
    print("INSTRUCT OMNI-TOKENIZER CREATION SUMMARY")
    print("=" * 60)
    print(f"Text tokenizer:           {stats['text_tokenizer']}")
    print(f"Vision tokenizer:         {stats['vision_tokenizer']}")
    print(f"Tokenizer type:           {stats['tokenizer_type']}")
    print(f"Final vocabulary size:    {stats['final_vocab_size']:,}")
    print(f"RESERVED_OMNI added:      {stats['reserved_omni_added']}")
    print(f"Vision tokens added:      {stats['vision_tokens_added']:,}")
    print(f"SFT sequences added:      {'✓' if stats['sft_sequences_added'] else '✗'}")
    print("=" * 60)
    print("\n✅ Instruct omni-tokenizer created successfully!")
    print(f"   Saved to: {args.output_path}")


if __name__ == "__main__":
    main()
