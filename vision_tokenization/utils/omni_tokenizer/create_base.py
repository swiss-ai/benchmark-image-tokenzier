#!/usr/bin/env python3
"""
Create Base Omni-Tokenizer

Creates a base omnimodal tokenizer by adding vision tokens to a text tokenizer.
Auto-detects codebook size from the vision tokenizer.

Usage:
    # For Llama3 + Emu3
    python create_base.py \\
        --text-tokenizer-path meta-llama/Llama-3-8B \\
        --vision-tokenizer-path BAAI/Emu3-VisionTokenizer \\
        --vision-tokenizer Emu3 \\
        --output-path /capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3_tokenizer

    # For Llama3 + Emu3.5
    python create_base.py \\
        --text-tokenizer-path meta-llama/Llama-3-8B \\
        --vision-tokenizer-path /capstor/store/cscs/swissai/infra01/MLLM/Emu3.5-VisionTokenizer \\
        --vision-tokenizer Emu3.5 \\
        --output-path /capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3.5_tokenizer
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .core import create_base_tokenizer
except ImportError:
    # Fallback for direct execution
    from core import create_base_tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Create base omni-tokenizer with auto-detected codebook size",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Create with Emu3 (32K visual tokens - auto-detected)
        python create_base.py \\
            --text-tokenizer-path meta-llama/Llama-3-8B \\
            --vision-tokenizer-path BAAI/Emu3-VisionTokenizer \\
            --vision-tokenizer Emu3 \\
            --output-path ./llama3_emu3_omni

        # Create with Emu3.5 (131K visual tokens - auto-detected)
        python create_base.py \\
            --text-tokenizer-path meta-llama/Llama-3-8B \\
            --vision-tokenizer-path /path/to/Emu3.5-VisionTokenizer \\
            --vision-tokenizer Emu3.5 \\
            --output-path ./llama3_emu3.5_omni

        Available vision tokenizers:
        - Emu3 (32K codebook)
        - Emu3.5 (131K codebook)
        """
    )

    parser.add_argument(
        "--text-tokenizer-path",
        type=str,
        required=True,
        help="Path to base text tokenizer (e.g., meta-llama/Llama-3-8B)"
    )
    parser.add_argument(
        "--vision-tokenizer-path",
        type=str,
        required=True,
        help="Path to vision tokenizer model"
    )
    parser.add_argument(
        "--vision-tokenizer",
        type=str,
        required=True,
        choices=["Emu3", "Emu3.5"],
        help="Vision tokenizer type (Emu3 or Emu3.5)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save omni-tokenizer"
    )
    parser.add_argument(
        "--num-reserved-tokens",
        type=int,
        default=200,
        help="Number of RESERVED_OMNI tokens to add (default: 200)"
    )

    args = parser.parse_args()

    # Create omni-tokenizer
    tokenizer, stats = create_base_tokenizer(
        text_tokenizer_path=args.text_tokenizer_path,
        output_path=args.output_path,
        vision_tokenizer_path=args.vision_tokenizer_path,
        vision_tokenizer=args.vision_tokenizer,
        num_reserved_tokens=args.num_reserved_tokens
    )

    print("\n" + "="*60)
    print("OMNI-TOKENIZER CREATION SUMMARY")
    print("="*60)
    print(f"Text tokenizer:           {stats['text_tokenizer']}")
    print(f"Vision tokenizer:         {stats['vision_tokenizer']}")
    print(f"Tokenizer type:           {stats['tokenizer_type']}")
    print(f"Original vocabulary size: {stats['original_vocab_size']:,}")
    print(f"Reserved tokens added:    {stats['reserved_tokens_added']:,} (includes {stats['structure_tokens_added']} for image structure)")
    print(f"Visual tokens added:      {stats['visual_tokens_added']:,}")
    print(f"Final vocabulary size:    {stats['final_vocab_size']:,}")
    print(f"Total tokens added:       {stats['final_vocab_size'] - stats['original_vocab_size']:,}")
    print("="*60)
    print("\n✅ Base omni-tokenizer created successfully!")
    print(f"   Saved to: {args.output_path}")


if __name__ == "__main__":
    main()
