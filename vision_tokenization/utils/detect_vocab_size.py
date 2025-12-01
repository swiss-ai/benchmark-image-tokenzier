#!/usr/bin/env python3
"""
Utility to detect vocabulary sizes from tokenizers.
This helps determine the correct offset for multimodal tokenization.
"""

import json
import os
import sys

from transformers import AutoTokenizer


def detect_text_vocab_size(tokenizer_path: str) -> dict:
    """
    Detect vocabulary size from a text tokenizer.

    Args:
        tokenizer_path: Path or HuggingFace model name for the tokenizer

    Returns:
        Dictionary with vocabulary information
    """
    try:
        # Load tokenizer
        print(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Get vocabulary size
        vocab_size = len(tokenizer)

        # Get some sample tokens
        sample_tokens = {}
        if hasattr(tokenizer, "get_vocab"):
            vocab = tokenizer.get_vocab()
            # Get first few and last few tokens
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            sample_tokens = {
                "first_10": sorted_vocab[:10],
                "last_10": sorted_vocab[-10:],
            }

        # Check for special tokens
        special_tokens = {}
        for attr in ["pad_token", "eos_token", "bos_token", "unk_token"]:
            if hasattr(tokenizer, attr):
                token = getattr(tokenizer, attr)
                if token is not None:
                    special_tokens[attr] = str(token)

        result = {
            "tokenizer_path": tokenizer_path,
            "vocab_size": vocab_size,
            "tokenizer_type": tokenizer.__class__.__name__,
            "special_tokens": special_tokens,
            "sample_tokens": sample_tokens,
        }

        return result

    except Exception as e:
        return {"tokenizer_path": tokenizer_path, "error": str(e), "vocab_size": None}


def main():
    """Main function to detect vocabulary sizes."""
    if len(sys.argv) < 2:
        print("Usage: python detect_vocab_size.py <tokenizer_path>")
        print("Example: python detect_vocab_size.py alehc/swissai-tokenizer")
        sys.exit(1)

    tokenizer_path = sys.argv[1]

    print("=" * 80)
    print("Vocabulary Size Detection")
    print("=" * 80)

    # Detect text vocabulary
    result = detect_text_vocab_size(tokenizer_path)

    if result.get("error"):
        print(f"❌ Error loading tokenizer: {result['error']}")
        sys.exit(1)

    print(f"✅ Successfully loaded tokenizer")
    print(f"📊 Vocabulary Information:")
    print(f"   - Path: {result['tokenizer_path']}")
    print(f"   - Type: {result['tokenizer_type']}")
    print(f"   - Vocabulary Size: {result['vocab_size']:,}")

    if result["special_tokens"]:
        print(f"🔧 Special Tokens:")
        for name, token in result["special_tokens"].items():
            print(f"   - {name}: '{token}'")

    if result["sample_tokens"]:
        print(f"📝 Sample Tokens:")
        print(f"   - First 10: {result['sample_tokens']['first_10']}")
        print(f"   - Last 10: {result['sample_tokens']['last_10']}")

    print("\n" + "=" * 80)
    print("Configuration for Multimodal Tokenization:")
    print("=" * 80)

    text_vocab_size = result["vocab_size"]
    image_vocab_size = 2**17  # Emu3 default
    total_vocab_size = text_vocab_size + image_vocab_size

    print(f"For your multimodal setup:")
    print(f"--text-vocab-size {text_vocab_size} \\")
    print(f"--image-vocab-size {image_vocab_size} \\")
    print(f"# Total vocabulary: {total_vocab_size:,} tokens")
    print(f"# Image tokens will be offset by {text_vocab_size:,}")
    print(f"# Image token range: [{text_vocab_size:,}, {total_vocab_size-1:,}]")

    # Save results to file
    output_file = f"vocab_info_{tokenizer_path.replace('/', '_')}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n📄 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
