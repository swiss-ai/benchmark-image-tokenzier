#!/usr/bin/env python3
"""
Example of how to use the EMU3-style tokenizer after modification.
"""

from transformers import AutoTokenizer
from add_special_tokens_emu3_style import add_emu3_special_tokens
import os

def get_or_create_emu3_tokenizer(
    base_model_path="/iopsstor/scratch/cscs/xyixuan/Llama-3.1-8B",
    tokenizer_save_path="/iopsstor/scratch/cscs/xyixuan/llama3_emu3_tokenizer",
    visual_vocab_size=32768
):
    """
    Get EMU3-style tokenizer, creating it if it doesn't exist.
    
    Args:
        base_model_path: Path to base model/tokenizer
        tokenizer_save_path: Where to save/load the modified tokenizer
        visual_vocab_size: Number of visual tokens
    
    Returns:
        The modified tokenizer object
    """
    
    # Check if modified tokenizer already exists
    if os.path.exists(tokenizer_save_path) and os.path.exists(os.path.join(tokenizer_save_path, "tokenizer_config.json")):
        print(f"Loading existing EMU3 tokenizer from {tokenizer_save_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
    else:
        print(f"Creating new EMU3 tokenizer based on {base_model_path}")
        # Create the modified tokenizer
        stats, tokenizer = add_emu3_special_tokens(
            model_path=base_model_path,
            output_path=tokenizer_save_path,
            visual_vocab_size=visual_vocab_size,
            return_tokenizer=True
        )
        print(f"Created tokenizer with {stats['final_vocab_size']} tokens")
    
    return tokenizer


def encode_image_sequence(tokenizer, image_token_ids, h, w):
    """
    Encode an image into EMU3 format.
    
    Args:
        tokenizer: The EMU3-style tokenizer
        image_token_ids: 2D array of token IDs from vision tokenizer
        h: Height in tokens
        w: Width in tokens
    
    Returns:
        List of token IDs for the complete sequence
    """
    
    # Build the sequence string
    sequence_parts = []
    
    # Structure tokens
    sequence_parts.append("<|img_start|>")
    sequence_parts.append(f"{h}*{w}")
    sequence_parts.append("<|img_token_start|>")
    
    # Add visual tokens with EOL between rows
    for row_idx, row in enumerate(image_token_ids):
        for token_id in row:
            sequence_parts.append(f"<|visual token {token_id:06d}|>")
        if row_idx < h - 1:  # Add EOL between rows
            sequence_parts.append("<|img_end_of_row|>")
    
    # End tokens
    sequence_parts.append("<|img_end_of_row|>")
    sequence_parts.append("<|img_end_of_frame|>")
    sequence_parts.append("<|img_end|>")
    
    # Convert to string and tokenize
    sequence_str = "".join(sequence_parts)
    token_ids = tokenizer.encode(sequence_str, add_special_tokens=False)
    
    return token_ids


def decode_image_sequence(tokenizer, token_ids):
    """
    Decode token IDs back to readable format.
    
    Args:
        tokenizer: The EMU3-style tokenizer
        token_ids: List of token IDs
    
    Returns:
        Decoded string
    """
    return tokenizer.decode(token_ids, skip_special_tokens=False)


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("EMU3 Tokenizer Usage Example")
    print("="*60)
    
    # Get or create the tokenizer
    tokenizer = get_or_create_emu3_tokenizer()
    
    print(f"\nTokenizer vocabulary size: {len(tokenizer)}")
    
    # Check that structure tokens exist
    print("\nStructure tokens:")
    for token in ["<|img_start|>", "<|img_end|>", "<|img_token_start|>", "<|img_end_of_row|>", "<|img_end_of_frame|>"]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {token}: ID {token_id}")
    
    # Example: Encode a 2x2 image
    print("\n" + "="*60)
    print("Example: Encoding a 2x2 image")
    print("="*60)
    
    # Simulate vision tokenizer output (would come from your image tokenizer)
    fake_image_tokens = [
        [1234, 5678],  # Row 1
        [9012, 3456]   # Row 2
    ]
    
    # Encode the image
    encoded_ids = encode_image_sequence(tokenizer, fake_image_tokens, h=2, w=2)
    print(f"\nEncoded to {len(encoded_ids)} tokens")
    print(f"First 20 token IDs: {encoded_ids[:20]}...")
    
    # Decode back
    decoded = decode_image_sequence(tokenizer, encoded_ids)
    print(f"\nDecoded (first 200 chars):")
    print(decoded[:200] + "..." if len(decoded) > 200 else decoded)
    
    print("\n" + "="*60)
    print("Tokenizer is ready for use in your pipeline!")
    print("="*60)