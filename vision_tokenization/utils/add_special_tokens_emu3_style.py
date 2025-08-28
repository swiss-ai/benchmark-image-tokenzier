#!/usr/bin/env python3
"""
EMU3-style Special Token Adder
Adds EMU3-compatible special tokens to any Hugging Face tokenizer.
Part of the vision tokenization pipeline for multimodal models.

Workflow:
1. Add special structure tokens (<|img_start|>, <|img_end|>, etc.)
2. Add generation mode token (<|GEN:IMAGE:MODE:HASH|>)
3. Add visual tokens (<|visual token 000000|> through <|visual token XXXXXX|>)
4. Save tokenizer with vision token mapping for consistent index conversion

Usage:
    # Step 1: Modify tokenizer
    python add_special_tokens_emu3_style.py --model-path meta-llama/Llama-3-8B \\
        --output-path ./llama3_emu3 --visual-vocab-size 32768
    
    # Step 2: Use in code
    from transformers import AutoTokenizer
    from add_special_tokens_emu3_style import get_vision_token_id
    
    tokenizer = AutoTokenizer.from_pretrained("./llama3_emu3")
    
    # Convert vision indices (from vision tokenizer) to token IDs
    vision_indices = [0, 100, 1000]  # Output from vision tokenizer
    token_ids = [get_vision_token_id(idx, "./llama3_emu3") for idx in vision_indices]

Important: The vision tokenizer outputs indices 0-32767, which must be converted
to the actual token IDs in the vocabulary using the saved mapping.

Generation Mode Token: A special token <|GEN:IMAGE:MODE:HASH|> is added for controlling
image generation during training. This token should be hidden/removed at deployment.
"""

from transformers import AutoTokenizer, PreTrainedTokenizerFast
import json
import os
import shutil
from typing import Optional
try:
    # Import generation mode utils if available (for training)
    from hidden_mode_token_registry import HiddenModeTokenRegistry
    HAS_GENERATION_MODE = True
except ImportError:
    # Not available (for public deployment)
    HAS_GENERATION_MODE = False


class EMU3Tokenizer(PreTrainedTokenizerFast):
    """
    EMU3 tokenizer that returns the correct vocab_size including all special tokens.
    
    The default HuggingFace tokenizer.vocab_size property returns only the base 
    vocabulary size (128000 for Llama), excluding added special tokens. This can
    cause issues when initializing models that need the full vocabulary size.
    
    This wrapper overrides vocab_size to return the full size including EMU3 tokens.
    """
    
    @property
    def vocab_size(self) -> int:
        """
        Returns the full vocabulary size including all EMU3 special tokens.
        
        Returns:
            int: Total vocabulary size (base + added tokens)
        """
        # Return the full vocabulary size including added tokens
        return len(self.get_vocab())
    
    @property
    def base_vocab_size(self) -> int:
        """
        Returns the base vocabulary size without added tokens.
        
        Returns:
            int: Base vocabulary size (original tokenizer vocab)
        """
        # This calls the original vocab_size implementation
        return super().vocab_size
    
    @property
    def added_tokens_count(self) -> int:
        """
        Returns the number of added special tokens.
        
        Returns:
            int: Number of tokens added to base vocabulary
        """
        return self.vocab_size - self.base_vocab_size

# ============================================================================
# Helper Functions
# ============================================================================

def save_tokenizer_with_vocab_fix(tokenizer, save_path: str) -> None:
    """
    Save tokenizer and configure it to use EMU3Tokenizer class for correct vocab_size.
    
    Args:
        tokenizer: The tokenizer instance with potentially updated vocabulary
        save_path: Path where tokenizer will be saved
    """
    
    # First save the tokenizer
    tokenizer.save_pretrained(save_path)
    
    actual_vocab_size = len(tokenizer)
    base_vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else actual_vocab_size
    
    # Update tokenizer config to use our custom class
    config_path = os.path.join(save_path, "tokenizer_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Store vocabulary information
    config['vocab_size'] = actual_vocab_size
    config['base_vocab_size'] = base_vocab_size
    config['added_tokens_count'] = actual_vocab_size - base_vocab_size
    
    # Configure to use EMU3Tokenizer class for correct vocab_size
    config['tokenizer_class'] = 'EMU3Tokenizer'
    config['auto_map'] = {
        'AutoTokenizer': ['add_special_tokens_emu3_style.EMU3Tokenizer', None]
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Copy this module to tokenizer directory for auto-loading
    current_file = os.path.abspath(__file__)
    target_file = os.path.join(save_path, 'add_special_tokens_emu3_style.py')
    shutil.copy(current_file, target_file)
    
    print(f"✓ Saved tokenizer with EMU3 configuration")
    print(f"  Total vocabulary: {actual_vocab_size:,}")
    print(f"  Base vocabulary: {base_vocab_size:,}")
    print(f"  Added tokens: {actual_vocab_size - base_vocab_size:,}")



def deduplicate_tokens(tokens_to_add, existing_vocab, verbose=True):
    """
    Remove duplicate tokens and filter out existing tokens from vocabulary.
    
    Args:
        tokens_to_add: List of tokens to potentially add
        existing_vocab: Dictionary of existing vocabulary
        verbose: Whether to print progress information
    
    Returns:
        List of unique tokens not in existing vocabulary
    """
    unique_new_tokens = []
    seen = set()
    
    for token in tokens_to_add:
        if token not in existing_vocab and token not in seen:
            unique_new_tokens.append(token)
            seen.add(token)
    
    if verbose and len(tokens_to_add) != len(unique_new_tokens):
        filtered_count = len(tokens_to_add) - len(unique_new_tokens)
        print(f"Filtered out {filtered_count} duplicate/existing tokens")
    
    return unique_new_tokens



def add_tokens_with_feedback(tokenizer, tokens, token_type="special"):
    """
    Add tokens to tokenizer with user feedback.
    
    Args:
        tokenizer: The tokenizer to add tokens to
        tokens: List of tokens to add
        token_type: Type of tokens being added (for display purposes)
    
    Returns:
        Number of tokens successfully added
    """
    if not tokens:
        print(f"No new {token_type} tokens to add")
        return 0
    
    print(f"\nAdding {len(tokens)} new {token_type} tokens to tokenizer...")
    
    # Show sample of tokens being added
    if len(tokens) <= 10:
        for token in tokens:
            print(f"  - {token}")
    else:
        # Show first and last few tokens for large lists
        for token in tokens[:3]:
            print(f"  - {token}")
        print(f"  ... ({len(tokens) - 6} more tokens)")
        for token in tokens[-3:]:
            print(f"  - {token}")
    
    # Add tokens to tokenizer
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": tokens
    })
    
    print(f"Successfully added {num_added} new tokens")
    return num_added



# ============================================================================
# Token Generation Functions
# ============================================================================

def generate_structure_tokens():
    """Generate EMU3 structure tokens for image formatting."""
    return [
        "<|img_start|>",          # BOI token (begin of image)
        "<|img_end|>",            # EOI token (end of image)
        "<|img_token_start|>",    # IMG token (marks start of visual tokens)
        "<|img_end_of_row|>",     # EOL token (end of line/row)
        "<|img_end_of_frame|>",   # EOF token (end of frame)
    ]


def generate_reserved_tokens(num_tokens: int):
    """Generate reserved tokens for future use."""
    return [f"<|RESERVED_{i:03d}|>" for i in range(num_tokens)]


def generate_visual_tokens(vocab_size: int):
    """Generate visual tokens in EMU3 format."""
    return [f"<|visual token {i:06d}|>" for i in range(vocab_size)]


def collect_tokens_to_add(existing_vocab, visual_vocab_size: int, num_reserved_tokens: int):
    """Collect all tokens that need to be added to the tokenizer."""
    tokens_to_add = []
    stats = {"structure_tokens_added": 0, "reserved_tokens_added": 0, "visual_tokens_added": 0}
    
    # Structure tokens
    structure_tokens = generate_structure_tokens()
    new_structure_tokens = [t for t in structure_tokens if t not in existing_vocab]
    if new_structure_tokens:
        tokens_to_add.extend(new_structure_tokens)
        stats["structure_tokens_added"] = len(new_structure_tokens)
        print(f"Adding {len(new_structure_tokens)} structure tokens")
    
    # Reserved tokens
    reserved_tokens = generate_reserved_tokens(num_reserved_tokens)
    tokens_to_add.extend(reserved_tokens)
    stats["reserved_tokens_added"] = num_reserved_tokens
    print(f"Adding {num_reserved_tokens} reserved tokens")
    
    # Visual tokens
    visual_tokens = generate_visual_tokens(visual_vocab_size)
    tokens_to_add.extend(visual_tokens)
    stats["visual_tokens_added"] = visual_vocab_size
    print(f"Adding {visual_vocab_size} visual tokens")
    
    return tokens_to_add, stats


def save_vision_mapping(tokenizer, save_path: str, visual_vocab_size: int, stats: dict):
    """Save vision token mapping to JSON file."""
    vision_mapping = {}
    for i in range(visual_vocab_size):
        token = f"<|visual token {i:06d}|>"
        token_id = tokenizer.convert_tokens_to_ids(token)
        vision_mapping[i] = token_id
    
    mapping_path = os.path.join(save_path, "vision_token_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump({
            "vision_token_ids": vision_mapping,
            "visual_vocab_size": visual_vocab_size,
            "vision_token_format": "<|visual token {:06d}|>",
            **stats
        }, f, indent=2)
    print(f"Saved vision token mapping to {mapping_path}")


def verify_tokens(tokenizer, visual_vocab_size: int, num_reserved_tokens: int):
    """Verify that tokens were added correctly."""
    print("\nVerification - Sample token IDs:")
    
    # Check structure tokens
    for token in ["<|img_start|>", "<|img_end|>", "<|img_end_of_row|>"]:
        if token in tokenizer.get_vocab():
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"  {token}: ID {token_id}")
    
    # Check sample reserved tokens
    print(f"  <|RESERVED_000|>: ID {tokenizer.convert_tokens_to_ids('<|RESERVED_000|>')}")
    if num_reserved_tokens > 0:
        last_reserved = f"<|RESERVED_{num_reserved_tokens-1:03d}|>"
        print(f"  {last_reserved}: ID {tokenizer.convert_tokens_to_ids(last_reserved)}")
    
    # Check sample visual tokens
    sample_indices = [0, visual_vocab_size // 2, visual_vocab_size - 1]
    for idx in sample_indices:
        token = f"<|visual token {idx:06d}|>"
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {token}: ID {token_id}")


# ============================================================================
# Main Function
# ============================================================================

def add_emu3_special_tokens(
    model_path: str,
    output_path: Optional[str] = None,
    visual_vocab_size: int = 32768,
    num_reserved_tokens: int = 100
):
    """
    Add EMU3-style special tokens to any Hugging Face tokenizer.
    
    Args:
        model_path: Path to the model/tokenizer directory
        output_path: Path to save updated tokenizer (if None, overwrites original)
        visual_vocab_size: Number of visual tokens to add (default: 32768)
        num_reserved_tokens: Number of reserved tokens to add (default: 100)
    
    Returns:
        Tuple of (tokenizer object, stats dict)
    """
    # Load tokenizer
    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    original_vocab_size = len(tokenizer)
    print(f"Original vocabulary size: {original_vocab_size}")
    
    # Collect tokens to add
    existing_vocab = tokenizer.get_vocab()
    tokens_to_add, add_stats = collect_tokens_to_add(
        existing_vocab, visual_vocab_size, num_reserved_tokens
    )
    
    # Deduplicate and add tokens
    unique_tokens = deduplicate_tokens(tokens_to_add, existing_vocab)
    if unique_tokens:
        add_tokens_with_feedback(tokenizer, unique_tokens)
    
    # Save tokenizer with correct vocab_size
    save_path = output_path or model_path
    print(f"\nSaving updated tokenizer to {save_path}")
    save_tokenizer_with_vocab_fix(tokenizer, save_path)
    
    # Create final stats
    stats = {
        "original_vocab_size": original_vocab_size,
        "final_vocab_size": len(tokenizer),
        **add_stats
    }
    
    # Save vision mapping
    save_vision_mapping(tokenizer, save_path, visual_vocab_size, stats)
    
    # Handle generation mode if available
    if HAS_GENERATION_MODE:
        manager = HiddenModeTokenRegistry()
        manager.select_generation_token(num_reserved_tokens)
        manager.save_to_file(save_path, tokenizer)
        print("[PRIVATE] Generation mode token configured")
    
    # Verify tokens
    verify_tokens(tokenizer, visual_vocab_size, num_reserved_tokens)
    
    print(f"\nNew vocabulary size: {stats['final_vocab_size']}")
    return tokenizer, stats



# ============================================================================
# Vision Token Mapping Functions
# ============================================================================

def load_vision_token_mapping(tokenizer_path: str):
    """
    Load vision token mapping from a tokenizer directory.
    
    Args:
        tokenizer_path: Path to tokenizer directory containing vision_token_mapping.json
        
    Returns:
        dict: Mapping information including vision_token_ids
    """
    mapping_path = os.path.join(tokenizer_path, "vision_token_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Vision token mapping not found at {mapping_path}")



def get_vision_token_id(vision_index: int, tokenizer_path: str = None, mapping: dict = None):
    """
    Convert vision index to token ID using saved mapping.
    
    Args:
        vision_index: Index from vision tokenizer (0 to visual_vocab_size-1)
        tokenizer_path: Path to tokenizer directory (if mapping not provided)
        mapping: Pre-loaded mapping dict (if tokenizer_path not provided)
        
    Returns:
        int: Token ID in the vocabulary
    """
    if mapping is None:
        mapping = load_vision_token_mapping(tokenizer_path)
    
    vision_token_ids = mapping["vision_token_ids"]
    if str(vision_index) in vision_token_ids:
        return vision_token_ids[str(vision_index)]
    else:
        raise ValueError(f"Vision index {vision_index} not found. Valid range: 0-{mapping['visual_vocab_size']-1}")



def create_emu3_sequence_example(h: int = 2, w: int = 2):
    """
    Create an example EMU3-style sequence for a small image.
    
    Args:
        h: Height in tokens
        w: Width in tokens
    
    Returns:
        String representation of the sequence
    """
    # Token names
    boi_token = "<|img_start|>"
    eoi_token = "<|img_end|>"
    img_token = "<|img_token_start|>"
    eol_token = "<|img_end_of_row|>"
    eof_token = "<|img_end_of_frame|>"
    
    # Build sequence
    sequence_parts = []
    
    # Add BOI
    sequence_parts.append(boi_token)
    
    # Add metadata - dimensions
    sequence_parts.append(f"{h}*{w}")

    # Add IMG token
    sequence_parts.append(img_token)
    
    # Add visual tokens with EOL between rows
    token_idx = 0
    for row in range(h):
        for _ in range(w):
            sequence_parts.append(f"<|visual token {token_idx:06d}|>")
            token_idx += 1
        if row < h - 1:  # Add EOL between rows (not after last row)
            sequence_parts.append(eol_token)
    
    # Add final EOL, EOF, EOI
    sequence_parts.append(eol_token)
    sequence_parts.append(eof_token)
    sequence_parts.append(eoi_token)
    
    return "".join(sequence_parts)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add EMU3-style special tokens to any Hugging Face tokenizer")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model/tokenizer directory")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Path to save updated tokenizer (default: overwrite original)")
    parser.add_argument("--visual-vocab-size", type=int, default=32768,
                        help="Number of visual tokens to add (default: 32768)")
    parser.add_argument("--num-reserved-tokens", type=int, default=100,
                        help="Number of reserved tokens to add (default: 100)")
    parser.add_argument("--example", action="store_true",
                        help="Show example EMU3 sequence after updating")
    
    args = parser.parse_args()
    
    # Update tokenizer
    tokenizer, stats = add_emu3_special_tokens(
        model_path=args.model_path,
        output_path=args.output_path,
        visual_vocab_size=args.visual_vocab_size,
        num_reserved_tokens=args.num_reserved_tokens
    )
    
    print("\n" + "="*60)
    print("TOKENIZER UPDATE SUMMARY")
    print("="*60)
    print(f"Original vocabulary size: {stats['original_vocab_size']:,}")
    print(f"Structure tokens added:   {stats['structure_tokens_added']:,}")
    print(f"Reserved tokens added:    {stats['reserved_tokens_added']:,}")
    print(f"Visual tokens added:      {stats['visual_tokens_added']:,}")
    print(f"Final vocabulary size:    {stats['final_vocab_size']:,}")
    print(f"Total tokens added:       {stats['final_vocab_size'] - stats['original_vocab_size']:,}")
    
    # Show example if requested
    if args.example:
        print("\n" + "="*60)
        print("EXAMPLE EMU3 SEQUENCE")
        print("="*60)
        
        # Load the updated tokenizer
        tokenizer_path = args.output_path or args.model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Create example sequence
        example_seq = create_emu3_sequence_example(h=2, w=2)
        print(f"\n2x2 image sequence (first 200 chars):")
        print(example_seq[:200] + "..." if len(example_seq) > 200 else example_seq)
        
        # Tokenize it
        token_ids = tokenizer.encode(example_seq, add_special_tokens=False)
        print(f"\nTokenized to {len(token_ids)} tokens")
        print(f"First 15 token IDs: {token_ids[:15]}...")
        
        # Show structure
        print("\nSequence structure:")
        print("  [BOI] H*W [IMG] <visual_tokens> [EOL] ... [EOF] [EOI]")
        print(f"  Where each visual token maps to a single vocabulary ID")
    
    print("\n" + "="*60)