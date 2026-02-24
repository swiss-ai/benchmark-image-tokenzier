#!/usr/bin/env python3
"""
Core utilities for creating omni-tokenizers.

Shared functions for adding vision tokens to text tokenizers.
"""

import json
import os
import sys
from typing import Any, Dict, Tuple

from transformers import AutoTokenizer

# Modality registry: name -> (mapping_file, offset_key, vocab_size_key, start_token, end_token)
MODALITY_REGISTRY = {
    "vision": ("vision_token_mapping.json", "vision_token_offset", "visual_vocab_size", "<|img_start|>", "<|img_end|>"),
    "audio": ("audio_token_mapping.json", "audio_token_offset", "audio_vocab_size", "<|audio_start|>", "<|audio_end|>"),
}


def _read_modality_info(output_path: str, name: str, tokenizer) -> Dict[str, Any] | None:
    """Read modality info from its mapping file. Returns None if not found or incomplete."""
    if name not in MODALITY_REGISTRY:
        return None

    mapping_file, offset_key, vocab_key, start_token, end_token = MODALITY_REGISTRY[name]
    mapping_path = os.path.join(output_path, mapping_file)

    if not os.path.exists(mapping_path):
        return None

    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if offset_key not in data:
        return None

    start_token_id = tokenizer.convert_tokens_to_ids(start_token)
    end_token_id = tokenizer.convert_tokens_to_ids(end_token)

    if start_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Modality start token {start_token} not found in tokenizer at {output_path}")
    if end_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Modality end token {end_token} not found in tokenizer at {output_path}")

    return {
        "name": name,
        "offset": data[offset_key],
        "vocab_size": data.get(vocab_key),
        "start_token": start_token_id,
        "end_token": end_token_id,
    }


def _build_omnimodal_config(output_path: str, base_vocab_size: int, tokenizer) -> Dict[str, Any]:
    """
    Build omnimodal_config from all present modality mapping files.

    Stores raw data only - omni_special_vocab_size computed at load time as:
        first_modality_offset - omni_special_token_offset
    """
    modalities = [
        info for name in MODALITY_REGISTRY if (info := _read_modality_info(output_path, name, tokenizer)) is not None
    ]

    if not modalities:
        return {}

    modalities.sort(key=lambda m: m["offset"])

    return {
        "omni_special_token_offset": base_vocab_size,
        "modalities": modalities,
    }


# Add Tokenizer directory to path for importing vision tokenizers
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


# Vision tokenizer mapping - maps user-friendly names to implementation classes
VISION_TOKENIZER_MAP = {
    "Emu3": {
        "class": "Emu3VisionTokenizer",
        "module": "Tokenizer.Emu3VisionTokenizer",
    },
    "Emu3.5": {
        "class": "Emu3_5_IBQ",
        "module": "Tokenizer.Emu3_5_IBQ",
    },
}


def save_tokenizer_with_correct_vocab_size(
    tokenizer,
    save_path: str,
    original_vocab_size: int,
    vision_tokenizer_type: str,
    vision_tokenizer_path: str,
    codebook_size: int,
) -> None:
    """
    Save tokenizer with correct vocab_size and vision tokenizer info in config.

    Args:
        tokenizer: The tokenizer instance with potentially updated vocabulary
        save_path: Path where tokenizer will be saved
        original_vocab_size: The original vocab size before adding tokens
        vision_tokenizer_type: Type of vision tokenizer (e.g., 'Emu3', 'Emu3.5') - REQUIRED
        vision_tokenizer_path: Path to vision tokenizer model - REQUIRED
        codebook_size: Vision tokenizer codebook size - REQUIRED
    """
    # First save the tokenizer
    tokenizer.save_pretrained(save_path)

    # Get actual vocabulary size after adding tokens
    actual_vocab_size = len(tokenizer.get_vocab())
    base_vocab_size = original_vocab_size

    # Update tokenizer_config.json with correct vocab_size
    config_path = os.path.join(save_path, "tokenizer_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Set vocab_size to actual size for model.resize_token_embeddings() etc.
    config["vocab_size"] = actual_vocab_size
    config["base_vocab_size"] = base_vocab_size
    config["added_tokens_count"] = actual_vocab_size - base_vocab_size

    # Add vision tokenizer configuration
    config["vision_tokenizer"] = {
        "type": vision_tokenizer_type,
        "path": vision_tokenizer_path,
        "codebook_size": codebook_size,
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✓ Updated tokenizer_config.json:")
    print(f"  Total vocabulary: {actual_vocab_size:,}")
    print(f"  Base vocabulary: {base_vocab_size:,}")
    print(f"  Added tokens: {actual_vocab_size - base_vocab_size:,}")
    print(f"  Vision tokenizer: {vision_tokenizer_type}")
    print(f"  Vision tokenizer path: {vision_tokenizer_path}")
    print(f"  Codebook size: {codebook_size:,}")


def rename_reserved_token(save_path: str, tokenizer, old_token: str, new_token: str) -> None:
    """
    Rename a reserved token in saved tokenizer files (e.g., RESERVED_001 -> GEN:IMAGE:MODE).

    Args:
        save_path: Path where tokenizer was saved
        tokenizer: Tokenizer instance to get token ID
        old_token: Old token name (e.g., "<|RESERVED_001|>")
        new_token: New token name (e.g., "<|GEN:IMAGE:MODE|>")
    """
    token_id = tokenizer.convert_tokens_to_ids(old_token)
    if token_id == tokenizer.unk_token_id:
        print(f"  ⚠️  {old_token} not found, skipping rename to {new_token}")
        return

    # Modify tokenizer.json
    tokenizer_json_path = os.path.join(save_path, "tokenizer.json")
    if os.path.exists(tokenizer_json_path):
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Replace all occurrences
        content = content.replace(f'"{old_token}"', f'"{new_token}"')
        content = content.replace(old_token, new_token)

        with open(tokenizer_json_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"  ✅ Renamed {old_token} to {new_token} (ID {token_id}) in tokenizer.json")

    # Also modify tokenizer_config.json
    config_path = os.path.join(save_path, "tokenizer_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Replace in all config values
        def replace_in_dict(obj):
            if isinstance(obj, dict):
                return {k: replace_in_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_in_dict(item) for item in obj]
            elif isinstance(obj, str):
                return obj.replace(old_token, new_token)
            return obj

        config = replace_in_dict(config)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"  ✅ Renamed {old_token} to {new_token} in tokenizer_config.json")


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
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": tokens})

    print(f"Successfully added {num_added} new tokens")
    return num_added


def load_vision_tokenizer(vision_tokenizer_path: str, vision_tokenizer: str):
    """
    Load a vision tokenizer and extract its metadata.

    Args:
        vision_tokenizer_path: Path to the vision tokenizer model
        vision_tokenizer: User-friendly tokenizer name (e.g., "Emu3", "Emu3.5")

    Returns:
        Tuple of (tokenizer instance, codebook_size, tokenizer_name)
    """
    if vision_tokenizer not in VISION_TOKENIZER_MAP:
        available = list(VISION_TOKENIZER_MAP.keys())
        raise ValueError(f"Unknown vision tokenizer: '{vision_tokenizer}'. " f"Available: {available}")

    print(f"Loading vision tokenizer metadata ({vision_tokenizer})...")
    try:
        # Get tokenizer configuration
        config = VISION_TOKENIZER_MAP[vision_tokenizer]

        # Import only the needed class
        module = __import__(config["module"], fromlist=[config["class"]])
        TokenizerClass = getattr(module, config["class"])

        # Load metadata only (fast, no model weights)
        vision_tok = TokenizerClass(model_path=vision_tokenizer_path, metadata_only=True)

        codebook_size = vision_tok.codebook_size
        tokenizer_name = vision_tok.name

        print(f"  ✓ Vision tokenizer: {tokenizer_name}")
        print(f"  ✓ Codebook size: {codebook_size:,}")

        return vision_tok, codebook_size, tokenizer_name

    except Exception as e:
        raise RuntimeError(f"Failed to load vision tokenizer metadata for '{vision_tokenizer}': {e}")


def create_base_tokenizer(
    text_tokenizer_path: str,
    output_path: str,
    vision_tokenizer_path: str,
    vision_tokenizer: str,
    num_reserved_tokens: int = 200,
) -> Tuple[Any, Dict[str, int]]:
    """
    Create an omnimodal base tokenizer by adding vision tokens to a text tokenizer.

    This function:
    1. Auto-detects codebook size from the vision tokenizer
    2. Adds structure tokens for image formatting
    3. Adds reserved tokens (RESERVED_001 becomes <|GEN:IMAGE:MODE|>)
    4. Adds visual tokens: <|visual token 000000|> through <|visual token XXXXXX|>
    5. Saves vision_token_mapping.json with all token mappings

    Args:
        text_tokenizer_path: Path to the base text tokenizer directory
        output_path: Path to save omni-tokenizer
        vision_tokenizer_path: Path to vision tokenizer model
        vision_tokenizer: Vision tokenizer name (e.g., "Emu3", "Emu3.5")
        num_reserved_tokens: Number of RESERVED_OMNI tokens to add (default: 200)

    Returns:
        Tuple of (tokenizer object, stats dict)

    Files created:
        - tokenizer_config.json, special_tokens_map.json, etc. (standard tokenizer files)
        - vision_token_mapping.json (mapping from vision indices to token IDs)
    """

    print("=" * 60)
    print("CREATING OMNI-TOKENIZER (BASE)")
    print("=" * 60)

    # Load vision tokenizer to auto-detect codebook size
    _, visual_vocab_size, vision_tokenizer_name = load_vision_tokenizer(vision_tokenizer_path, vision_tokenizer)

    print(f"\nText tokenizer: {text_tokenizer_path}")
    print(f"Visual vocab size: {visual_vocab_size:,} (auto-detected)")
    print("=" * 60 + "\n")

    # Load the text tokenizer
    print(f"Loading text tokenizer from {text_tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path, use_fast=True)

    # Store original info - capture base vocab BEFORE any modifications
    original_vocab_size = len(tokenizer.get_vocab())
    print(f"Original vocabulary size: {original_vocab_size:,}")

    # Statistics
    stats = {
        "text_tokenizer": text_tokenizer_path,
        "vision_tokenizer": vision_tokenizer_name,
        "tokenizer_type": "base",
        "original_vocab_size": original_vocab_size,
        "visual_tokens_added": 0,
        "structure_tokens_added": 0,
        "reserved_tokens_added": 0,
        "final_vocab_size": 0,
    }

    # Collect all special tokens to add
    special_tokens_to_add = []

    # Get existing vocabulary
    existing_vocab = tokenizer.get_vocab()

    # Add RESERVED_OMNI tokens (200 tokens for all modalities and control)
    print(f"\nAdding {num_reserved_tokens} RESERVED_OMNI tokens...")
    reserved_tokens = []
    for i in range(num_reserved_tokens):
        reserved_tokens.append(f"<|RESERVED_OMNI_{i:03d}|>")

    special_tokens_to_add.extend(reserved_tokens)
    stats["reserved_tokens_added"] = num_reserved_tokens
    stats["structure_tokens_added"] = 7  # Will be renamed from RESERVED_OMNI_001-007

    # Add visual tokens
    print(f"\nGenerating {visual_vocab_size:,} visual tokens for {vision_tokenizer_name}...")
    visual_tokens = []
    for i in range(visual_vocab_size):
        visual_tokens.append(f"<|visual token {i}|>")

    special_tokens_to_add.extend(visual_tokens)
    stats["visual_tokens_added"] = len(visual_tokens)
    print(f"Token format: <|visual token N|>")
    print(f"Range: <|visual token 0|> to <|visual token {visual_vocab_size-1}|>")

    # Deduplicate and filter tokens
    unique_new_tokens = deduplicate_tokens(special_tokens_to_add, existing_vocab)

    # Add tokens with user feedback
    add_tokens_with_feedback(tokenizer, unique_new_tokens)

    # Update final vocab size
    stats["final_vocab_size"] = len(tokenizer)
    print(f"\nNew vocabulary size: {stats['final_vocab_size']:,}")

    # Save the updated tokenizer and update vocab_size in config
    print(f"\nSaving omni-tokenizer to {output_path}...")
    save_tokenizer_with_correct_vocab_size(
        tokenizer,
        output_path,
        original_vocab_size,
        vision_tokenizer_type=vision_tokenizer,
        vision_tokenizer_path=vision_tokenizer_path,
        codebook_size=visual_vocab_size,
    )

    # Rename RESERVED_OMNI tokens to image structure tokens
    print(f"\nRenaming RESERVED_OMNI tokens to image structure tokens...")
    token_renames = [
        ("<|RESERVED_OMNI_001|>", "<|img_start|>"),
        ("<|RESERVED_OMNI_002|>", "<|img_end|>"),
        ("<|RESERVED_OMNI_003|>", "<|img_token_start|>"),
        ("<|RESERVED_OMNI_004|>", "<|img_end_of_row|>"),
        ("<|RESERVED_OMNI_005|>", "<|img_end_of_frame|>"),
        ("<|RESERVED_OMNI_006|>", "<|img_generation_start|>"),
        ("<|RESERVED_OMNI_007|>", "<|image|>"),  # For chat template (instruct will use this)
    ]

    for old_token, new_token in token_renames:
        rename_reserved_token(output_path, tokenizer, old_token, new_token)

    print(f"  ℹ️  Remember to re-initialize <|img_generation_start|> embedding before release")

    # Save vision token mapping
    print("\nCreating vision token mapping...")
    vision_mapping = {}
    for i in range(visual_vocab_size):
        token = f"<|visual token {i}|>"
        token_id = tokenizer.convert_tokens_to_ids(token)
        vision_mapping[i] = token_id

    # Save public mapping file
    mapping_path = os.path.join(output_path, "vision_token_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(
            {
                "tokenizer_type": "base",
                "vision_tokenizer": vision_tokenizer_name,
                "vision_tokenizer_type": vision_tokenizer,
                "vision_tokenizer_path": vision_tokenizer_path,
                "vision_token_ids": vision_mapping,
                "vision_token_offset": vision_mapping[0],
                "visual_vocab_size": visual_vocab_size,
                "vision_token_format": "<|visual token N|>",
                "num_reserved_tokens": num_reserved_tokens,
                "original_vocab_size": original_vocab_size,
                "structure_tokens_added": stats["structure_tokens_added"],
                "reserved_tokens_added": stats["reserved_tokens_added"],
                "final_vocab_size": stats["final_vocab_size"],
            },
            f,
            indent=2,
        )
    print(f"Saved vision token mapping to {mapping_path}")

    # Now update tokenizer_config.json with omnimodal_config (after mapping files exist)
    config_path = os.path.join(output_path, "tokenizer_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    tokenizer_for_config = AutoTokenizer.from_pretrained(output_path, use_fast=True)
    omnimodal_config = _build_omnimodal_config(output_path, original_vocab_size, tokenizer_for_config)
    if omnimodal_config:
        config["omnimodal_config"] = omnimodal_config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Updated omnimodal_config in tokenizer_config.json")

    # Verification - show some token IDs
    print("\n" + "=" * 60)
    print("VERIFICATION - Sample Token IDs")
    print("=" * 60)

    # Check boundary marker
    print("\nBoundary marker:")
    boundary_id = tokenizer.convert_tokens_to_ids("<|RESERVED_OMNI_000|>")
    if boundary_id != tokenizer.unk_token_id:
        print(f"  <|RESERVED_OMNI_000|>: ID {boundary_id}")

    # Check image structure tokens
    print("\nImage structure tokens:")
    for token in ["<|img_start|>", "<|img_end|>", "<|img_token_start|>", "<|img_end_of_row|>", "<|img_end_of_frame|>"]:
        if token in tokenizer.get_vocab():
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"  {token}: ID {token_id}")

    # Check special image tokens
    print("\nSpecial image tokens:")
    gen_token_id = tokenizer.convert_tokens_to_ids("<|img_generation_start|>")
    if gen_token_id != tokenizer.unk_token_id:
        print(f"  <|img_generation_start|>: ID {gen_token_id}")

    image_token_id = tokenizer.convert_tokens_to_ids("<|image|>")
    if image_token_id != tokenizer.unk_token_id:
        print(f"  <|image|>: ID {image_token_id}")

    # Check sample visual tokens
    print(f"\nVisual tokens (for {vision_tokenizer_name}):")
    sample_indices = [0, visual_vocab_size // 2, visual_vocab_size - 1]
    for idx in sample_indices:
        token = f"<|visual token {idx}|>"
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {token}: ID {token_id}")

    print(f"\nVision token offset: {vision_mapping[0]}")

    print("\n" + "=" * 60)

    return tokenizer, stats


def load_vision_token_mapping(tokenizer_path: str) -> Dict:
    """
    Load vision token mapping from a tokenizer directory.

    Args:
        tokenizer_path: Path to tokenizer directory containing vision_token_mapping.json

    Returns:
        dict: Mapping information including vision_token_ids
    """
    mapping_path = os.path.join(tokenizer_path, "vision_token_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Vision token mapping not found at {mapping_path}")


def get_vision_token_id(vision_index: int, tokenizer_path: str = None, mapping: dict = None) -> int:
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
