"""
Omni-Tokenizer Creation Module

This module provides utilities for creating omnimodal tokenizers by adding
vision tokens to text tokenizers. Supports multiple vision tokenizers with
different codebook sizes (Emu3, Emu3.5, etc.).

Architecture:
- Base Tokenizer: Extends base text tokenizer for pretraining
- Instruct Tokenizer: Extends instruct text tokenizer for SFT/chat

Main functions:
- create_base_tokenizer: Create base omni-tokenizer (auto-detects codebook size)
- create_instruct_tokenizer: Create instruct omni-tokenizer with chat template & SFT sequences
- load_vision_token_mapping: Load vision token mapping from tokenizer
- get_vision_token_id: Convert vision index to token ID

Usage:
    from omni_tokenizer import create_base_tokenizer, create_instruct_tokenizer

    # Create base tokenizer (for pretraining)
    base_tok, stats = create_base_tokenizer(
        text_tokenizer_path="meta-llama/Llama-3.2-3B",
        vision_tokenizer_path="BAAI/Emu3-VisionTokenizer",
        vision_tokenizer="Emu3",
        output_path="./llama3_emu3_base"
    )

    # Create instruct tokenizer (for SFT/chat)
    instruct_tok, stats = create_instruct_tokenizer(
        text_tokenizer_path="meta-llama/Llama-3.2-3B-Instruct",
        vision_tokenizer_path="BAAI/Emu3-VisionTokenizer",
        vision_tokenizer="Emu3",
        output_path="./llama3_emu3_instruct"
    )
"""

from .core import (
    create_base_tokenizer,
    get_vision_token_id,
    load_vision_token_mapping,
)
from .create_instruct import create_instruct_tokenizer

__all__ = [
    "create_base_tokenizer",
    "create_instruct_tokenizer",
    "load_vision_token_mapping",
    "get_vision_token_id",
]
