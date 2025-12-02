#!/usr/bin/env python3
"""
Factory function for creating vision v_tokenizers for VLM benchmarking.

This module provides a simple factory pattern for instantiating different
vision tokenizer implementations. New v_tokenizers can be easily added by:
1. Creating a new file implementing VLMVisionTokenizer
2. Registering it in the TOKENIZER_REGISTRY below
"""

from typing import Any, Dict
from .base import VLMVisionTokenizer


# Registry of available v_tokenizers
TOKENIZER_REGISTRY = {}


def register_tokenizer(name: str):
    """Decorator to register a tokenizer class."""
    def decorator(cls):
        TOKENIZER_REGISTRY[name] = cls
        return cls
    return decorator


def create_vision_tokenizer(tokenizer_type: str, **kwargs) -> VLMVisionTokenizer:
    """
    Factory function to create a vision tokenizer instance.

    This function instantiates the appropriate vision tokenizer based on the
    tokenizer_type string. It follows the factory pattern used in
    vision_tokenization/vokenizers/emu/__init__.py.

    Args:
        tokenizer_type: Type of tokenizer to create. Options:
                       - 'emu3': EMU3 vision tokenizer
                       - 'emu3.5', 'emu3.5-ibq': EMU3.5 IBQ vision tokenizer
                       - Additional v_tokenizers as registered
        **kwargs: Tokenizer-specific initialization arguments:
                 - min_pixels: Minimum pixel count (default: 256*256)
                 - max_pixels: Maximum pixel count (default: 512*512)
                 - device: Device to load on (default: 'cuda')
                 - model_path: Path to tokenizer model (required for some v_tokenizers)

    Returns:
        VLMVisionTokenizer instance

    Raises:
        ValueError: If tokenizer_type is not supported

    Example:
        >>> # Create EMU3 tokenizer
        >>> tokenizer = create_vision_tokenizer(
        ...     'emu3',
        ...     min_pixels=256*256,
        ...     max_pixels=512*512,
        ...     device='cuda:1'
        ... )
        >>>
        >>> # Create EMU3.5 tokenizer
        >>> tokenizer = create_vision_tokenizer(
        ...     'emu3.5-ibq',
        ...     model_path='/path/to/emu3.5/tokenizer',
        ...     min_pixels=256*256,
        ...     max_pixels=512*512,
        ...     device='cuda:1'
        ... )
    """
    tokenizer_type_lower = tokenizer_type.lower()

    # Map tokenizer types to classes
    tokenizer_map = {
        'emu3': 'emu3',
        'emu3.5': 'emu3_5_ibq',
        'emu3.5-ibq': 'emu3_5_ibq',
    }

    if tokenizer_type_lower not in tokenizer_map:
        available = ', '.join(sorted(tokenizer_map.keys()))
        raise ValueError(
            f"Unsupported tokenizer type: '{tokenizer_type}'. "
            f"Available types: {available}"
        )

    # Import the appropriate tokenizer class
    tokenizer_module = tokenizer_map[tokenizer_type_lower]

    if tokenizer_module == 'emu3':
        from .emu3 import EMU3VisionTokenizer
        return EMU3VisionTokenizer(**kwargs)
    elif tokenizer_module == 'emu3_5_ibq':
        from .emu3_5_ibq import EMU35IBQVisionTokenizer
        return EMU35IBQVisionTokenizer(**kwargs)
    else:
        raise ValueError(f"Tokenizer module '{tokenizer_module}' not implemented")


__all__ = [
    'VLMVisionTokenizer',
    'create_vision_tokenizer',
    'register_tokenizer',
    'TOKENIZER_REGISTRY',
]