"""
EMU Tokenizers for vision-language models.

This module provides tokenizers for different data types:
- EMUImageOnlyTokenizer: For pure image tokenization
- EMUImageTextPairTokenizer: For image-text pair tokenization
- EMUSftTokenizer: For SFT/instruction-tuning data

Supports both Emu3 and Emu3.5 vision tokenizers.
"""

from typing import Union

from .image_only import EMUImageOnlyTokenizer
from .image_text_pair import EMUImageTextPairTokenizer
from .sft import EMUSftTokenizer


def create_tokenizer(
    mode: str,
    text_tokenizer_path: str,
    device: str = "cuda",
    *,
    min_pixels: int,
    max_pixels: int,
    **kwargs,
) -> Union[EMUImageOnlyTokenizer, EMUImageTextPairTokenizer, EMUSftTokenizer]:
    """
    Factory function to create the appropriate EMU tokenizer based on mode.

    Args:
        mode: Tokenization mode ("image_only", "image2text", "text2image", or "sft")
        text_tokenizer_path: Path to the text tokenizer
        device: Device for tokenization (cuda or cpu)
        min_pixels: Minimum pixels for tokenizer resize (required, no default)
        max_pixels: Maximum pixels for tokenizer resize (required, no default)
        **kwargs: Additional tokenizer-specific arguments

    Returns:
        The appropriate tokenizer instance based on mode

    Raises:
        ValueError: If mode is not recognized
    """
    tokenizers = {
        "image_only": EMUImageOnlyTokenizer,
        "image2text": EMUImageTextPairTokenizer,  # image->text (captioning)
        "text2image": EMUImageTextPairTokenizer,  # text->image (generation)
        "sft": EMUSftTokenizer,
    }

    if mode not in tokenizers:
        raise ValueError(f"Unknown tokenizer mode: {mode}. " f"Must be one of: {', '.join(tokenizers.keys())}")

    tokenizer_class = tokenizers[mode]

    return tokenizer_class(
        text_tokenizer_path=text_tokenizer_path,
        device=device,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        mode=mode,
        **kwargs,
    )


__all__ = ["EMUImageOnlyTokenizer", "EMUImageTextPairTokenizer", "EMUSftTokenizer", "create_tokenizer"]
