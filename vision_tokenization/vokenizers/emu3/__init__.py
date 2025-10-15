"""
EMU3 Tokenizers for vision-language models.

This module provides tokenizers for different data types:
- EMU3ImageOnlyTokenizer: For pure image tokenization
- EMU3ImageTextPairTokenizer: For image-text pair tokenization
- EMU3SftTokenizer: For SFT/instruction-tuning data
"""

from .image_only import EMU3ImageOnlyTokenizer
from .image_text_pair import EMU3ImageTextPairTokenizer
from .sft import EMU3SftTokenizer
from typing import Union


def create_tokenizer(
    mode: str,
    text_tokenizer_path: str,
    device: str = "cuda",
    min_pixels: int = 512 * 512,
    max_pixels: int = 1024 * 1024,
    **kwargs
) -> Union[EMU3ImageOnlyTokenizer, EMU3ImageTextPairTokenizer, EMU3SftTokenizer]:
    """
    Factory function to create the appropriate EMU3 tokenizer based on mode.

    Args:
        mode: Tokenization mode ("image_only", "image2text", "text2image", or "sft")
        text_tokenizer_path: Path to the text tokenizer
        device: Device for tokenization (cuda or cpu)
        min_pixels: Minimum pixels for image preprocessing
        max_pixels: Maximum pixels for image preprocessing
        **kwargs: Additional tokenizer-specific arguments

    Returns:
        The appropriate tokenizer instance based on mode

    Raises:
        ValueError: If mode is not recognized
    """
    tokenizers = {
        "image_only": EMU3ImageOnlyTokenizer,
        "image2text": EMU3ImageTextPairTokenizer,  # image->text (captioning)
        "text2image": EMU3ImageTextPairTokenizer,  # text->image (generation)
        "sft": EMU3SftTokenizer
    }

    if mode not in tokenizers:
        raise ValueError(
            f"Unknown tokenizer mode: {mode}. "
            f"Must be one of: {', '.join(tokenizers.keys())}"
        )

    tokenizer_class = tokenizers[mode]

    return tokenizer_class(
        text_tokenizer_path=text_tokenizer_path,
        device=device,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        mode=mode,
        **kwargs
    )


__all__ = [
    'EMU3ImageOnlyTokenizer',
    'EMU3ImageTextPairTokenizer',
    'EMU3SftTokenizer',
    'create_tokenizer'
]