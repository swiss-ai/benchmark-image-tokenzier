"""Shared image resize and token geometry helpers."""

import math
from typing import Optional, Tuple


def smart_resize_dims(
    height: int,
    width: int,
    *,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    factor: int,
) -> Tuple[int, int]:
    """Apply the tokenizer's smart resize policy to image dimensions."""
    if min_pixels is None or max_pixels is None:
        return height, width

    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor}"
        )

    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor

    if resized_height * resized_width > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        resized_height = math.floor(height / scale / factor) * factor
        resized_width = math.floor(width / scale / factor) * factor
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor

    return resized_height, resized_width


def estimate_image_tokens(
    height: int,
    width: int,
    *,
    spatial_factor: int = 16,
) -> int:
    """Estimate emitted image-token sequence length for one resized image."""
    token_height = height // spatial_factor
    token_width = width // spatial_factor
    vision_tokens = token_height * token_width
    structural_tokens = (
        1  # BOS
        + 1  # img_start
        + 3  # dimension tokens
        + 1  # img_token_start
        + token_height  # EOL per row
        + 1  # EOF
        + 1  # img_end
        + 1  # EOS
    )
    return vision_tokens + structural_tokens
