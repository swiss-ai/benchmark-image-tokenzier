"""Shared image resize and token geometry helpers."""

import math
import os
from typing import Optional, Tuple

import numpy as np

from vision_tokenization.common.layout import (
    dim_tokens_upper_bound,
    image_sequence_length,
)

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - optional dependency
    njit = None
    prange = None


_USE_NUMBA = os.environ.get("VISION_TOKENIZATION_USE_NUMBA", "1").lower() not in {
    "0",
    "false",
    "no",
}
_NUMBA_AVAILABLE = njit is not None and _USE_NUMBA


def _pixel_limit(value: Optional[int]) -> int:
    """Encode optional pixel bounds for compiled helpers."""
    return -1 if value is None else int(value)


if njit is not None:
    @njit(cache=True)
    def _smart_resize_dims_numba(
        height: int,
        width: int,
        min_pixels: int,
        max_pixels: int,
        factor: int,
    ) -> Tuple[int, int]:
        if min_pixels < 0 or max_pixels < 0:
            return height, width

        if height < factor or width < factor:
            raise ValueError("height/width must be larger than factor")

        resized_height = int(round(height / factor)) * factor
        resized_width = int(round(width / factor)) * factor

        if resized_height * resized_width > max_pixels:
            scale = math.sqrt((height * width) / max_pixels)
            resized_height = math.floor(height / scale / factor) * factor
            resized_width = math.floor(width / scale / factor) * factor
        elif resized_height * resized_width < min_pixels:
            scale = math.sqrt(min_pixels / (height * width))
            resized_height = math.ceil(height * scale / factor) * factor
            resized_width = math.ceil(width * scale / factor) * factor

        return resized_height, resized_width


    @njit(parallel=True, cache=True)
    def _smart_resize_dims_batch_numba(
        heights: np.ndarray,
        widths: np.ndarray,
        min_pixels: int,
        max_pixels: int,
        factor: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(heights)
        out_h = np.empty(n, dtype=np.int32)
        out_w = np.empty(n, dtype=np.int32)
        for idx in prange(n):
            rh, rw = _smart_resize_dims_numba(
                int(heights[idx]), int(widths[idx]),
                min_pixels, max_pixels, factor,
            )
            out_h[idx] = rh
            out_w[idx] = rw
        return out_h, out_w


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
    return image_sequence_length(
        token_height, token_width, dim_tokens_upper_bound(token_height, token_width))


def smart_resize_dims_batch(
    heights: np.ndarray,
    widths: np.ndarray,
    *,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    factor: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized smart_resize_dims for arrays of heights/widths.

    Returns (final_heights, final_widths) as int32 arrays.
    """
    heights_arr = np.asarray(heights, dtype=np.int64)
    widths_arr = np.asarray(widths, dtype=np.int64)
    min_pixels_i = _pixel_limit(min_pixels)
    max_pixels_i = _pixel_limit(max_pixels)

    if _NUMBA_AVAILABLE:
        return _smart_resize_dims_batch_numba(
            heights_arr, widths_arr, min_pixels_i, max_pixels_i, int(factor),
        )

    n = len(heights_arr)
    out_h = np.empty(n, dtype=np.int32)
    out_w = np.empty(n, dtype=np.int32)
    for idx in range(n):
        rh, rw = smart_resize_dims(
            int(heights_arr[idx]), int(widths_arr[idx]),
            min_pixels=min_pixels, max_pixels=max_pixels, factor=factor,
        )
        out_h[idx] = rh
        out_w[idx] = rw
    return out_h, out_w
