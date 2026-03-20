import numpy as np

from vision_tokenization.utils.image_geometry import (
    estimate_image_tokens,
    estimate_image_tokens_batch,
    smart_resize_dims,
)


def test_estimate_image_tokens_batch_matches_scalar_helpers():
    heights = np.array([128, 255, 512, 777, 1400], dtype=np.int32)
    widths = np.array([128, 511, 384, 1025, 896], dtype=np.int32)

    batch_tokens = estimate_image_tokens_batch(
        heights,
        widths,
        spatial_factor=16,
        min_pixels=128 * 128,
        max_pixels=1400 * 1400,
    )

    scalar_tokens = []
    for height, width in zip(heights, widths):
        final_height, final_width = smart_resize_dims(
            int(height),
            int(width),
            min_pixels=128 * 128,
            max_pixels=1400 * 1400,
            factor=16,
        )
        scalar_tokens.append(
            estimate_image_tokens(
                final_height,
                final_width,
                spatial_factor=16,
            )
        )

    assert np.array_equal(batch_tokens, np.array(scalar_tokens, dtype=np.int64))


def test_estimate_image_tokens_batch_handles_unbounded_resize():
    heights = np.array([64, 128, 256], dtype=np.int32)
    widths = np.array([80, 144, 272], dtype=np.int32)

    batch_tokens = estimate_image_tokens_batch(
        heights,
        widths,
        spatial_factor=16,
        min_pixels=None,
        max_pixels=None,
    )

    expected = np.array(
        [estimate_image_tokens(int(h), int(w), spatial_factor=16) for h, w in zip(heights, widths)],
        dtype=np.int64,
    )
    assert np.array_equal(batch_tokens, expected)
