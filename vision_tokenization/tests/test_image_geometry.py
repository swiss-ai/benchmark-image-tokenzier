import numpy as np

from vision_tokenization.utils.image_geometry import (
    smart_resize_dims,
    smart_resize_dims_batch,
)


def test_batch_resize_matches_the_scalar_policy():
    """The compiled batch path is what the planner runs on every image.

    njit(cache=True) freezes the compiled form on disk, so a change to the scalar
    policy does not reach it. Comparing values is what catches the stale copy.
    """
    rng = np.random.default_rng(0)
    heights = rng.integers(64, 4096, 500).astype(np.int64)
    widths = rng.integers(64, 4096, 500).astype(np.int64)
    kw = dict(min_pixels=128 * 128, max_pixels=2048 * 2048)

    batch_h, batch_w = smart_resize_dims_batch(heights, widths, factor=16, **kw)
    for h, w, bh, bw in zip(heights, widths, batch_h, batch_w):
        assert (int(bh), int(bw)) == smart_resize_dims(int(h), int(w), factor=16, **kw)


def test_batch_resize_passes_dimensions_through_when_unbounded():
    heights = np.array([64, 128, 256], dtype=np.int64)
    widths = np.array([80, 144, 272], dtype=np.int64)

    batch_h, batch_w = smart_resize_dims_batch(
        heights, widths, factor=16, min_pixels=None, max_pixels=None)

    np.testing.assert_array_equal(batch_h, heights)
    np.testing.assert_array_equal(batch_w, widths)
