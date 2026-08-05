"""The image-block formula must have exactly one definition.

It used to be written four times: allocated in image_only, allocated again in common.assembly,
predicted in image_geometry, and predicted a fourth time inside a numba kernel as a bare ``9``.
They agreed by coincidence, and the numba copy could not be corrected by fixing the others.
"""

import numpy as np
import pytest

from vision_tokenization.common.layout import (
    dim_tokens_upper_bound,
    image_block_length,
    image_sequence_length,
)
from vision_tokenization.utils.image_geometry import estimate_image_tokens

SIZES = [(14, 14), (32, 32), (64, 96), (128, 128)]


@pytest.mark.parametrize("h,w", SIZES)
def test_sequence_is_block_plus_wrapper(h, w):
    assert image_sequence_length(h, w, 5) == image_block_length(h, w, 5) + 2


@pytest.mark.parametrize("h,w", SIZES)
def test_block_accounts_for_every_token(h, w):
    """img_start + dims + img_token_start + (vision + eol) per row + eof + img_end."""
    n_dims = 5
    expected = 1 + n_dims + 1 + h * w + h + 1 + 1
    assert image_block_length(h, w, n_dims) == expected


@pytest.mark.parametrize("h,w", SIZES)
def test_estimator_matches_the_formula(h, w):
    """The planner's prediction is the same arithmetic the writers allocate with."""
    assert estimate_image_tokens(h * 16, w * 16, spatial_factor=16) == \
        image_sequence_length(h, w, dim_tokens_upper_bound(h, w))


@pytest.mark.parametrize("h,w", SIZES + [(1024, 1024), (2048, 2048)])
def test_bound_is_never_below_the_real_token_count(h, w):
    """A token is never shorter than a character,
    so the character count of "H*W" bounds its token count from above
    for any tokenizer that does not split a character.
    Exact for the Apertus tokenizers, which keep digits separate."""
    assert dim_tokens_upper_bound(h, w) == len(f"{h}*{w}")
    for actual in range(1, dim_tokens_upper_bound(h, w) + 1):
        assert image_sequence_length(h, w, dim_tokens_upper_bound(h, w)) >= \
            image_sequence_length(h, w, actual)


def test_bound_grows_with_the_dimensions():
    """A fixed constant silently under-counts once dims reach four digits."""
    assert dim_tokens_upper_bound(32, 32) == 5
    assert dim_tokens_upper_bound(100, 100) == 7
    assert dim_tokens_upper_bound(1024, 1024) == 9


def test_bound_counts_zero_as_a_digit():
    assert dim_tokens_upper_bound(0, 6) == len("0*6")
    assert dim_tokens_upper_bound(0, 0) == len("0*0")
