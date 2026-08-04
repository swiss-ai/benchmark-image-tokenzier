"""The image-block wire format, in one place.

Both writers allocate from this, the planner predicts from it, and the numba
estimator freezes its constants at compile time. Kept free of torch and numba so
the planner, which is offline and torch-free, can import it.

One image block is::

    img_start · dim tokens · img_token_start · [row of vision ids · eol] × H · eof · img_end

and a standalone single-image sequence wraps that in BOS/EOS.
"""

IMAGE_BLOCK_FIXED_TOKENS = 4   # img_start, img_token_start, eof, img_end
SEQUENCE_WRAPPER_TOKENS = 2    # BOS, EOS

def dim_tokens_upper_bound(height: int, width: int) -> int:
    """Ceiling on how many tokens ``"H*W"`` encodes to, without a tokenizer.

    A token is never shorter than one character, so the character count bounds the
    token count from above for any tokenizer that does not split a character —
    byte-level BPE only ever merges, so it comes in at or under this. It happens to
    be exact for the Apertus tokenizers, which keep digits separate.

    The planner has no tokenizer and must bound rather than measure. Over-estimating
    only makes batches slightly smaller than they could be; under-estimating makes
    them overrun max_batch_tokens.
    """
    return len(f"{height}*{width}")


def image_block_length(height: int, width: int, n_dim_tokens: int) -> int:
    """Tokens in one image block, excluding the BOS/EOS wrapper."""
    return IMAGE_BLOCK_FIXED_TOKENS + n_dim_tokens + height * (width + 1)


def image_sequence_length(height: int, width: int, n_dim_tokens: int) -> int:
    """Tokens in a standalone single-image sequence, wrapper included."""
    return SEQUENCE_WRAPPER_TOKENS + image_block_length(height, width, n_dim_tokens)
