"""EMU tokenizer factory.

The package init stays lazy on purpose: importing this module should not load
all tokenizer implementations or their optional dependencies. Call
``create_tokenizer()`` to instantiate the concrete tokenizer for a mode.
"""

from typing import Any


def _pin_tf32() -> None:
    """Pin TF32 on so token output doesn't depend on the container's default.

    The NGC images enable TF32; stock PyTorch leaves matmul TF32 off. Measured
    on GH200 (profile/precision_parity.py, 2026-06-09): TF32 vs strict FP32 is
    3.6-4.3x encode throughput AND ~0.5% of output tokens differ — so an
    unpinned default silently controls both speed and token reproducibility.
    All shipped datasets were tokenized under TF32.
    """
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def create_tokenizer(
    mode: str,
    text_tokenizer_path: str,
    device: str = "cuda",
    *,
    min_pixels: int,
    max_pixels: int,
    **kwargs: Any,
):
    """Create the concrete EMU tokenizer for the requested tokenization mode."""
    _pin_tf32()
    if mode == "image_only":
        from .image_only import EMUImageOnlyTokenizer

        tokenizer_class = EMUImageOnlyTokenizer
    elif mode in ("image2text", "text2image"):
        from .image_text_pair import EMUImageTextPairTokenizer

        tokenizer_class = EMUImageTextPairTokenizer
    elif mode == "sft":
        from .sft import EMUSftTokenizer

        tokenizer_class = EMUSftTokenizer
    elif mode == "interleave":
        from .interleave import EMUInterleaveTokenizer

        tokenizer_class = EMUInterleaveTokenizer
    else:
        raise ValueError(
            f"Unknown tokenizer mode: {mode}. "
            "Must be one of: image_only, image2text, text2image, sft, interleave"
        )

    return tokenizer_class(
        text_tokenizer_path=text_tokenizer_path,
        device=device,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        mode=mode,
        **kwargs,
    )


__all__ = ["create_tokenizer"]
