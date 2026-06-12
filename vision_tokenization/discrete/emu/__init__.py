"""EMU tokenizer factory.

The package init stays lazy on purpose: importing this module should not load
all tokenizer implementations or their optional dependencies. Call
``create_tokenizer()`` to instantiate the concrete tokenizer for a mode.
"""

from typing import Any


def create_tokenizer(
    mode: str,
    text_tokenizer_path: str,
    device: str = "cuda",
    *,
    min_pixels: int,
    max_pixels: int,
    **kwargs: Any,
):
    """Create the concrete EMU tokenizer for the requested tokenization mode.

    TF32 pinning lives in the vision-wrapper constructors (Emu3_5_IBQ,
    Emu3VisionTokenizer) — the layer every consumer shares.
    """
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
    elif mode == "posttraining":
        # Posttraining freezes media as raw image blocks (no paired text
        # render); the image-only tokenizer is exactly the encode + encapsulate
        # path it needs. It swallows the unused ``mode`` kwarg via ``**kwargs``.
        from .image_only import EMUImageOnlyTokenizer

        tokenizer_class = EMUImageOnlyTokenizer
    else:
        raise ValueError(
            f"Unknown tokenizer mode: {mode}. "
            "Must be one of: image_only, image2text, text2image, sft, interleave, posttraining"
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
