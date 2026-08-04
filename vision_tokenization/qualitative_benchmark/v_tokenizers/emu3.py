#!/usr/bin/env python3
"""EMU3 vision tokenizer wrapper for VLM benchmarking."""

from typing import Any, Dict, Tuple

import torch
from PIL import Image

from .base import SpatialTokenizer


class EMU3VisionTokenizer(SpatialTokenizer):
    """
    EMU3 vision tokenizer for VLM inference.

    Wraps the Tokenizer/Emu3VisionTokenizer.py implementation and provides
    the VLM-specific interface for encoding images and formatting tokens.

    EMU3 uses a spatial 2D grid format with special tokens:
    - <|img_start|>: Start of image
    - <|img_token_start|>: Start of vision tokens
    - <|visual token XXXXXX|>: Individual vision tokens (6-digit hex)
    - <|img_end_of_row|>: End of each token row
    - <|img_end_of_frame|>: End of frame
    - <|img_end|>: End of image
    """

    def __init__(
        self,
        min_pixels: int = 256 * 256,
        max_pixels: int = 512 * 512,
        device: str = "cuda",
        model_path: str = None,
        tokenizer_path: str = None,
    ):
        """
        Initialize EMU3 vision tokenizer.

        Args:
            min_pixels: Minimum pixel count for aspect ratio
            max_pixels: Maximum pixel count for aspect ratio
            device: Device to load model on ('cuda' or 'cpu')
            model_path: Optional path to EMU3 model (uses default if None)
            tokenizer_path: Path to text tokenizer (needed for vision token range detection)
        """
        from Tokenizer.Emu3VisionTokenizer import Emu3VisionTokenizer as CoreEmu3Tokenizer

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.device = device
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        # Initialize core EMU3 tokenizer
        core_kwargs = {"min_pixels": min_pixels, "max_pixels": max_pixels}
        if model_path is not None:
            core_kwargs["model_path"] = model_path
        self.tokenizer = CoreEmu3Tokenizer(**core_kwargs)

        # Move to specified device
        if torch.cuda.is_available() and device.startswith("cuda"):
            self.tokenizer.model = self.tokenizer.model.to(device)
            self.tokenizer.device = device

        # Set default special token strings (overridden by _cache_special_tokens if tokenizer_path is available)
        self.boi_token = "<|img_start|>"
        self.img_token = "<|img_token_start|>"
        self.eol_token = "<|img_end_of_row|>"
        self.eof_token = "<|img_end_of_frame|>"
        self.eoi_token = "<|img_end|>"

        # Detect vision token range and cache special tokens from text tokenizer
        self._vision_token_range = self._detect_vision_token_range()

    @property
    def name(self) -> str:
        """Return tokenizer name."""
        return "EMU3"

    def encode_for_vlm(self, image: Image.Image) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode image to discrete indices for VLM input.

        Args:
            image: PIL Image in RGB format

        Returns:
            indices: Discrete token indices [B, H, W] or [H, W]
            metadata: Dict with 'height', 'width', 'num_tokens'
        """
        # Preprocess image
        img_tensor = self.tokenizer.preprocess(image)
        img_tensor = img_tensor.to(self.tokenizer.device)

        # Encode to discrete indices
        with torch.no_grad():
            indices, _ = self.tokenizer.encode(img_tensor)

        # Extract spatial dimensions
        if indices.ndim == 3:  # [B, H, W]
            h, w = indices.shape[1], indices.shape[2]
        elif indices.ndim == 2:  # [H, W]
            h, w = indices.shape[0], indices.shape[1]
        else:
            raise ValueError(f"Unexpected indices shape: {indices.shape}")

        metadata = {"height": h, "width": w, "num_tokens": h * w}

        return indices, metadata

    def get_resolution_params(self) -> Dict[str, Any]:
        """Get resolution parameters for this tokenizer."""
        params = {"min_pixels": self.min_pixels, "max_pixels": self.max_pixels}
        if self.model_path is not None:
            params["model_path"] = self.model_path
        return params
