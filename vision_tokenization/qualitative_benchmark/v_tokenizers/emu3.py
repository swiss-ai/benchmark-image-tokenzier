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
        self, min_pixels: int = 256 * 256, max_pixels: int = 512 * 512, device: str = "cuda", model_path: str = None
    ):
        """
        Initialize EMU3 vision tokenizer.

        Args:
            min_pixels: Minimum pixel count for aspect ratio
            max_pixels: Maximum pixel count for aspect ratio
            device: Device to load model on ('cuda' or 'cpu')
            model_path: Optional path to EMU3 model (uses default if None)
        """
        from Tokenizer.Emu3VisionTokenizer import Emu3VisionTokenizer as CoreEmu3Tokenizer

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.device = device

        # Initialize core EMU3 tokenizer
        self.tokenizer = CoreEmu3Tokenizer(min_pixels=min_pixels, max_pixels=max_pixels)

        # Move to specified device
        if torch.cuda.is_available() and device.startswith("cuda"):
            self.tokenizer.model = self.tokenizer.model.to(device)
            self.tokenizer.device = device

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

    def format_tokens_for_chat(
        self, indices: torch.Tensor, metadata: Dict[str, Any], special_tokens: Dict[str, int]
    ) -> str:
        """
        Format vision tokens as string for chat template insertion.

        Builds EMU3-specific token string with spatial structure:
        <|img_start|>H*W<|img_token_start|>
        <|visual token XXXXXX|>...<|img_end_of_row|>
        ...
        <|img_end_of_frame|><|img_end|>

        Args:
            indices: Discrete token indices [B, H, W] or [H, W]
            metadata: Dict with 'height' and 'width'
            special_tokens: Dict of special token IDs (not used, EMU3 uses string tokens)

        Returns:
            Formatted token string
        """
        h = metadata["height"]
        w = metadata["width"]

        # Flatten indices to 1D list
        if indices.ndim == 3:  # [B, H, W]
            visual_indices = indices[0].flatten().cpu().tolist()
        elif indices.ndim == 2:  # [H, W]
            visual_indices = indices.flatten().cpu().tolist()
        else:
            raise ValueError(f"Unexpected indices shape: {indices.shape}")

        # Build token string with EMU3 format
        img_tokens_str = f"<|img_start|>{h}*{w}<|img_token_start|>"

        # Add all image tokens row by row
        for row in range(h):
            row_start = row * w
            row_end = row_start + w
            row_tokens = visual_indices[row_start:row_end]

            for token_idx in row_tokens:
                img_tokens_str += f"<|visual token {token_idx:06d}|>"
            img_tokens_str += "<|img_end_of_row|>"

        # End image
        img_tokens_str += "<|img_end_of_frame|><|img_end|>"

        return img_tokens_str

    def get_resolution_params(self) -> Dict[str, Any]:
        """
        Get resolution parameters for this tokenizer.

        Returns:
            Dict with 'min_pixels' and 'max_pixels'
        """
        return {"min_pixels": self.min_pixels, "max_pixels": self.max_pixels}
