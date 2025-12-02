#!/usr/bin/env python3
"""EMU3.5 IBQ vision tokenizer wrapper for VLM benchmarking."""

import torch
from PIL import Image
from typing import Any, Dict, Tuple

from .base import SpatialTokenizer


class EMU35IBQVisionTokenizer(SpatialTokenizer):
    """
    EMU3.5 IBQ vision tokenizer for VLM inference.

    Wraps the Tokenizer/Emu3_5_IBQ.py implementation and provides
    the VLM-specific interface for encoding images and formatting tokens.

    EMU3.5 uses the same spatial 2D grid format and special tokens as EMU3:
    - <|img_start|>: Start of image
    - <|img_token_start|>: Start of vision tokens
    - <|visual token XXXXXX|>: Individual vision tokens (6-digit hex)
    - <|img_end_of_row|>: End of each token row
    - <|img_end_of_frame|>: End of frame
    - <|img_end|>: End of image

    Key differences from EMU3:
    - Uses information bottleneck quantization (IBQ)
    - Smart resize to 16x multiples
    - May have different compression ratios
    """

    def __init__(
        self,
        model_path: str,
        min_pixels: int = 256 * 256,
        max_pixels: int = 512 * 512,
        device: str = "cuda",
    ):
        """
        Initialize EMU3.5 IBQ vision tokenizer.

        Args:
            model_path: Path to EMU3.5 IBQ model (required)
            min_pixels: Minimum pixel count for aspect ratio
            max_pixels: Maximum pixel count for aspect ratio
            device: Device to load model on ('cuda' or 'cpu')
        """
        from Tokenizer.Emu3_5_IBQ import Emu3_5_IBQ

        self.model_path = model_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.device = device

        # Initialize core EMU3.5 IBQ tokenizer
        self.tokenizer = Emu3_5_IBQ(
            model_path=model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            device=device
        )

    @property
    def name(self) -> str:
        """Return tokenizer name."""
        return "EMU3.5-IBQ"

    def encode_for_vlm(self, image: Image.Image) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode image to discrete indices for VLM input.

        Args:
            image: PIL Image in RGB format

        Returns:
            indices: Discrete token indices [B, H, W] or [H, W]
            metadata: Dict with 'height', 'width', 'num_tokens'
        """
        # Preprocess image (includes smart resize)
        img_tensor = self.tokenizer.preprocess(image)
        img_tensor = img_tensor.to(self.tokenizer.device)

        # Encode to discrete indices
        with torch.no_grad():
            indices, additional_info = self.tokenizer.encode(img_tensor)

        # Extract spatial dimensions
        if indices.ndim == 3:  # [B, H, W]
            h, w = indices.shape[1], indices.shape[2]
        elif indices.ndim == 2:  # [H, W]
            h, w = indices.shape[0], indices.shape[1]
        else:
            raise ValueError(f"Unexpected indices shape: {indices.shape}")

        metadata = {
            'height': h,
            'width': w,
            'num_tokens': h * w,
            'latent_shape': additional_info.get('latent_shape')  # Keep for potential future use
        }

        return indices, metadata

    def format_tokens_for_chat(
        self,
        indices: torch.Tensor,
        metadata: Dict[str, Any],
        special_tokens: Dict[str, int]
    ) -> str:
        """
        Format vision tokens as string for chat template insertion.

        Uses the same format as EMU3:
        <|img_start|>H*W<|img_token_start|>
        <|visual token XXXXXX|>...<|img_end_of_row|>
        ...
        <|img_end_of_frame|><|img_end|>

        Args:
            indices: Discrete token indices [B, H, W] or [H, W]
            metadata: Dict with 'height' and 'width'
            special_tokens: Dict of special token IDs (not used, EMU3.5 uses string tokens)

        Returns:
            Formatted token string
        """
        h = metadata['height']
        w = metadata['width']

        # Flatten indices to 1D list
        if indices.ndim == 3:  # [B, H, W]
            visual_indices = indices[0].flatten().cpu().tolist()
        elif indices.ndim == 2:  # [H, W]
            visual_indices = indices.flatten().cpu().tolist()
        else:
            raise ValueError(f"Unexpected indices shape: {indices.shape}")

        # Build token string with EMU3 format (same as EMU3)
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
        return {
            'min_pixels': self.min_pixels,
            'max_pixels': self.max_pixels,
            'model_path': self.model_path
        }
