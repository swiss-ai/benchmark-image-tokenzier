#!/usr/bin/env python3
"""EMU3.5 IBQ vision tokenizer wrapper for VLM benchmarking."""

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from PIL import Image

from .base import SpatialTokenizer

# repo base
base_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(base_dir))
sys.path.append(str(base_dir / "Tokenizer"))


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

    # Default path to shared model weights on CSCS infrastructure
    DEFAULT_MODEL_PATH = "/capstor/store/cscs/swissai/infra01/MLLM/Emu3.5-VisionTokenizer"

    def __init__(
        self,
        model_path: str = None,  # Will use DEFAULT_MODEL_PATH if not provided
        min_pixels: int = 256 * 256,
        max_pixels: int = 512 * 512,
        device: str = "cuda",
        tokenizer_path: str = None,
    ):
        """
        Initialize EMU3.5 IBQ vision tokenizer.

        Args:
            model_path: Path to EMU3.5 IBQ model (required)
            min_pixels: Minimum pixel count for aspect ratio
            max_pixels: Maximum pixel count for aspect ratio
            device: Device to load model on ('cuda' or 'cpu')
            tokenizer_path: Path to text tokenizer (needed for vision token range detection)
        """
        from Tokenizer.Emu3_5_IBQ import Emu3_5_IBQ

        # Use default path if not provided
        if model_path is None:
            model_path = self.DEFAULT_MODEL_PATH
            print(f"Using default EMU3.5 model path: {model_path}")

        self.model_path = model_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.device = device
        self.tokenizer_path = tokenizer_path

        # Initialize core EMU3.5 IBQ tokenizer
        self.tokenizer = Emu3_5_IBQ(model_path=model_path, min_pixels=min_pixels, max_pixels=max_pixels, device=device)

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
            "height": h,
            "width": w,
            "num_tokens": h * w,
            "latent_shape": additional_info.get("latent_shape"),  # Keep for potential future use
        }

        return indices, metadata

    def get_resolution_params(self) -> Dict[str, Any]:
        """Get resolution parameters for this tokenizer."""
        return {"min_pixels": self.min_pixels, "max_pixels": self.max_pixels, "model_path": self.model_path}
