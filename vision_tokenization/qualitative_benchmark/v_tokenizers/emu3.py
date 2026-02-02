#!/usr/bin/env python3
"""EMU3 vision tokenizer wrapper for VLM benchmarking."""

import logging
from typing import Any, Dict, Tuple, Union

import torch
from PIL import Image
from transformers import AutoTokenizer

from .base import SpatialTokenizer

logger = logging.getLogger(__name__)


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
        self.tokenizer_path = tokenizer_path

        # Initialize core EMU3 tokenizer
        self.tokenizer = CoreEmu3Tokenizer(min_pixels=min_pixels, max_pixels=max_pixels)

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

    def _format_image_tokens_rows(
        self, visual_indices: list, height: int, width: int, num_rows: int, include_end_tokens: bool = True
    ) -> str:
        """
        Helper method to format image tokens into EMU3 string format.

        Args:
            visual_indices: List of visual token indices
            height: Total image height in token rows
            width: Width in tokens per row
            num_rows: Number of rows to include in output
            include_end_tokens: If True, add EOF + EOI at the end

        Returns:
            Formatted token string
        """
        img_tokens_str = f"{self.boi_token}{height}*{width}{self.img_token}"

        for row in range(num_rows):
            row_start = row * width
            row_end = row_start + width
            row_tokens = visual_indices[row_start:row_end]

            for token_idx in row_tokens:
                img_tokens_str += f"<|visual token {token_idx:06d}|>"
            img_tokens_str += self.eol_token

        if include_end_tokens:
            img_tokens_str += f"{self.eof_token}{self.eoi_token}"

        return img_tokens_str

    def format_tokens_for_chat(
        self, indices: torch.Tensor, metadata: Dict[str, Any], special_tokens: Dict[str, int]
    ) -> str:
        """
        Format vision tokens as string for chat template insertion.

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

        return self._format_image_tokens_rows(visual_indices, h, w, h, include_end_tokens=True)

    def get_resolution_params(self) -> Dict[str, Any]:
        """Get resolution parameters for this tokenizer."""
        return {"min_pixels": self.min_pixels, "max_pixels": self.max_pixels}

    def create_partial_prompt(self, visual_indices: list, height: int, width: int, given_rows: int) -> str:
        """
        Create a prompt with partial image tokens (for image completion tasks).

        Args:
            visual_indices: Full list of visual token indices
            height: Total height in token rows
            width: Width in tokens per row
            given_rows: Number of rows to include in prompt (rest will be generated)

        Returns:
            Formatted prompt string with partial image tokens
        """
        return self._format_image_tokens_rows(visual_indices, height, width, given_rows, include_end_tokens=False)

    def _detect_vision_token_range(self):
        """Detect vision token range from the text tokenizer.

        Returns a ``VisionTokenRange`` if a text tokenizer path is available,
        otherwise ``None``.
        """
        from emu3_reconstruct_helper import VisionTokenRange

        if not self.tokenizer_path:
            logger.warning("No tokenizer_path provided, vision token range will not be available")
            return None

        txt_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)

        # Cache special tokens using the base class helper
        self._cache_special_tokens(txt_tokenizer)

        first_id = txt_tokenizer.convert_tokens_to_ids("<|visual token 000000|>")
        if first_id == txt_tokenizer.unk_token_id:
            logger.warning(
                "Text tokenizer does not contain EMU3 vision tokens "
                "(<|visual token 000000|> mapped to unk). "
                "Vision token range will not be available."
            )
            return None

        codebook_size = self.tokenizer.codebook_size
        logger.info(
            f"Detected vision token range: first_id={first_id}, codebook_size={codebook_size}"
        )
        return VisionTokenRange(first_id, codebook_size)

    @property
    def vision_mapping(self) -> Union["VisionTokenRange", Dict[int, int]]:
        """
        Get the vision token mapping.

        Returns a ``VisionTokenRange`` for range-based lookups (preferred),
        or an empty dict if detection failed.
        """
        if self._vision_token_range is not None:
            return self._vision_token_range
        return {}
