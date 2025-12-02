#!/usr/bin/env python3
"""
Abstract base classes for vision v_tokenizers used in VLM benchmarking.

This module provides the interface for vision v_tokenizers that prepare image tokens
for VLM inference. It is separate from:
- Tokenizer/base.py: For reconstruction benchmarking
- vision_tokenization/vokenizers/base.py: For dataset tokenization

This focuses specifically on VLM inference needs: encoding images and formatting
tokens for insertion into chat templates.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import torch
from PIL import Image


class VLMVisionTokenizer(ABC):
    """
    Abstract interface for vision v_tokenizers used in VLM benchmarking.

    Vision v_tokenizers handle three key responsibilities:
    1. Encoding images to discrete token indices
    2. Formatting tokens as strings for chat template insertion
    3. Managing resolution parameters (min/max pixels)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name for the tokenizer.

        Returns:
            Name string (e.g., "EMU3", "EMU3.5-IBQ", "Cosmos")
        """
        pass

    @abstractmethod
    def encode_for_vlm(self, image: Image.Image) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode image to discrete indices for VLM input.

        This method handles preprocessing and encoding in one step, returning
        discrete token indices suitable for VLM inference.

        Args:
            image: PIL Image in RGB format

        Returns:
            indices: Discrete token indices (shape varies by tokenizer type)
                    - Spatial v_tokenizers: [B, H, W] or [H, W]
                    - Flattened v_tokenizers: [B, N] or [N]
            metadata: Dictionary with tokenizer-specific info:
                     - 'height', 'width': Spatial dimensions (for spatial v_tokenizers)
                     - 'num_tokens': Total token count
                     - Additional tokenizer-specific fields
        """
        pass

    @abstractmethod
    def format_tokens_for_chat(
        self,
        indices: torch.Tensor,
        metadata: Dict[str, Any],
        special_tokens: Dict[str, int]
    ) -> str:
        """
        Format vision tokens as string for insertion into chat template.

        This method converts discrete indices into a string representation that
        can be inserted into the VLM's chat template. The format is tokenizer-specific.

        Args:
            indices: Discrete token indices from encode_for_vlm()
            metadata: Metadata dict from encode_for_vlm()
            special_tokens: Dict mapping special token names to IDs (from inferencer)
                          e.g., {'img_start': 128256, 'img_end': 128257, ...}

        Returns:
            String representation of vision tokens, ready for chat template insertion.
            For EMU3: "<|img_start|>H*W<|img_token_start|><|visual token XXXXXX|>..."
        """
        pass

    @abstractmethod
    def get_resolution_params(self) -> Dict[str, Any]:
        """
        Get resolution parameters for this tokenizer.

        Returns:
            Dictionary with:
                - 'min_pixels': Minimum pixel count
                - 'max_pixels': Maximum pixel count
                - Additional tokenizer-specific params
        """
        pass


class SpatialTokenizer(VLMVisionTokenizer):
    """
    Base class for v_tokenizers with 2D spatial grids.

    These v_tokenizers preserve spatial structure, encoding images as 2D grids
    of tokens. Examples: EMU3, EMU3.5, Cosmos, OpenMAGViT2.

    Token format: [B, H, W] where H and W represent spatial dimensions.
    """
    pass


class FlattenedTokenizer(VLMVisionTokenizer):
    """
    Base class for v_tokenizers with 1D sequences.

    These v_tokenizers flatten spatial structure into 1D sequences of tokens.
    Examples: UniTok, LlamaGen, TokenFlow, VQGAN.

    Token format: [B, N] where N is the total number of tokens.
    """
    pass