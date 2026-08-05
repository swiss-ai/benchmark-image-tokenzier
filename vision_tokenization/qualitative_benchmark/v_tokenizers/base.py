#!/usr/bin/env python3
"""
Abstract base classes for vision v_tokenizers used in VLM benchmarking.

This module provides the interface for vision v_tokenizers that prepare image tokens
for VLM inference. It is separate from:
- Tokenizer/base.py: For reconstruction benchmarking
- vision_tokenization/discrete/emu/: For dataset tokenization

This focuses specifically on VLM inference needs: encoding images and formatting
tokens for insertion into chat templates.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)


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
        self, indices: torch.Tensor, metadata: Dict[str, Any], special_tokens: Dict[str, int]
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

    def _cache_special_tokens(self, txt_tokenizer):
        """Cache special token strings from text tokenizer with fallbacks.

        Probes tokenizer attributes first (e.g. ``boi_token``), then falls
        back to hardcoded EMU3 defaults.  This mirrors the approach used
        by lmms-eval and makes the tokenizer work correctly even when
        special token strings differ across model families.
        """

        def _get_attr(tok, attr, default):
            try:
                val = getattr(tok, attr)
                if val is not None:
                    return val
            except AttributeError:
                pass
            return default

        self.boi_token = _get_attr(txt_tokenizer, "boi_token", "<|img_start|>")
        self.img_token = _get_attr(txt_tokenizer, "img_token", "<|img_token_start|>")
        self.eol_token = _get_attr(txt_tokenizer, "eol_token", "<|img_end_of_row|>")
        self.eof_token = _get_attr(txt_tokenizer, "eof_token", "<|img_end_of_frame|>")
        self.eoi_token = _get_attr(txt_tokenizer, "eoi_token", "<|img_end|>")
        self.visual_token_template = self._detect_visual_token_template(txt_tokenizer)

    def _detect_visual_token_template(self, txt_tokenizer) -> str:
        """Infer the text form used for visual tokens by probing the tokenizer.

        Every tokenizer that shipped a vision_token_mapping.json declared the format
        this probes for first, and Apertus 2 ships no such file.
        """
        probe_formats = [
            ("<|visual token 0|>", "<|visual token {token}|>"),
            ("<|visual token 000000|>", "<|visual token {token:06d}|>"),
        ]
        for probe_token, token_template in probe_formats:
            probe_id = txt_tokenizer.convert_tokens_to_ids(probe_token)
            if probe_id != txt_tokenizer.unk_token_id:
                return token_template

        logger.warning("Could not determine visual token text format; falling back to legacy zero-padded tokens")
        return "<|visual token {token:06d}|>"

    def _get_first_visual_token_id(self, txt_tokenizer):
        """Resolve the first visual token id using the detected token string format."""
        probe_token = self.visual_token_template.format(token=0)
        first_id = txt_tokenizer.convert_tokens_to_ids(probe_token)
        if first_id == txt_tokenizer.unk_token_id:
            return None, probe_token
        return first_id, probe_token

    def _format_image_tokens_rows(
        self, visual_indices: list, height: int, width: int, num_rows: int,
        include_end_tokens: bool = True,
    ) -> str:
        """Format vision token indices into the EMU-family string format.

        The output starts with ``<|img_start|>H*W<|img_token_start|>`` and
        then emits each row of ``<|visual token XXXXXX|>`` tokens separated
        by the end-of-row marker.  When ``include_end_tokens`` is True the
        string is terminated with end-of-frame + end-of-image markers
        (used for full-image prompts); otherwise it is left open so the
        model can continue generating (used for partial / completion prompts).
        """
        img_tokens_str = f"{self.boi_token}{height}*{width}{self.img_token}"
        for row in range(num_rows):
            row_start = row * width
            row_end = row_start + width
            for token_idx in visual_indices[row_start:row_end]:
                img_tokens_str += self.visual_token_template.format(token=int(token_idx))
            img_tokens_str += self.eol_token
        if include_end_tokens:
            img_tokens_str += f"{self.eof_token}{self.eoi_token}"
        return img_tokens_str

    def format_tokens_for_chat(
        self, indices: torch.Tensor, metadata: Dict[str, Any],
        special_tokens: Dict[str, int],
    ) -> str:
        """Format vision tokens as a full-image string for chat template insertion."""
        h = metadata["height"]
        w = metadata["width"]
        if indices.ndim == 3:
            visual_indices = indices[0].flatten().cpu().tolist()
        elif indices.ndim == 2:
            visual_indices = indices.flatten().cpu().tolist()
        else:
            raise ValueError(f"Unexpected indices shape: {indices.shape}")
        return self._format_image_tokens_rows(visual_indices, h, w, h, include_end_tokens=True)

    def create_partial_prompt(
        self, visual_indices: list, height: int, width: int, given_rows: int,
    ) -> str:
        """Emit the first ``given_rows`` rows with no end markers — for image completion."""
        return self._format_image_tokens_rows(
            visual_indices, height, width, given_rows, include_end_tokens=False,
        )

    def _detect_vision_token_range(self):
        """Resolve the vision token ID range from ``self.tokenizer_path``.

        Requires ``self.tokenizer_path`` and ``self.tokenizer.codebook_size``
        to be set by the subclass ``__init__``.  Caches special token strings
        from the text tokenizer as a side effect.  Returns ``None`` if the
        tokenizer path is missing or the text tokenizer doesn't know the
        vision vocabulary.
        """
        from transformers import AutoTokenizer
        from emu3_reconstruct_helper import VisionTokenRange

        if not self.tokenizer_path:
            logger.warning("No tokenizer_path provided, vision token range will not be available")
            return None

        txt_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        self._cache_special_tokens(txt_tokenizer)

        first_id, probe_token = self._get_first_visual_token_id(txt_tokenizer)
        if first_id is None:
            logger.warning(
                "Text tokenizer does not contain EMU-family vision tokens "
                f"({probe_token} mapped to unk). "
                "Vision token range will not be available."
            )
            return None

        codebook_size = self.tokenizer.codebook_size
        logger.info(f"Detected vision token range: first_id={first_id}, codebook_size={codebook_size}")
        return VisionTokenRange(first_id, codebook_size)

    @property
    def vision_mapping(self) -> Union["VisionTokenRange", Dict[int, int]]:
        """``VisionTokenRange`` when detection succeeded, empty dict otherwise."""
        return self._vision_token_range if self._vision_token_range is not None else {}


class FlattenedTokenizer(VLMVisionTokenizer):
    """
    Base class for v_tokenizers with 1D sequences.

    These v_tokenizers flatten spatial structure into 1D sequences of tokens.
    Examples: UniTok, LlamaGen, TokenFlow, VQGAN.

    Token format: [B, N] where N is the total number of tokens.
    """

    pass
