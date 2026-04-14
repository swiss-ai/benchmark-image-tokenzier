"""Lazy vision-tokenizer wrapper used together with the on-disk image token cache."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Callable, Dict, Tuple, Union

import torch
from PIL import Image
from transformers import AutoTokenizer

from .base import SpatialTokenizer, VLMVisionTokenizer

logger = logging.getLogger(__name__)


_DEFAULT_SPECIAL_TOKENS = {
    "boi_token": "<|img_start|>",
    "img_token": "<|img_token_start|>",
    "eol_token": "<|img_end_of_row|>",
    "eof_token": "<|img_end_of_frame|>",
    "eoi_token": "<|img_end|>",
}

_TOKENIZER_NAMES = {
    "emu3": "EMU3",
    "emu3.5": "EMU3.5-IBQ",
    "emu3.5-ibq": "EMU3.5-IBQ",
}


class LazyCachedVisionTokenizer(SpatialTokenizer):
    """Delay loading the heavy vision tokenizer until a cache miss requires encoding."""

    def __init__(
        self,
        tokenizer_type: str,
        tokenizer_factory: Callable[[], VLMVisionTokenizer],
        tokenizer_path: str,
        device: str = "cpu",
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        model_path: str | None = None,
    ):
        self.tokenizer_type = tokenizer_type.lower()
        self._tokenizer_factory = tokenizer_factory
        self._loaded_tokenizer: VLMVisionTokenizer | None = None
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.model_path = model_path
        self.tokenizer = SimpleNamespace(device=device)
        self._vision_token_range = None
        self._tokenizer_name = _TOKENIZER_NAMES.get(self.tokenizer_type, tokenizer_type)

        for attr, default in _DEFAULT_SPECIAL_TOKENS.items():
            setattr(self, attr, default)

        self._initialize_text_side_metadata()

    @property
    def name(self) -> str:
        if self._loaded_tokenizer is not None:
            return self._loaded_tokenizer.name
        return self._tokenizer_name

    def _initialize_text_side_metadata(self):
        if not self.tokenizer_path:
            logger.warning("No tokenizer_path provided, cached tokenizer will use default special tokens only")
            return

        txt_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        self._cache_special_tokens(txt_tokenizer)

        first_id, probe_token = self._get_first_visual_token_id(txt_tokenizer)
        if first_id is None:
            logger.warning(
                "Text tokenizer does not contain EMU-family vision tokens "
                f"({probe_token} mapped to unk). "
                "Vision token range will not be available."
            )
            return

        codebook_size = self._load_codebook_size_metadata_only()
        if codebook_size is None:
            logger.warning("Could not determine codebook size without loading the vision tokenizer")
            return

        from emu3_reconstruct_helper import VisionTokenRange

        self._vision_token_range = VisionTokenRange(first_id, codebook_size)

    def _load_codebook_size_metadata_only(self):
        try:
            if self.tokenizer_type == "emu3":
                from Tokenizer.Emu3VisionTokenizer import Emu3VisionTokenizer as CoreEmu3Tokenizer

                tokenizer = CoreEmu3Tokenizer(model_path=self.model_path or "BAAI/Emu3-VisionTokenizer", metadata_only=True)
                return tokenizer.codebook_size

            from Tokenizer.Emu3_5_IBQ import Emu3_5_IBQ
            from vision_tokenization.qualitative_benchmark.v_tokenizers.emu3_5_ibq import EMU35IBQVisionTokenizer

            tokenizer = Emu3_5_IBQ(
                model_path=self.model_path or EMU35IBQVisionTokenizer.DEFAULT_MODEL_PATH,
                metadata_only=True,
            )
            return tokenizer.codebook_size
        except Exception as exc:
            logger.warning(f"Metadata-only vision tokenizer init failed: {exc}")
            return None

    def _ensure_loaded(self) -> VLMVisionTokenizer:
        if self._loaded_tokenizer is None:
            print("Loading vision tokenizer on first cache miss...")
            self._loaded_tokenizer = self._tokenizer_factory()
            self.tokenizer = getattr(self._loaded_tokenizer, "tokenizer", self.tokenizer)
            self._tokenizer_name = self._loaded_tokenizer.name
        return self._loaded_tokenizer

    def encode_for_vlm(self, image: Image.Image) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self._ensure_loaded().encode_for_vlm(image)

    def get_resolution_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {"min_pixels": self.min_pixels, "max_pixels": self.max_pixels}
        if self.model_path is not None:
            params["model_path"] = self.model_path
        return params

    @property
    def vision_mapping(self) -> Union["VisionTokenRange", Dict[int, int]]:
        if self._loaded_tokenizer is not None:
            return self._loaded_tokenizer.vision_mapping
        if self._vision_token_range is not None:
            return self._vision_token_range
        return {}
