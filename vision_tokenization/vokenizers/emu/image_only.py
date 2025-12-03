#!/usr/bin/env python3
"""
EMU image-only tokenizer with core functionality.
Supports both Emu3 and Emu3.5 vision tokenizers.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

# Add paths for Tokenizer imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Tokenizer"))

from ..base import BaseTokenizer


class EMUImageOnlyTokenizer(BaseTokenizer):
    """
    EMU tokenizer for image-only sequences.
    Provides direct image tokenization with EMU special tokens.
    Supports both Emu3 and Emu3.5 vision tokenizers.
    """

    def __init__(self, text_tokenizer_path: str, min_pixels: int, max_pixels: int, device: str = "cuda", **kwargs):
        """
        Initialize with text tokenizer that has EMU vision tokens and image tokenizer.

        Args:
            text_tokenizer_path: Path to text tokenizer with EMU tokens
            min_pixels: Minimum pixels for image preprocessing (required)
            max_pixels: Maximum pixels for image preprocessing (required)
            device: Device for image tokenizer (default: "cuda")
        """

        # Store device
        self.device = device

        # Load tokenizer with trust_remote_code for custom tokenizer class
        # Use fast tokenizer for better performance
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path, trust_remote_code=True, use_fast=True)

        # min_pixels and max_pixels are required parameters
        assert min_pixels is not None, "min_pixels must be provided"
        assert max_pixels is not None, "max_pixels must be provided"

        # Load vision tokenizer config from tokenizer_config.json
        config_path = Path(text_tokenizer_path) / "tokenizer_config.json"
        with open(config_path, "r") as f:
            tokenizer_config = json.load(f)

        if "vision_tokenizer" not in tokenizer_config:
            raise ValueError(
                f"No vision_tokenizer config found in {config_path}. "
                f"Make sure the omni-tokenizer was created with vision tokenizer info."
            )

        vision_config = tokenizer_config["vision_tokenizer"]
        vision_tokenizer_type = vision_config["type"]
        vision_tokenizer_path = vision_config["path"]

        print(f"Loading vision tokenizer: {vision_tokenizer_type} from {vision_tokenizer_path}")

        # Dynamically load the correct vision tokenizer class
        if vision_tokenizer_type == "Emu3":
            from Tokenizer.Emu3VisionTokenizer import Emu3VisionTokenizer

            self.image_tokenizer = Emu3VisionTokenizer(
                model_path=vision_tokenizer_path, device=self.device, min_pixels=min_pixels, max_pixels=max_pixels
            )
        elif vision_tokenizer_type == "Emu3.5":
            from Tokenizer.Emu3_5_IBQ import Emu3_5_IBQ

            self.image_tokenizer = Emu3_5_IBQ(
                model_path=vision_tokenizer_path, device=self.device, min_pixels=min_pixels, max_pixels=max_pixels
            )
        else:
            raise ValueError(
                f"Unsupported vision tokenizer type: {vision_tokenizer_type}. " f"Supported types: Emu3, Emu3.5"
            )

        # Cache for dimension tokens to avoid repeated encoding
        self.dim_cache = {}

        # Cache frequently used token IDs
        self._cache_special_tokens()

    def _cache_special_tokens(self):
        """Cache special token IDs to avoid repeated lookups."""
        # Structure tokens
        assert self.text_tokenizer.bos_token is not None, "BOS token must be defined"
        assert self.text_tokenizer.eos_token is not None, "EOS token must be defined"

        self.bos_id = self.text_tokenizer.bos_token_id
        self.eos_id = self.text_tokenizer.eos_token_id

        # EMU3 special tokens
        self.img_start_id = self.text_tokenizer.convert_tokens_to_ids("<|img_start|>")
        self.img_end_id = self.text_tokenizer.convert_tokens_to_ids("<|img_end|>")
        self.img_token_start_id = self.text_tokenizer.convert_tokens_to_ids("<|img_token_start|>")
        self.eol_id = self.text_tokenizer.convert_tokens_to_ids("<|img_end_of_row|>")
        self.eof_id = self.text_tokenizer.convert_tokens_to_ids("<|img_end_of_frame|>")

        # Compute the actual offset for vision tokens
        # Vision tokens are "<|visual token 000000|>" through "<|visual token XXXXXX|>"
        first_vision_token = self.text_tokenizer.convert_tokens_to_ids("<|visual token 000000|>")
        self.vision_token_offset = first_vision_token

    def _get_dim_tokens(self, height: int, width: int) -> List[int]:
        """
        Get dimension tokens with caching to avoid repeated encoding.

        Args:
            height: Image height in tokens
            width: Image width in tokens

        Returns:
            List of token IDs for the dimension string
        """
        dim_key = f"{height}*{width}"
        if dim_key not in self.dim_cache:
            # Encode and cache the dimension tokens
            self.dim_cache[dim_key] = self.text_tokenizer.encode(dim_key, add_special_tokens=False)
        return self.dim_cache[dim_key]

    def encapsulate_image(self, image_indices: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Directly tokenize image-only data without intermediate text conversion.

        Args:
            image_indices: Tensor of image indices from vision tokenizer [H*W]
            height: Image height in tokens
            width: Image width in tokens

        Returns:
            Token IDs ready for model input
        """
        num_tokens_needed = height * width
        assert (
            image_indices.numel() == num_tokens_needed
        ), f"Dimension mismatch: {height}x{width} needs {num_tokens_needed} indices, got {image_indices.numel()}"

        # Pre-allocate output tensor for efficiency
        # Structure: BOS + img_start + dims(~3) + img_token_start + vision_tokens + EOLs + EOF + img_end + EOS
        # Use cached dimension tokens to avoid repeated encoding
        dim_tokens = self._get_dim_tokens(height, width)

        # Calculate total size
        total_size = (
            1  # BOS
            + 1  # img_start
            + len(dim_tokens)  # dimension tokens
            + 1  # img_token_start
            + num_tokens_needed  # vision tokens
            + height  # EOL after each row
            + 1  # EOF
            + 1  # img_end
            + 1  # EOS
        )

        # Pre-allocate the entire output tensor
        output = torch.empty(total_size, dtype=torch.long)

        # Fill in the tokens using slicing (no Python list operations)
        idx = 0

        # Fixed tokens at the beginning
        output[idx] = self.bos_id
        output[idx + 1] = self.img_start_id
        idx += 2

        # Dimension tokens
        output[idx : idx + len(dim_tokens)] = torch.tensor(dim_tokens, dtype=torch.long)
        idx += len(dim_tokens)

        output[idx] = self.img_token_start_id
        idx += 1

        # Vision tokens with EOL markers - fully vectorized
        image_indices = image_indices.view(height, width)
        vision_tokens_with_offset = image_indices + self.vision_token_offset

        # Create vision part with EOL tokens in one operation
        vision_part = torch.empty((height, width + 1), dtype=torch.long)
        vision_part[:, :width] = vision_tokens_with_offset
        vision_part[:, -1] = self.eol_id

        # Copy all rows at once
        output[idx : idx + height * (width + 1)] = vision_part.flatten()
        idx += height * (width + 1)

        # Final tokens
        output[idx] = self.eof_id
        output[idx + 1] = self.img_end_id
        output[idx + 2] = self.eos_id

        return output

    @torch.inference_mode()
    def tokenize_image(self, image) -> torch.Tensor:
        """
        Complete pipeline: PIL image → vision indices → EMU3 encapsulated tokens.

        Args:
            image: PIL Image

        Returns:
            Token sequence with EMU3 structure tokens (BOS, img_start, dims, EOL, EOS, etc.)
        """
        assert self.image_tokenizer is not None, "Image tokenizer required for processing images"

        # Step 1: Preprocess image (PIL → tensor)
        img_tensor = self.image_tokenizer.preprocess(image)

        # Step 2: Encode to vision indices
        indices, _ = self.image_tokenizer.encode(img_tensor)

        # Step 3: Get dimensions and flatten
        # [1, H, W] → [H, W] → [H*W]
        indices_2d = indices.squeeze(0)
        height, width = indices_2d.shape
        image_indices = indices_2d.flatten()

        # Step 4: Encapsulate with EMU3 structure tokens
        return self.encapsulate_image(image_indices, height, width)

    def translate_image_to_text(self, image) -> str:
        """
        Translate a PIL image to EMU3 text representation with special tokens.

        Args:
            image: PIL Image

        Returns:
            Text string with EMU3 special tokens like:
            '<|img_start|>32*32<|img_token_start|><|visual token 000000|>...<|img_end|>'
        """
        # First tokenize the image to get token IDs
        token_ids = self.tokenize_image(image)
        token_ids_no_eos_bos = token_ids[1:-1]  # Remove BOS and EOS for text conversion
        # Decode to text using the text tokenizer
        text = self.text_tokenizer.decode(token_ids_no_eos_bos, skip_special_tokens=False)

        return text

    def tokenize(self, image, text=None) -> torch.Tensor:
        """
        Unified tokenization interface for image-only mode.

        Args:
            image: PIL Image to tokenize (required)
            text: Ignored for image-only tokenization

        Returns:
            Tokenized image as tensor
        """
        # Image is required for image-only tokenizer
        # Ignore text parameter, only process image
        return self.tokenize_image(image)
