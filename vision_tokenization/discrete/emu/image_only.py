#!/usr/bin/env python3
"""
EMU image-only tokenizer with core functionality.
Supports both Emu3 and Emu3.5 vision tokenizers.
"""

from pathlib import Path
import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from vision_tokenization.utils.json import json_load
from vision_tokenization.discrete.emu.token_layout import (
    STRUCTURE_TOKENS,
    resolve_token_ids,
    vision_band,
)

# Tokenizer imports require the repo root on PYTHONPATH (set by SLURM scripts)


class EMUImageOnlyTokenizer:
    """
    EMU tokenizer for image-only sequences.
    Provides direct image tokenization with EMU special tokens.
    Supports both Emu3 and Emu3.5 vision tokenizers.
    """

    def __init__(
        self,
        text_tokenizer_path: str,
        min_pixels: int,
        max_pixels: int,
        device: str = "cuda",
        max_encode_pixels: Optional[int] = 8_000_000,
        torch_compile: bool = False,
        torch_compile_mode: str = "reduce-overhead",
        vision_tokenizer_type: Optional[str] = None,
        vision_tokenizer_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize with text tokenizer that has EMU vision tokens and image tokenizer.

        Args:
            text_tokenizer_path: Path to text tokenizer with EMU tokens
            min_pixels: Minimum pixels for image preprocessing (required)
            max_pixels: Maximum pixels for image preprocessing (required)
            device: Device for image tokenizer (default: "cuda")
            vision_tokenizer_type: Which discrete vision tokenizer to use, "Emu3" or "Emu3.5".
                Falls back to the text tokenizer's ``vision_tokenizer`` config section,
                which only the Apertus 1.5 artifacts carry.
            vision_tokenizer_path: Weights path for that tokenizer, same fallback.
        """

        # Store device
        self.device = device
        self.torch_compile = torch_compile
        self.torch_compile_mode = torch_compile_mode

        self.text_tokenizer_load_time = 0.0
        self.model_load_time = 0.0

        # Load tokenizer with trust_remote_code for custom tokenizer class.
        # Use fast tokenizer for better performance.
        text_load_t0 = time.perf_counter()
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_tokenizer_path,
            trust_remote_code=True,
            use_fast=True,
        )
        self.text_tokenizer_load_time = time.perf_counter() - text_load_t0

        # min_pixels and max_pixels are required parameters
        assert min_pixels is not None, "min_pixels must be provided"
        assert max_pixels is not None, "max_pixels must be provided"

        # Load vision tokenizer config from tokenizer_config.json
        config_path = Path(text_tokenizer_path) / "tokenizer_config.json"
        tokenizer_config = json_load(config_path)

        vision_config = tokenizer_config.get("vision_tokenizer", {})
        vision_tokenizer_type = vision_tokenizer_type or vision_config.get("type")
        vision_tokenizer_path = vision_tokenizer_path or vision_config.get("path")

        missing = [name for name, value in
                   (("vision_tokenizer_type", vision_tokenizer_type),
                    ("vision_tokenizer_path", vision_tokenizer_path)) if not value]
        if missing:
            raise ValueError(
                f"{', '.join(missing)} not set. Pass it in the pipeline config, or use a "
                f"text tokenizer whose vision_tokenizer section supplies it ({config_path})."
            )

        print(f"Loading vision tokenizer: {vision_tokenizer_type} from {vision_tokenizer_path}")

        # Dynamically load the correct vision tokenizer class
        model_load_t0 = time.perf_counter()
        if vision_tokenizer_type == "Emu3":
            from Tokenizer.Emu3VisionTokenizer import Emu3VisionTokenizer

            self.image_tokenizer = Emu3VisionTokenizer(
                model_path=vision_tokenizer_path, device=self.device, min_pixels=min_pixels, max_pixels=max_pixels
            )
        elif vision_tokenizer_type == "Emu3.5":
            from Tokenizer.Emu3_5_IBQ import Emu3_5_IBQ

            self.image_tokenizer = Emu3_5_IBQ(
                model_path=vision_tokenizer_path,
                device=self.device,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                torch_compile=self.torch_compile,
                torch_compile_mode=self.torch_compile_mode,
            )
        else:
            raise ValueError(
                f"Unsupported vision tokenizer type: {vision_tokenizer_type}. " f"Supported types: Emu3, Emu3.5"
            )
        self.model_load_time = time.perf_counter() - model_load_t0

        # Pixel budget per encode call — controls GPU memory chunking.
        self.max_encode_pixels = max_encode_pixels

        # Cache for dimension tokens to avoid repeated encoding
        self.dim_cache = {}

        # Cache frequently used token IDs
        self._cache_special_tokens(tokenizer_config)

    def _cache_special_tokens(self, tokenizer_config: dict):
        """Cache special token IDs to avoid repeated lookups."""
        # Structure tokens
        assert self.text_tokenizer.bos_token is not None, "BOS token must be defined"
        assert self.text_tokenizer.eos_token is not None, "EOS token must be defined"

        self.bos_id = self.text_tokenizer.bos_token_id
        self.eos_id = self.text_tokenizer.eos_token_id

        resolved = resolve_token_ids(self.text_tokenizer, STRUCTURE_TOKENS)
        self.img_start_id = resolved["img_start"]
        self.img_end_id = resolved["img_end"]
        self.img_token_start_id = resolved["img_token_start"]
        self.eol_id = resolved["eol"]
        self.eof_id = resolved["eof"]

        self.vision_token_offset, _ = vision_band(tokenizer_config)

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

    def encapsulate_batch(self, image_indices: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Tokenize a batch of image indices, adding EMU3 structure tokens to each sequence.

        Args:
            image_indices: Tensor of image indices from vision tokenizer [B, H*W]
            height: Image height in tokens
            width: Image width in tokens

        Returns:
            Batched token IDs ready for model input. Shape: [B, Total_Token_Length]
        """

        # B = Batch Size, N = H*W (Number of vision tokens)
        batch_size, num_tokens_input = image_indices.shape
        num_tokens_needed = height * width

        assert (
            num_tokens_input == num_tokens_needed
        ), f"Dimension mismatch: {height}x{width} needs {num_tokens_needed} indices per image, got {num_tokens_input}"

        # 1. Calculate the length of the structural tokens (non-image tokens)
        # Total structural tokens per image (T_struct)
        dim_tokens = self._get_dim_tokens(height, width)

        T_struct = (
            1  # BOS
            + 1  # img_start
            + len(dim_tokens)  # dimension tokens
            + 1  # img_token_start
            + height  # EOL after each row
            + 1  # EOF
            + 1  # img_end
            + 1  # EOS
        )

        # 2. Calculate the total length of the final token sequence (T_total)
        # T_total = T_struct (all fixed tokens) + N (vision tokens)
        total_size = num_tokens_needed + T_struct

        # Pre-allocate output tensor for the entire batch: [B, Total_Token_Length]
        output = torch.empty((batch_size, total_size), dtype=torch.long, device=image_indices.device)

        # Convert image indices to HxW shape for EOL insertion and add offset
        # [B, H*W] -> [B, H, W]
        image_indices_2d = image_indices.view(batch_size, height, width)
        vision_tokens_with_offset = image_indices_2d + self.vision_token_offset

        # Create the vision part with EOL tokens for the whole batch
        # Shape: [B, H, W + 1] (W + 1 for the EOL token)
        vision_part_batched = torch.empty(
            (batch_size, height, width + 1), dtype=torch.long, device=image_indices.device
        )
        vision_part_batched[:, :, :width] = vision_tokens_with_offset
        # Add EOL token at the end of each row for all batches
        vision_part_batched[:, :, width] = self.eol_id

        # Flatten the vision part: [B, H, W + 1] -> [B, H * (W + 1)]
        vision_part_flat = vision_part_batched.flatten(start_dim=1)

        # --- Token Filling Logic (Vectorized) ---

        # 3. Create a template for the structural tokens
        # This must be done for all B rows simultaneously.

        # Create the structural prefix template once, then tile it.
        prefix_tokens = [
            self.bos_id,
            self.img_start_id,
            *dim_tokens,  # Insert dynamic dimension tokens
            self.img_token_start_id,
        ]
        prefix_tensor = torch.tensor(prefix_tokens, dtype=torch.long, device=image_indices.device)
        prefix_len = len(prefix_tokens)

        # Create the structural suffix template.
        suffix_tokens = [self.eof_id, self.img_end_id, self.eos_id]
        suffix_tensor = torch.tensor(suffix_tokens, dtype=torch.long, device=image_indices.device)

        # 4. Fill the final output tensor [B, T_total]

        # Vectorized assignment across the batch dimension (0)

        # Fill Prefix
        output[:, 0:prefix_len] = prefix_tensor.unsqueeze(0)  # [1, prefix_len] broadcasted to [B, prefix_len]

        # Fill Vision Tokens
        vision_start = prefix_len
        vision_end = vision_start + vision_part_flat.shape[1]
        output[:, vision_start:vision_end] = vision_part_flat  # [B, H*(W+1)]

        # Fill Suffix
        suffix_start = vision_end
        output[:, suffix_start:] = suffix_tensor.unsqueeze(0)  # [1, suffix_len] broadcasted to [B, suffix_len]

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

    @torch.inference_mode()
    def tokenize_images(self, images, resize_size: Tuple[int, int]) -> torch.Tensor:
        """
        Batched tokenization of images.
        As a batch is resized to have similar shape, output num tokens is equal.

        Args:
            images: List [PIL Image], or a CPU uint8 ``[B, H, W, C]`` tensor
                already preprocessed by ``preprocess_cpu`` in the prefetch
                workers (the resize to *resize_size* has then already happened).
            resize_size: Target size for resizing images

        Returns:
            Batch of encoded images: B x num_img_tokens (on CPU)
        """
        assert self.image_tokenizer is not None, "Image tokenizer required for processing images"

        # Compute images per chunk from pixel budget and resize dimensions.
        chunk_size = len(images)
        if self.max_encode_pixels is not None and resize_size is not None:
            h, w = resize_size
            pixels_per_image = max(1, h * w)
            chunk_size = max(1, self.max_encode_pixels // pixels_per_image)

        if chunk_size >= len(images):
            # Fast path: single chunk — preprocess + encode all at once.
            img_tensors = self.image_tokenizer.preprocess_batch(images, resize_size)
            indices, _ = self.image_tokenizer.encode(img_tensors)
            del img_tensors
        else:
            # Chunk before preprocess to bound peak GPU pixel memory.
            all_indices = []
            for i in range(0, len(images), chunk_size):
                img_tensors = self.image_tokenizer.preprocess_batch(
                    images[i : i + chunk_size], resize_size,
                )
                idx, _ = self.image_tokenizer.encode(img_tensors)
                all_indices.append(idx)
                del img_tensors
            indices = torch.cat(all_indices, dim=0)

        # Move indices to CPU — encapsulate is pure int ops, no model weights.
        batch_size, height, width = indices.shape
        image_indices = indices.flatten(start_dim=1).cpu()
        del indices
        return self.encapsulate_batch(image_indices, height, width)

    def tokenize_batch(self, images, resize_size, text=None, group_slices=None):
        """
        Batched tokenization interface for image-only mode.

        Args:
            images: List of PIL Images to tokenize (required)
            resize_size: Target size for resizing images
            text: Ignored for image-only tokenization
            group_slices: Optional ``(num_groups, 2)`` array mapping groups
                to positions in *images*.  When provided, returns one
                concatenated sequence per group instead of one per image.

        Returns:
            List of tokenized image tensors (one per image, or one per group)
        """
        batched_tokens = self.tokenize_images(images, resize_size)  # [B, seq_len]

        if group_slices is None:
            return [batched_tokens[i] for i in range(len(batched_tokens))]

        # Multi-image: concatenate per group (strip per-image BOS/EOS, wrap group)
        results = []
        batched_tokens_cpu = batched_tokens.cpu()
        for gs, ge in group_slices:
            gs, ge = int(gs), int(ge)
            group_img_tokens = batched_tokens_cpu[gs:ge]  # [num_imgs, seq_len]
            # Strip per-image BOS/EOS, keep inner structure
            img_structs = [group_img_tokens[i, 1:-1] for i in range(group_img_tokens.shape[0])]
            # Wrap with single BOS/EOS from first image
            bos = group_img_tokens[0, :1]
            eos = group_img_tokens[0, -1:]
            results.append(torch.cat([bos] + img_structs + [eos]))
        return results
