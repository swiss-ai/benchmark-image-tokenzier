#!/usr/bin/env python3
"""
EMU tokenizer for SFT (Supervised Fine-Tuning) data.
Handles conversations with single images and text.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from vision_tokenization.vokenizers.conversation_policy import (
    ConversationPolicy,
    apply_conversation_policy,
)

from .image_only import EMUImageOnlyTokenizer

logger = logging.getLogger(__name__)


def _replace_images(
    text_tokens: torch.Tensor, image_positions: List[int], image_tokens_list: List[torch.Tensor]
) -> torch.Tensor:
    """
    Replace multiple <|image|> placeholders with vision tokens in order.
    Optimized using torch.cat for better memory efficiency. Can potentially support multiple images per sample.
    Works on one sample only, not a list of samples.

    Args:
        text_tokens: Original tokens with <|image|> placeholders
        image_positions: List of positions of <|image|> tokens (must be sorted)
        image_tokens_list: List of tokenized images (without BOS/EOS), one per placeholder

    Returns:
        Final token tensor with all images inserted

    Raises:
        ValueError: If number of positions doesn't match number of image tokens
    """
    if len(image_positions) != len(image_tokens_list):
        raise ValueError(
            f"Number of image positions ({len(image_positions)}) must match "
            f"number of image tokens ({len(image_tokens_list)})"
        )

    # Handle empty case
    if len(image_positions) == 0:
        return text_tokens

    # Build parts list by iterating through positions
    parts = []
    last_pos = 0

    for i, pos in enumerate(image_positions):
        # Add text before this image (skip placeholder token at pos)
        if pos > last_pos:
            parts.append(text_tokens[last_pos:pos])

        # Add image tokens
        parts.append(image_tokens_list[i])

        # Update last_pos to skip the placeholder token
        last_pos = pos + 1

    # Add remaining text after last image
    if last_pos < len(text_tokens):
        parts.append(text_tokens[last_pos:])

    # Use torch.cat which is optimized at C++ level
    return torch.cat(parts, dim=0) if parts else torch.tensor([], dtype=text_tokens.dtype, device=text_tokens.device)


class EMUSftTokenizer(EMUImageOnlyTokenizer):
    """
    Tokenizer for SFT (Supervised Fine-Tuning) data.
    Supports non batched tokenization: supports >=1 images per conversation
    Supports batched tokenization: only supports 1 image per conversation
    """

    def __init__(self, *args, conversation_policy: Optional[ConversationPolicy] = None, **kwargs):
        """
        Initialize SFT tokenizer with optional conversation policy.

        Args:
            *args: Positional arguments for parent class
            conversation_policy: ConversationPolicy controlling message normalization.
            **kwargs: Keyword arguments for parent class
        """
        super().__init__(*args, **kwargs)

        # Initialize ThreadPoolExecutor for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="TokenizerPool")

        # Cache the image token ID for faster lookup
        self.image_token_id = self.text_tokenizer.convert_tokens_to_ids("<|image|>")

        # Configure conversation normalization policy
        self.conversation_policy = conversation_policy or ConversationPolicy()

    def _add_special_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Add BOS and EOS tokens if missing.

        This is necessary because:
        - Some chat templates hardcode BOS/EOS in the template (e.g., {{- bos_token }})
        - The add_special_tokens parameter in apply_chat_template is often ignored
        - Different templates have different behavior (some add BOS, some don't)
        - We need to verify and add only if missing to avoid double BOS/EOS tokens

        Uses efficient torch.cat approach (17% faster than F.pad).

        Args:
            tokens: 1D tensor of token IDs (non-empty from apply_chat_template)

        Returns:
            Token tensor with BOS at start and EOS at end
        """
        # Check what's missing (bos_id and eos_id guaranteed non-None by assertions)
        needs_bos = tokens[0] != self.bos_id
        needs_eos = tokens[-1] != self.eos_id

        # Fast path: nothing to do
        if not needs_bos and not needs_eos:
            return tokens

        # Build result with torch.cat
        parts = []
        if needs_bos:
            parts.append(torch.tensor([self.bos_id], dtype=tokens.dtype, device=tokens.device))
        parts.append(tokens)
        if needs_eos:
            parts.append(torch.tensor([self.eos_id], dtype=tokens.dtype, device=tokens.device))

        return torch.cat(parts)

    def _tokenize_conversation_text_cpu(self, messages: List[Dict[str, Any]]):
        """
        Common CPU thread for text tokenization of a single conversation.
        Supports multiple images per conversation.

        Args:
            messages: List of message dicts with role and content

        Returns:
            Tuple of (text_tokens, num_images, image_positions)
            - text_tokens: 1D tensor of token IDs with BOS/EOS
            - num_images: Number of <|image|> placeholders found
            - image_positions: List of positions where <|image|> tokens are located
        """
        with torch.cuda.device(-1):  # Use CPU
            # Apply chat template and tokenize in one step
            text_tokens = self.text_tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
            )
            text_tokens = text_tokens.squeeze(0)  # Remove batch dimension

            # Add BOS/EOS tokens if missing (handles different chat templates)
            text_tokens = self._add_special_tokens(text_tokens)

            # Find all image positions (support multiple images)
            image_mask = text_tokens == self.image_token_id
            num_images = image_mask.sum().item()

            # Get all image positions as a list
            if num_images > 0:
                image_positions = image_mask.nonzero(as_tuple=True)[0].tolist()
            else:
                image_positions = []

            return text_tokens, num_images, image_positions

    @torch.inference_mode()
    def tokenize_conversation(self, messages: List[Dict[str, Any]], images: List[Any]) -> torch.Tensor:
        """
        Tokenize ONE conversation with one or multiple images.
        Uses parallel processing: text on CPU, all images on GPU.

        Args:
            messages: List of message dicts with role and content. Assumed to be in correct format for tokenizers chat template.
            images: List of PIL images corresponding to <|image|> placeholders in order

        Returns:
            Token tensor with <|image|> placeholders replaced by Emu3 vision tokens

        Raises:
            ValueError: If images list is empty or number of images doesn't match number of placeholders
        """
        if not images:
            raise ValueError("images list cannot be empty")

        # Submit text tokenization on CPU
        text_future = self.executor.submit(self._tokenize_conversation_text_cpu, messages)

        # Submit all image tokenizations in parallel on GPU (strip BOS/EOS) -> Dont batch here as setup and VRAm and tokenizer not dynamically analyzed.
        image_futures = [self.executor.submit(lambda img=img: self.tokenize_image(img)[1:-1]) for img in images]

        # Get text tokenization results
        text_tokens, num_images, image_positions = text_future.result()

        # Validate image count matches placeholders
        if len(images) != num_images:
            raise ValueError(
                f"Number of images ({len(images)}) must match number of <|image|> placeholders ({num_images})"
            )

        # Collect all image tokens
        image_tokens_list = [future.result() for future in image_futures]

        # Move text tokens to same device as image tokens for final assembly
        if image_tokens_list:
            text_tokens = text_tokens.to(image_tokens_list[0].device)

        # Replace all <|image|> placeholders with actual vision tokens
        final_tokens = _replace_images(text_tokens, image_positions, image_tokens_list)

        return final_tokens

    def tokenize(self, images, text) -> torch.Tensor:
        """
        Unified tokenization interface for SFT mode. One sample, non batched processing.

        Args:
            images: List of PIL Images (required, one or more images)
            text: Dataset-specific conversation format.

        Returns:
            Tokenized tensor ready for model input

        Raises:
            ValueError: If images list is empty or number of images doesn't match placeholders
        """
        messages = apply_conversation_policy(text, self.conversation_policy)
        return self.tokenize_conversation(messages, images)

    def tokenize_batch(self, images, resize_size, text=None, group_slices=None):
        """
        Batched tokenization interface for SFT mode.

        Single-image is a special case of multi-image where every group has
        exactly one image.  Both paths share the same code: GPU image
        tokenization runs in parallel with CPU conversation tokenization.

        Args:
            images: List of PIL Images to tokenize.
            resize_size: Target size for resizing images (batch-wide).
            text: List of conversation data (required for SFT mode).
                One per image (single-image) or one per group (multi-image).
            group_slices: Optional ``(num_groups, 2)`` array mapping groups
                to positions in *images*.  When ``None``, each image is
                treated as its own group.

        Returns:
            List of token tensors (variable lengths).  ``None`` entries
            indicate skipped groups (placeholder / image count mismatch).
        """
        if text is None or len(text) == 0:
            raise ValueError("Text (conversations) is required for SFT tokenization")

        # Single-image is multi-image with trivial 1-image groups
        if group_slices is None:
            if len(images) != len(text):
                raise ValueError(
                    f"Number of images ({len(images)}) must match "
                    f"number of conversations ({len(text)})"
                )
            group_slices = np.array(
                [[i, i + 1] for i in range(len(images))], dtype=np.int64,
            )

        # GPU image tokenization ∥ CPU conversation tokenization
        def tokenize_all_texts_cpu():
            results = []
            for g_idx in range(len(group_slices)):
                messages = apply_conversation_policy(text[g_idx], self.conversation_policy)
                results.append(self._tokenize_conversation_text_cpu(messages))
            return results

        image_future = self.executor.submit(self.tokenize_images, images, resize_size)
        text_future = self.executor.submit(tokenize_all_texts_cpu)

        image_tokens_batch = image_future.result()  # [total_images, seq_len]
        text_results = text_future.result()

        image_tokens_batch = image_tokens_batch.cpu()

        # Per-group: replace <|image|> placeholders with vision tokens
        results = []
        for g_idx, (gs, ge) in enumerate(group_slices):
            gs, ge = int(gs), int(ge)
            text_tokens, num_images, image_positions = text_results[g_idx]

            num_group_images = ge - gs
            if num_images != num_group_images:
                logger.warning(
                    f"Group has {num_group_images} images but conversation has "
                    f"{num_images} <|image|> placeholders — skipping"
                )
                results.append(None)
                continue

            img_tokens_list = [image_tokens_batch[i, 1:-1] for i in range(gs, ge)]
            final_tokens = _replace_images(text_tokens, image_positions, img_tokens_list)
            results.append(final_tokens)

        return results
