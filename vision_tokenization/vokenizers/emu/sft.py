#!/usr/bin/env python3
"""
EMU tokenizer for SFT (Supervised Fine-Tuning) data.
Handles conversations with single images and text.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import torch

from vision_tokenization.vokenizers.conversation_transforms import (
    BaseConversationTransform,
    ConversationTransformRegistry,
)

from .image_only import EMUImageOnlyTokenizer


def _replace_single_image(text_tokens: torch.Tensor, image_position: int, image_tokens: torch.Tensor) -> torch.Tensor:
    """
    Replace single <|image|> placeholder with vision tokens. # TODO: In future might want to support multiple images for multi-turn
    Optimized using torch.cat for better memory efficiency.

    Args:
        text_tokens: Original tokens with <|image|> placeholder
        image_position: Position of <|image|> token
        image_tokens: Tokenized image (without BOS/EOS)

    Returns:
        Final token tensor with image inserted
    """
    # Use torch.cat which is optimized at C++ level
    parts = []

    # Text before image
    if image_position > 0:
        parts.append(text_tokens[:image_position])

    # Image tokens
    parts.append(image_tokens)

    # Text after image placeholder
    if image_position + 1 < len(text_tokens):
        parts.append(text_tokens[image_position + 1 :])

    return torch.cat(parts, dim=0)


class EMUSftTokenizer(EMUImageOnlyTokenizer):
    """
    Tokenizer for SFT (Supervised Fine-Tuning) data with a single image and text.
    Optimized for single image per conversation (most common case).
    Replaces <|image|> placeholder with actual EMU vision tokens.

    Designed for SFT/pretraining with FineVision-style data where conversations
    already include assistant responses.
    """

    def __init__(self, *args, conversation_transform: Optional[str] = None, **kwargs):
        """
        Initialize SFT tokenizer with optional conversation transform.

        Args:
            *args: Positional arguments for parent class
            conversation_transform: Name of conversation transform to use (e.g., "llava_to_llama").
                                   If None, text is passed directly to tokenize_conversation (no transform).
            **kwargs: Keyword arguments for parent class
        """
        super().__init__(*args, **kwargs)

        # Initialize ThreadPoolExecutor for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="TokenizerPool")

        # Cache the image token ID for faster lookup
        self.image_token_id = self.text_tokenizer.convert_tokens_to_ids("<|image|>")

        # Configure conversation transform
        self.conversation_transform: Optional[BaseConversationTransform] = None
        if conversation_transform is not None:
            transform_cls = ConversationTransformRegistry.get_transform(conversation_transform)
            self.conversation_transform = transform_cls()

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
    def tokenize_conversation(self, messages: List[Dict[str, Any]], image: Any = None) -> torch.Tensor:
        """
        Tokenize ONE conversation. Potentially supports multiple images per conversation.
        Uses parallel processing: text on CPU, image on GPU.

        Args:
            messages: List of message dicts with role and content. Assumed to be in correct format for tokenizers chat template.
            image: Single PIL image corresponding to <|image|> placeholder

        Returns:
            Token tensor with <|image|> placeholder replaced by Emu3 vision tokens
        """
        # TODO: this check must be removed to support text-only sft
        if image is None:
            print("Warning: No image provided to tokenize_conversation")
            return torch.tensor([], dtype=torch.long)

        # Submit both tasks in parallel using common tokenization method
        text_future = self.executor.submit(self._tokenize_conversation_text_cpu, messages)
        image_future = self.executor.submit(lambda: self.tokenize_image(image)[1:-1])  # GPU thread

        text_tokens, num_images, image_positions = text_future.result()

        if num_images != 1:
            # Only single image samples are supported
            print(f"Warning: Found {num_images} image placeholders, expected 1. Skipping sample.")
            return torch.tensor([], dtype=torch.long)

        image_tokens = image_future.result()

        # Move text tokens to same device as image tokens for final assembly
        text_tokens = text_tokens.to(image_tokens.device)

        # Replace <|image|> placeholder with actual vision tokens
        # For single image case, use the first (and only) position
        final_tokens = _replace_single_image(text_tokens, image_positions[0], image_tokens)

        return final_tokens

    def tokenize(self, image, text) -> torch.Tensor:
        """
        Unified tokenization interface for SFT mode. One sample, non batched processing.

        Args:
            image: PIL Image (required)
            text: Dataset-specific conversation format.

        Returns:
            Tokenized tensor ready for model input
        """
        # Apply conversation transform if configured (goal: bring read text data from dataset in correct format for selected tokenizer chat template)
        if self.conversation_transform is not None:
            messages = self.conversation_transform.transform(text)
        else:
            messages = text

        return self.tokenize_conversation(messages, image)

    def tokenize_batch(self, images, resize_size, text=None):
        """
        Batched tokenization interface for SFT mode.
        THIS ONLY WORKS WITH ONE IMG PER CONVERSATION!

        NOTE: SFT has variable-length sequences due to different conversation lengths.
        We batch the image preprocessing for efficiency, then process each conversation
        with its corresponding image. Returns a list of variable-length token sequences.

        Args:
            images: List of PIL Images to tokenize (one per sample)
            resize_size: Target size for resizing images (batch wide size)
            text: List of conversation data (required for SFT mode)

        Returns:
            List of token tensors, one per image-conversation pair (variable lengths)
        """
        if text is None or len(text) == 0:
            raise ValueError("Text (conversations) is required for SFT tokenization")

        if len(images) != len(text):
            raise ValueError(f"Number of images ({len(images)}) must match number of conversations ({len(text)})")

        # Apply conversation transforms if configured
        if self.conversation_transform is not None:
            messages_batch = [self.conversation_transform.transform(t) for t in text]
        else:
            messages_batch = text

        def tokenize_conversation_texts_cpu():
            """CPU thread for batch text tokenization"""
            text_results = []
            for messages in messages_batch:
                result = self._tokenize_conversation_text_cpu(messages)
                text_results.append(result)
            return text_results

        # Img tokenization on GPU and text tokenization on CPU!
        image_future = self.executor.submit(self.tokenize_images, images, resize_size) # [B, num-tokens]
        text_future = self.executor.submit(tokenize_conversation_texts_cpu)

        image_tokens_batch = image_future.result()  # [B, image_seq_len] same seq len as img in one batch are resized to same size.
        text_results_batch = text_future.result()   # List of (text_tokens, num_images, image_position)

        # Combine each image-conversation pair
        combined_batch = []
        for i in range(len(images)):
            text_tokens, num_images, image_positions = text_results_batch[i]
            image_tokens = image_tokens_batch[i]  # Full image sequence with BOS/EOS

            assert num_images == 1, "Only single image samples are supported for batched SFT"

            # Get image tokens without BOS/EOS for replacement
            image_tokens_no_special = image_tokens[1:-1]

            # Move text tokens to same device as image tokens
            text_tokens = text_tokens.to(image_tokens.device)

            # Replace <|image|> placeholder with actual vision tokens
            # For single image case, use the first (and only) position
            final_tokens = _replace_single_image(text_tokens, image_positions[0], image_tokens_no_special)

            combined_batch.append(final_tokens)

        # Return list of variable-length sequences (not a stacked tensor)
        return combined_batch
