#!/usr/bin/env python3
"""
EMU tokenizer for SFT (Supervised Fine-Tuning) data.
Handles conversations with single images and text.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import torch

from vision_tokenization.vokenizers.conversation_transforms import BaseConversationTransform, ConversationTransformRegistry
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

    @torch.inference_mode()
    def tokenize_conversation(self, messages: List[Dict[str, Any]], image: Any = None) -> torch.Tensor:
        """
        Tokenize a conversation with a single image and text.
        Uses parallel processing: text on CPU, image on GPU.

        Args:
            messages: List of message dicts with role and content. Assumed to be in correct format for tokenizers chat template.
            image: Single PIL image corresponding to <|image|> placeholder

        Returns:
            Token tensor with <|image|> placeholder replaced by Emu3 vision tokens
        """

        def tokenize_text_cpu():
            """CPU thread for text tokenization."""
            # Force text operations to CPU
            with torch.cuda.device(-1):  # Use CPU
                # Apply chat template and tokenize in one step
                text_tokens = self.text_tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
                )
                text_tokens = text_tokens.squeeze(0)  # Remove batch dimension

                # Add BOS/EOS tokens if missing (handles different chat templates)
                text_tokens = self._add_special_tokens(text_tokens)

                # Step 3: Find image position
                image_mask = text_tokens == self.image_token_id
                num_images = image_mask.sum().item()

                if num_images == 1:
                    image_position = image_mask.nonzero(as_tuple=True)[0][0].item()
                else:
                    image_position = None

                return text_tokens, num_images, image_position

        # Check if image exists before starting parallel processing
        if image is None:
            print("Warning: No image provided to tokenize_conversation")
            return torch.tensor([], dtype=torch.long)

        # Submit both tasks in parallel
        text_future = self.executor.submit(tokenize_text_cpu)
        image_future = self.executor.submit(lambda: self.tokenize_image(image)[1:-1])  # GPU thread

        # Wait for both results
        text_tokens, num_images, image_position = text_future.result()

        if num_images != 1:
            # Only single image samples are supported
            print(f"Warning: Found {num_images} image placeholders, expected 1. Skipping sample.")
            return torch.tensor([], dtype=torch.long)

        image_tokens = image_future.result()

        # Move text tokens to same device as image tokens for final assembly
        text_tokens = text_tokens.to(image_tokens.device)

        # Replace <|image|> placeholder with actual vision tokens
        final_tokens = _replace_single_image(text_tokens, image_position, image_tokens)

        return final_tokens

    def tokenize(self, image, text) -> torch.Tensor:
        """
        Unified tokenization interface for SFT mode.

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

        # Tokenize with tokenize_conversation
        return self.tokenize_conversation(messages, image)
