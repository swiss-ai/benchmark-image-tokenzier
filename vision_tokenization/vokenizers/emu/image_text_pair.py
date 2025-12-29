#!/usr/bin/env python3
"""
EMU tokenizer for image-text pairs with parallel GPU/CPU processing.
"""

from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image

from .image_only import EMUImageOnlyTokenizer


class EMUImageTextPairTokenizer(EMUImageOnlyTokenizer):
    """
    Extended tokenizer for image-text pairs with parallel GPU/CPU processing.
    Image tokenization happens on GPU while text tokenization happens on CPU in parallel.
    """

    def __init__(self, *args, mode=None, **kwargs):
        """Initialize with same parameters as parent class."""
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="TokenizerPool")

    def tokenize_image_text_pair(
        self,
        image,
        text: str,
    ) -> torch.Tensor:
        """
        Tokenize an image-text pair with parallel processing using ThreadPoolExecutor.
        Image is processed on GPU while text is processed on CPU simultaneously.

        Args:
            image: PIL Image to tokenize
            text: Text string to append after image

        Returns:
            Combined tokens: [BOS] + [image tokens without EOS] + [text tokens] + [EOS]
        """

        def tokenize_text_cpu():
            """CPU thread for text tokenization."""
            # Force text tokenization to CPU
            with torch.cuda.device(-1):  # Use CPU
                text_tokens_dict = self.text_tokenizer(
                    text, truncation=False, add_special_tokens=False, return_tensors="pt"
                )
                return text_tokens_dict["input_ids"].squeeze(0)

        # Submit both tasks to executor
        # Image on GPU (usually the bottleneck)
        image_future = self.executor.submit(self.tokenize_image, image)

        # Text on CPU (fast, runs in parallel)
        text_future = self.executor.submit(tokenize_text_cpu)

        # Wait for both and get results
        image_tokens = image_future.result()  # img will be encapsulated with bos and eos tokens!
        text_tokens = text_future.result()

        # Move text tokens to same device as image tokens for concatenation
        text_tokens = text_tokens.to(image_tokens.device)

        # Combine based on mode
        if self.mode == "text2image":
            # Text first, then image
            combined_tokens = torch.cat(
                [
                    image_tokens[:1],  # BOS token
                    text_tokens,  # Text tokens
                    image_tokens[1:],  # Image tokens (including EOS)
                ]
            )
        elif self.mode == "image2text":
            # Image first, then text
            combined_tokens = torch.cat(
                [
                    image_tokens[:-1],  # Image tokens without EOS
                    text_tokens,  # Text tokens
                    image_tokens[-1:],  # EOS token
                ]
            )
        else:
            # TODO: Might want to support random order as well.
            raise ValueError(f"Invalid mode for image_text_pair tokenizer: {self.mode}")

        return combined_tokens

    def tokenize(self, image: Image, text: str) -> torch.Tensor:
        """
        Unified tokenization interface for image-text pair mode.

        Args:
            image: PIL Image to tokenize (required)
            text: Text string to append after image (required)

        Returns:
            Combined tokenized output as tensor
        """
        # Both image and text are required for image-text pair tokenizer
        return self.tokenize_image_text_pair(image, text)

    def tokenize_images(self, images, resize_size, text=None):
        """
        Batched tokenization interface for image-text pair mode.

        NOTE: Image-text pairs have variable-length sequences due to different text lengths.
        We batch the image preprocessing for efficiency, then combine each image with its
        corresponding text. Returns a list of variable-length token sequences.

        Args:
            images: List of PIL Images to tokenize
            resize_size: Target size for resizing images
            text: List of text strings (required for image-text pair mode)

        Returns:
            List of token tensors, one per image-text pair (variable lengths)
            Worker code should handle this by processing each sequence individually
        """
        if text is None or len(text) == 0:
            raise ValueError("Text is required for image-text pair tokenization")

        if len(images) != len(text):
            raise ValueError(f"Number of images ({len(images)}) must match number of texts ({len(text)})")

        def tokenize_texts_cpu():
            """CPU thread for batch text tokenization."""
            # Force text tokenization to CPU
            with torch.cuda.device(-1):  # Use CPU
                text_tokens_dict = self.text_tokenizer(
                    text,
                    truncation=False,
                    add_special_tokens=False,
                    return_tensors="pt",
                    padding=False  # Don't pad - we want individual sequences
                )
                # Returns dict with "input_ids" as list of tensors (one per text)
                return text_tokens_dict["input_ids"]

        # Submit both tasks to executor for parallel processing
        # Image tokenization on GPU (bottleneck)
        image_future = self.executor.submit(self.tokenize_batch, images, resize_size)

        # Text tokenization on CPU (fast, runs in parallel)
        text_future = self.executor.submit(tokenize_texts_cpu)

        # Wait for both and get results
        image_tokens_batch = image_future.result()  # [B, image_seq_len] - each image encapsulated with BOS/EOS
        text_tokens_batch = text_future.result()    # List of [text_seq_len] tensors

        # Move text tokens to same device as image tokens for concatenation
        text_tokens_batch = [t.to(image_tokens_batch.device) for t in text_tokens_batch]

        # Combine each image-text pair based on mode
        combined_batch = []
        for i in range(len(images)):
            image_tokens = image_tokens_batch[i]  # [image_seq_len]
            text_tokens = text_tokens_batch[i]    # [text_seq_len]

            if self.mode == "text2image":
                # Text first, then image: [BOS] + [text] + [image without BOS]
                combined = torch.cat([
                    image_tokens[:1],     # BOS token
                    text_tokens,          # Text tokens
                    image_tokens[1:],     # Image tokens (including EOS)
                ])
            elif self.mode == "image2text":
                # Image first, then text: [image without EOS] + [text] + [EOS]
                combined = torch.cat([
                    image_tokens[:-1],    # Image tokens without EOS
                    text_tokens,          # Text tokens
                    image_tokens[-1:],    # EOS token
                ])
            else:
                raise ValueError(f"Invalid mode for image_text_pair tokenizer: {self.mode}")

            combined_batch.append(combined)

        # Return list of variable-length sequences (not a stacked tensor)
        # Worker code will detect this and process each sequence individually
        return combined_batch
