#!/usr/bin/env python3
"""
EMU tokenizer for image-text pairs with parallel GPU/CPU processing.
"""

from concurrent.futures import ThreadPoolExecutor

import torch

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
        image_tokens = image_future.result()
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
            raise ValueError(f"Invalid mode for image_text_pair tokenizer: {self.mode}")

        return combined_tokens

    def tokenize(self, image, text) -> torch.Tensor:
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
