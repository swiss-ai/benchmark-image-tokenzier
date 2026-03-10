#!/usr/bin/env python3
"""
EMU tokenizer for image-text pairs with parallel GPU/CPU processing.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
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
        Tokenize one image-text pair with one image per pair.

        Args:
            image: PIL Image to tokenize (required)
            text: Text string to append after image (required)

        Returns:
            Combined tokenized output as tensor
        """
        # Both image and text are required for image-text pair tokenizer
        return self.tokenize_image_text_pair(image, text)

    def tokenize_batch(self, images, resize_size, text=None, group_slices=None):
        """
        Batched tokenization interface for image-text pair mode.

        Single-image is a special case of multi-image where every group has
        exactly one image.  Both paths share the same code: GPU image
        tokenization runs in parallel with CPU text tokenization.

        Per-group output::

            image2text: [BOS] [img0_struct] [img1_struct] ... [text] [EOS]
            text2image: [BOS] [text] [img0_struct] [img1_struct] ... [EOS]

        Args:
            images: List of PIL Images to tokenize.
            resize_size: Target size for resizing images (batch-wide).
            text: List of text strings (required).  One per image
                (single-image) or one per group (multi-image).
            group_slices: Optional ``(num_groups, 2)`` array mapping groups
                to positions in *images*.  When ``None``, each image is
                treated as its own group.

        Returns:
            List of token tensors (variable lengths).
        """
        if text is None or len(text) == 0:
            raise ValueError("Text is required for image-text pair tokenization")

        # Single-image is multi-image with trivial 1-image groups
        if group_slices is None:
            if len(images) != len(text):
                raise ValueError(
                    f"Number of images ({len(images)}) must match "
                    f"number of texts ({len(text)})"
                )
            group_slices = np.array(
                [[i, i + 1] for i in range(len(images))], dtype=np.int64,
            )

        # GPU image tokenization ∥ CPU text tokenization
        def tokenize_texts_cpu():
            with torch.cuda.device(-1):
                text_tokens_dict = self.text_tokenizer(
                    text,
                    truncation=False,
                    add_special_tokens=False,
                    return_tensors=None,
                    padding=False,
                )
                return [torch.tensor(ids) for ids in text_tokens_dict["input_ids"]]

        image_future = self.executor.submit(self.tokenize_images, images, resize_size)
        text_future = self.executor.submit(tokenize_texts_cpu)

        image_tokens_batch = image_future.result()  # [total_images, seq_len]
        text_tokens_list = text_future.result()

        image_tokens_batch = image_tokens_batch.cpu()

        # Per-group assembly
        results = []
        for g_idx, (gs, ge) in enumerate(group_slices):
            gs, ge = int(gs), int(ge)
            group_img_tokens = image_tokens_batch[gs:ge]  # [num_imgs, seq_len]
            text_tokens = text_tokens_list[g_idx]

            # Strip per-image BOS/EOS to get bare image structure tokens
            img_structs = [group_img_tokens[i, 1:-1] for i in range(group_img_tokens.shape[0])]
            bos = group_img_tokens[0, :1]
            eos = group_img_tokens[0, -1:]

            if self.mode == "image2text":
                parts = [bos] + img_structs + [text_tokens, eos]
            elif self.mode == "text2image":
                parts = [bos, text_tokens] + img_structs + [eos]
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

            results.append(torch.cat(parts))

        return results
