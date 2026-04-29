#!/usr/bin/env python3
"""
EMU tokenizer for image-text pairs with parallel GPU/CPU processing.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
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
