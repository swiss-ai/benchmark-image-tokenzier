#!/usr/bin/env python3
"""
Base tokenizer class for unified tokenization interface.

Any vision tokenizer (EMU, Cosmos, Chameleon, …) should subclass
``BaseTokenizer`` and implement ``tokenize`` (single-sample) and
``tokenize_batch`` (batched, with optional ``group_slices`` for
multi-image datasets).
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import torch


class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers with unified interface.

    All tokenizers should implement:
    - ``tokenize``: single-sample tokenization.
    - ``tokenize_batch``: batched tokenization used by the distributed pipeline.

    The ``tokenize_batch`` contract::

        def tokenize_batch(
            self,
            images: List[PIL.Image],
            resize_size: int,
            text: Optional[List] = None,
            group_slices: Optional[np.ndarray] = None,
        ) -> List[torch.Tensor]:
            ...

    Returns one token sequence per sample (or per *group* when
    ``group_slices`` is provided).
    """

    @abstractmethod
    def tokenize(self, image=None, text=None) -> Union[torch.Tensor, list]:
        """
        Single-sample tokenization interface.

        Different tokenizer implementations have different requirements:
        - Image-only: image is required, text is ignored
        - Image-text pair: both image and text are required
        - SFT: at least one of image or text is required

        Args:
            image: Input image (PIL Image or similar)
            text: Input text (str or list for conversations)

        Returns:
            Tokenized output as tensor or list of token IDs

        Raises:
            ValueError: If required inputs are missing for the tokenizer mode
        """
        pass

    @abstractmethod
    def tokenize_batch(
        self,
        images: list,
        resize_size: int,
        text: Optional[list] = None,
        group_slices: Optional[np.ndarray] = None,
    ) -> List[torch.Tensor]:
        """
        Batched tokenization interface used by the distributed pipeline.

        Args:
            images: List of PIL Images to tokenize.
            resize_size: Target size for resizing images (batch-wide).
            text: Optional list of text data (captions, conversations, …).
            group_slices: Optional ``(num_groups, 2)`` array mapping groups
                to positions in *images*.  When provided, returns one
                sequence per group instead of one per image.

        Returns:
            List of token tensors (variable lengths).
        """
        pass

    def __call__(self, image=None, text=None) -> Union[torch.Tensor, list]:
        """
        Allow tokenizer to be called directly.

        This enables usage like: tokens = tokenizer(image=img, text=txt)
        """
        return self.tokenize(image=image, text=text)
