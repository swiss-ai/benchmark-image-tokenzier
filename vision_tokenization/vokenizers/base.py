#!/usr/bin/env python3
"""
Base tokenizer class for unified tokenization interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
import torch


class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers with unified interface.

    All tokenizers should implement the tokenize method which accepts
    both image and text inputs, using what's appropriate for their mode.
    """

    @abstractmethod
    def tokenize(self, image=None, text=None) -> Union[torch.Tensor, list]:
        """
        Unified tokenization interface.

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

    def __call__(self, image=None, text=None) -> Union[torch.Tensor, list]:
        """
        Allow tokenizer to be called directly.

        This enables usage like: tokens = tokenizer(image=img, text=txt)
        """
        return self.tokenize(image=image, text=text)