#!/usr/bin/env python3
"""Base tokenizer class for the batched tokenization interface.

Any vision tokenizer (EMU, Cosmos, Chameleon, …) should subclass
``BaseTokenizer`` and implement ``tokenize_batch`` — the batched entry
point used by the distributed pipeline. Single-sample tokenization is
not part of the contract; tokenizer-specific primitives like
``tokenize_image`` may exist but are not required.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import torch


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers with a batched tokenization interface."""

    @abstractmethod
    def tokenize_batch(
        self,
        images: list,
        resize_size: int,
        text: Optional[list] = None,
        group_slices: Optional[np.ndarray] = None,
    ) -> List[torch.Tensor]:
        """Batched tokenization interface used by the distributed pipeline.

        Args:
            images: List of PIL Images to tokenize.
            resize_size: Target size for resizing images (batch-wide).
            text: Optional list of text data (captions, conversations, …).
            group_slices: Optional ``(num_groups, 2)`` array mapping groups
                to positions in *images*. When provided, returns one
                sequence per group instead of one per image.

        Returns:
            List of token tensors (variable lengths).
        """
        pass
