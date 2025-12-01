import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


class Tokenizer(ABC):
    """Abstract base class for discrete image tokenizers"""

    def __init__(self, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.config = kwargs
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Load the specific tokenizer model"""
        pass

    @abstractmethod
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image to tensor format expected by the model"""
        pass

    @abstractmethod
    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """Postprocess tensor back to PIL image"""
        pass

    @abstractmethod
    def encode(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Encode tensor to discrete tokens
        Returns: (indices, additional_info)
        """
        pass

    @abstractmethod
    def decode(self, indices: torch.Tensor, additional_info: Any = None) -> torch.Tensor:
        """Decode discrete tokens back to tensor
        Args:
            indices: discrete token indices
            additional_info: additional information needed for decoding (e.g., latent shape)
        """
        pass

    @abstractmethod
    def get_num_tokens(self, indices: torch.Tensor) -> int:
        """Get number of tokens for compression ratio calculation"""
        pass

    def reconstruct(self, image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        """Full reconstruction pipeline with metrics"""
        # Preprocess
        input_tensor = self.preprocess(image)

        # Encode
        indices, additional_info = self.encode(input_tensor)

        # Decode - now passes additional_info
        reconstructed_tensor = self.decode(indices, additional_info)

        # Postprocess
        reconstructed_image = self.postprocess(reconstructed_tensor)

        # Calculate metrics
        num_tokens = self.get_num_tokens(indices)
        original_pixels = input_tensor.shape[-2] * input_tensor.shape[-1]
        compression_ratio = original_pixels / num_tokens

        metrics = {
            "num_tokens": num_tokens,
            "original_pixels": original_pixels,
            "compression_ratio": compression_ratio,
            "input_shape": input_tensor.shape,
            "indices_shape": indices.shape,
            "additional_info": additional_info,
        }

        return reconstructed_image, metrics

    def get_params(self) -> Dict[str, Any]:
        """Get parameter counts for the model"""
        assert self.model is not None, "Model must be loaded before getting parameters"
        assert hasattr(self, "name"), "Model must have name attribute"

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"{self.name} loaded:")
        print(f"  Total parameters: {Tokenizer._format_number(total_params)} ({total_params:,})")
        print(f"  Trainable parameters: {Tokenizer._format_number(trainable_params)} ({trainable_params:,})")

    @staticmethod
    def _format_number(num: int) -> str:
        """Format large numbers with appropriate units"""
        if num >= 1e9:
            return f"{num/1e9:.2f}B"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        else:
            return str(num)
