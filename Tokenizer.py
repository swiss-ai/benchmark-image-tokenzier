from abc import ABC, abstractmethod
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict, Any
import torchvision.transforms as T


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
            'num_tokens': num_tokens,
            'original_pixels': original_pixels,
            'compression_ratio': compression_ratio,
            'input_shape': input_tensor.shape,
            'indices_shape': indices.shape,
            'additional_info': additional_info
        }
        
        return reconstructed_image, metrics