import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional, Union, Dict, Any
import torchvision.transforms as T
from abc import ABC, abstractmethod
from FQGAN.tokenizer.vq_model_triple import VQ_models
from .base import Tokenizer

class FQGAN(Tokenizer):
    """FQGAN Tokenizer using triple VQ model architecture"""
    
    def __init__(self, 
                 vq_model: str = "VQ-16",
                 vq_ckpt: str = None,
                 codebook_size: int = 16384,
                 codebook_embed_dim: int = 8,
                 image_size: int = 256,
                 with_clip_supervision: bool = False,
                 with_disentanglement: bool = False,
                 disentanglement_ratio: float = 0.1,
                 **kwargs):
        """
        Initialize FQGAN tokenizer
        
        Args:
            vq_model: VQ model architecture name
            vq_ckpt: Path to VQ model checkpoint
            codebook_size: Size of the codebook
            codebook_embed_dim: Embedding dimension of codebook
            image_size: Input image size
            with_clip_supervision: Whether to use CLIP supervision
            with_disentanglement: Whether to use disentanglement
            disentanglement_ratio: Disentanglement ratio
        """
        self.name = f"FQGAN-{vq_model}"  # More descriptive name
        self.vq_model_name = vq_model    # For VQ_models lookup
        self.vq_ckpt = vq_ckpt
        self.codebook_size = codebook_size
        self.codebook_embed_dim = codebook_embed_dim
        self.image_size = image_size
        self.with_clip_supervision = with_clip_supervision
        self.with_disentanglement = with_disentanglement
        self.disentanglement_ratio = disentanglement_ratio
        
        super().__init__(**kwargs)
    
    def _load_model(self) -> None:
        """Load the FQGAN VQ model"""
        
        # Create VQ model and assign to self.model for base class compatibility
        self.model = VQ_models[self.vq_model_name](
            codebook_size=self.codebook_size,
            codebook_embed_dim=self.codebook_embed_dim,
            with_clip_supervision=self.with_clip_supervision,
            with_disentanglement=self.with_disentanglement,
            disentanglement_ratio=self.disentanglement_ratio,
        )

        # Load checkpoint if provided
        if self.vq_ckpt:
            checkpoint = torch.load(self.vq_ckpt, map_location="cpu")
            
            # Handle different checkpoint formats
            if "ema" in checkpoint:  # ema
                model_weight = checkpoint["ema"]
            elif "model" in checkpoint:  # ddp
                model_weight = checkpoint["model"]
            elif "state_dict" in checkpoint:
                model_weight = checkpoint["state_dict"]
            else:
                raise Exception("Checkpoint format not recognized. Expected 'ema', 'model', or 'state_dict' keys.")
            
            self.model.load_state_dict(model_weight)
            del checkpoint
        
        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Disable gradients
        torch.set_grad_enabled(False)

        # Print parameter count automatically when model is loaded
        self.get_params()
       
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image to tensor format expected by FQGAN"""
        # Convert PIL image to numpy array (assuming it's already the right size from your tiler)
        arr = np.array(image)
        
        # Convert to tensor using torch.tensor (avoids numpy dependency issues)
        tensor = torch.tensor(arr, dtype=torch.float32)
        
        # Normalize from [0, 255] to [0, 1] then to [-1, 1]
        tensor = tensor / 255.0
        tensor = (tensor - 0.5) / 0.5
        
        # Add batch dimension and rearrange to CHW format
        if len(tensor.shape) == 3:
            tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor.to(self.device)
    
    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """Postprocess tensor back to PIL image"""
        # Remove batch dimension if present
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Convert from CHW to HWC
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        
        # Clamp and convert to uint8
        tensor = torch.clamp(127.5 * tensor + 128.0, 0, 255)
        tensor = tensor.to(torch.uint8).cpu().numpy()
        
        # Convert to PIL Image
        return Image.fromarray(tensor)
    
    def encode(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Encode tensor to discrete tokens using triple VQ"""
        with torch.no_grad():
            # Encode using triple VQ model (using self.model)
            latent_vis, latent_sem_mid, latent_sem_high, \
            [_, _, indices_vis], [_, _, indices_sem_mid], [_, _, indices_sem_high] = self.model.encode(tensor)
            
            # Combine all indices
            indices = {
                'indices_vis': indices_vis,
                'indices_sem_mid': indices_sem_mid,
                'indices_sem_high': indices_sem_high
            }
            
            # Store additional info needed for decoding
            additional_info = {
                'latent_shapes': {
                    'vis_shape': latent_vis.shape,
                    'sem_mid_shape': latent_sem_mid.shape,
                    'sem_high_shape': latent_sem_high.shape
                }
            }
            
            return indices, additional_info
    
    def decode(self, indices: Union[torch.Tensor, Dict[str, torch.Tensor]], additional_info: Dict[str, Any] = None) -> torch.Tensor:
        """Decode discrete tokens back to tensor"""
        with torch.no_grad():
            if isinstance(indices, dict):
                # Extract individual indices
                indices_vis = indices['indices_vis']
                indices_sem_mid = indices['indices_sem_mid']
                indices_sem_high = indices['indices_sem_high']
                
                # Extract latent shapes
                latent_shapes = additional_info['latent_shapes']
                vis_shape = latent_shapes['vis_shape']
                sem_mid_shape = latent_shapes['sem_mid_shape']
                sem_high_shape = latent_shapes['sem_high_shape']
                
                # Decode using VQ model (using self.model)
                reconstructed = self.model.decode_code(
                    indices_vis, indices_sem_mid, indices_sem_high,
                    vis_shape, sem_mid_shape, sem_high_shape
                )
            else:
                raise ValueError("FQGAN requires indices as dictionary with 'indices_vis', 'indices_sem_mid', 'indices_sem_high' keys")
            
            return reconstructed
    
    def get_num_tokens(self, indices: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> int:
        """Get total number of tokens across all quantizers"""
        if isinstance(indices, dict):
            total_tokens = 0
            for key, idx in indices.items():
                if isinstance(idx, torch.Tensor):
                    total_tokens += idx.numel()
                else:
                    total_tokens += len(idx.flatten()) if hasattr(idx, 'flatten') else len(idx)
            return total_tokens
        else:
            return indices.numel() if isinstance(indices, torch.Tensor) else len(indices)
    
    def get_codebook_usage(self, indices: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, int]]:
        """Get codebook usage statistics for each quantizer"""
        usage_stats = {}
        
        for key, idx in indices.items():
            if isinstance(idx, torch.Tensor):
                idx_flat = idx.flatten()
                unique_tokens = torch.unique(idx_flat)
                usage_stats[key] = {
                    'unique_tokens': len(unique_tokens),
                    'total_tokens': len(idx_flat),
                    'usage_ratio': len(unique_tokens) / self.codebook_size,
                    'most_frequent': torch.mode(idx_flat)[0].item(),
                    'token_range': (idx_flat.min().item(), idx_flat.max().item())
                }
        
        return usage_stats
    
    def reconstruct(self, image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        """Full reconstruction pipeline with FQGAN-specific metrics"""
        # Call parent reconstruct method
        reconstructed_image, metrics = super().reconstruct(image)
        
        # Add FQGAN-specific metrics
        if isinstance(metrics['additional_info'], dict) and 'latent_shapes' in metrics['additional_info']:
            # Get indices for codebook usage analysis
            input_tensor = self.preprocess(image)
            indices, _ = self.encode(input_tensor)
            
            # Add codebook usage statistics
            if isinstance(indices, dict):
                metrics['codebook_usage'] = self.get_codebook_usage(indices)
                
                # Add per-quantizer token counts
                metrics['tokens_per_quantizer'] = {
                    key: idx.numel() if isinstance(idx, torch.Tensor) else len(idx)
                    for key, idx in indices.items()
                }
        
        return reconstructed_image, metrics