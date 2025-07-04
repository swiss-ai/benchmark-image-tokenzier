import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from LlamaGen.tokenizer.tokenizer_image.vq_model import VQ_models
from LlamaGen.dataset.augmentation import center_crop_arr
from typing import Tuple, Any
from .base import Tokenizer

class LlamaGen(Tokenizer):
    """LlamaGen tokenizer_image implementation"""
    
    def __init__(self, 
                 vq_model: str = "VQ-16",
                 vq_ckpt: str = None,
                 codebook_size: int = 16384,
                 codebook_embed_dim: int = 8,
                 image_size: int = 512,
                 seed: int = 0,
                 **kwargs):
        self.vq_model = vq_model
        self.vq_ckpt = vq_ckpt
        self.codebook_size = codebook_size
        self.codebook_embed_dim = codebook_embed_dim
        self.image_size = image_size
        self.seed = seed
        
        # Set seed for reproducibility
        torch.manual_seed(self.seed)
        torch.set_grad_enabled(False)
        
        super().__init__(**kwargs)
    
    def _load_model(self) -> None:
        """Load the LlamaGen tokenizer_image model"""
        if self.vq_ckpt is None:
            raise ValueError("vq_ckpt path must be provided")
            
        # Create model
        self.model = VQ_models[self.vq_model](
            codebook_size=self.codebook_size,
            codebook_embed_dim=self.codebook_embed_dim
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Load checkpoint
        checkpoint = torch.load(self.vq_ckpt, map_location="cpu")
        if "ema" in checkpoint:  # ema
            model_weight = checkpoint["ema"]
        elif "model" in checkpoint:  # ddp
            model_weight = checkpoint["model"]
        elif "state_dict" in checkpoint:
            model_weight = checkpoint["state_dict"]
        else:
            raise Exception("please check model weight")
        
        self.model.load_state_dict(model_weight)
        del checkpoint
        
        print(f"Loaded LlamaGen tokenizer_image model: {self.vq_model}")
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image to tensor format expected by tokenizer_image"""
        if not image.mode == "RGB":
            image = image.convert("RGB")
        
        # Store original size for later reconstruction
        # self.original_size = image.size
        
        # Center crop to square (like in original code)
        # img = center_crop_arr(image, self.image_size)
        
        # Normalize to [-1, 1] range
        x = np.array(image) / 177.5 - 1.0 
        
        # Convert to tensor and rearrange dimensions
        x = torch.tensor(x, dtype=torch.float32)
        x = x.unsqueeze(dim=0)  # Add batch dimension
        x = torch.einsum('nhwc->nchw', x)  # Convert to NCHW format
        
        # Move to device
        x_input = x.to(self.device)
        
        return x_input
    
    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """Postprocess tensor back to PIL image"""
        # Interpolate to target size (using bicubic like in original)
        target_size = [self.image_size, self.image_size]
        output = F.interpolate(tensor, size=target_size, mode='bicubic')
        
        # Convert from NCHW to NHWC and remove batch dimension
        output = output.permute(0, 2, 3, 1)[0]
        
        # Convert from [-1, 1] range to [0, 255] uint8
        sample = torch.clamp(127.5 * output + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
        
        return Image.fromarray(sample)
    
    def encode(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Encode tensor to discrete tokens"""
        with torch.no_grad():
            # Fixed: use correct tensor instead of x*input (which was a bug in original)
            latent, _, [_, _, indices] = self.model.encode(tensor)
        
        # Store latent shape for decoding
        additional_info = {
            'latent_shape': latent.shape,
            'latent': latent
        }
        
        return indices, additional_info
    
    def decode(self, indices: torch.Tensor, additional_info: Any = None) -> torch.Tensor:
        """Decode discrete tokens back to tensor"""
        if additional_info is None:
            raise ValueError("additional_info with latent_shape is required for tokenizer_image decoding")
        
        latent_shape = additional_info['latent_shape']
        
        with torch.no_grad():
            # Use decode_code method which takes indices and latent shape
            output = self.model.decode_code(indices, latent_shape)
        
        return output
    
    def get_num_tokens(self, indices: torch.Tensor) -> int:
        """Get number of tokens for compression ratio calculation"""
        return indices.numel()
