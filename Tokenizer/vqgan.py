import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from LlamaGen.tokenizer.vqgan.model import VQModel
from LlamaGen.tokenizer.vqgan.model import VQGAN_FROM_TAMING
from typing import Tuple, Any
from .base import Tokenizer


class VQGAN(Tokenizer):
    """VQGAN tokenizer implementation"""
    
    def __init__(self, 
                 vqgan_model: str = "vqgan_openimage_f8_16384", 
                 image_size: int = 512,
                 seed: int = 0,
                 **kwargs):
        self.vqgan_model = vqgan_model
        self.image_size = image_size
        self.seed = seed
        
        # Set seed for reproducibility
        torch.manual_seed(self.seed)
        torch.set_grad_enabled(False)
        
        super().__init__(**kwargs)


    def _load_model(self) -> None:
        """Load the LlamaGen VQGAN model"""
        # Get config and checkpoint paths
        cfg, ckpt = VQGAN_FROM_TAMING[self.vqgan_model]
        
        # Load config and create model
        config = OmegaConf.load(cfg)
        self.model = VQModel(**config.model.get("params", dict()))
        
        # Load checkpoint and setup
        self.model.init_from_ckpt(ckpt)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image to tensor format expected by LlamaGen VQGAN"""
        if not image.mode == "RGB":
            image = image.convert("RGB")
        
        # Store original size for later reconstruction
        self.original_size = image.size
        
        # Resize to model input size
        # image = image.resize((self.image_size, self.image_size))
        
        # Normalize to [-1, 1] range as expected by VQGAN
        x = np.array(image) / 127.5 - 1.0
        
        # Convert to tensor and rearrange dimensions
        x = torch.tensor(x, dtype=torch.float32)
        x = x.unsqueeze(dim=0)  # Add batch dimension
        x = torch.einsum('nhwc->nchw', x)  # Convert to NCHW format
        
        # Move to device
        x_input = x.to(self.device)
        
        return x_input
    
    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """Postprocess tensor back to PIL image"""
        # Interpolate back to original size if needed
        # if hasattr(self, 'original_size') and self.original_size:
        #     # Note: PIL size is (width, height), but interpolate expects (height, width)
        #     target_size = [self.original_size[1], self.original_size[0]]
        #     tensor = F.interpolate(tensor, size=target_size, mode='bilinear')
        
        # Convert from NCHW to NHWC and remove batch dimension
        output = tensor.permute(0, 2, 3, 1)[0]
        
        # Convert from [-1, 1] range to [0, 255] uint8
        sample = torch.clamp(127.5 * output + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
        
        return Image.fromarray(sample)
    
    def encode(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Encode tensor to discrete tokens"""
        with torch.no_grad():
            # LlamaGen VQGAN encode returns: latent, *, [*, *, indices]
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
            raise ValueError("additional_info with latent_shape is required for LlamaGen decoding")
        
        latent_shape = additional_info['latent_shape']
        
        with torch.no_grad():
            # Use decode_code method which takes indices and latent shape
            output = self.model.decode_code(indices, latent_shape)
        
        return output
    
    def get_num_tokens(self, indices: torch.Tensor) -> int:
        """Get number of tokens for compression ratio calculation"""
        return indices.numel()