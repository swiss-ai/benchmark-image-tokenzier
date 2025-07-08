import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Optional
from OpenMAGViT2.src.Open_MAGVIT2.models.lfqgan import VQModel
from OpenMAGViT2.evaluation_image import load_config
from .base import Tokenizer

class OpenMAGViT2(Tokenizer):
    """OpenMAGViT2 discrete image tokenizer"""
    
    def __init__(self, 
                 config_file: str,
                 ckpt_path: str,
                 **kwargs):
        self.config_file = config_file
        self.ckpt_path = ckpt_path
        self.name = "OpenMAGViT2_256"
        
        super().__init__(**kwargs)
    
    def _load_model(self) -> None:
        """Load the OpenMAGViT2 model"""
        # Load model configuration
        config_model = load_config(self.config_file, display=False)
        
        # Initialize model
        self.model = VQModel(**config_model.model.init_args)
        
        # Load state dict
        sd = torch.load(self.ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"OpenMAGViT2 model loaded successfully")
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

        self.get_params()
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image to tensor format expected by OpenMAGViT2"""
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert to numpy and normalize to [-1, 1]
        image_np = np.array(image)
        image_np = (image_np / 127.5 - 1.0).astype(np.float32)
        
        # Convert to tensor (CHW format) and add batch dimension
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        image_batch = image_tensor.unsqueeze(0).to(self.device)
        
        return image_batch
    
    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """Postprocess tensor back to PIL image"""
        # Remove batch dimension if present
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Clamp to [-1, 1] and convert to [0, 1]
        tensor = tensor.clamp(-1, 1)
        tensor = (tensor + 1.0) / 2.0
        
        # Convert to numpy (HWC format) and scale to [0, 255]
        image_np = tensor.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        # Convert to PIL Image
        return Image.fromarray(image_np)
    
    def encode(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Encode tensor to discrete tokens"""
        with torch.no_grad():
            if self.model.use_ema:
                with self.model.ema_scope():
                    quant, diff, indices, _ = self.model.encode(tensor)
            else:
                quant, diff, indices, _ = self.model.encode(tensor)
            
            # Store additional info needed for decoding
            additional_info = {
                'quant': quant,
                'diff': diff,
                'original_shape': tensor.shape
            }
            
            return indices, additional_info
    
    def decode(self, indices: torch.Tensor, additional_info: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Decode discrete tokens back to tensor"""
        with torch.no_grad():

            quant = additional_info['quant'].to(self.device)

            if self.model.use_ema:
                with self.model.ema_scope():
                    reconstructed = self.model.decode(quant)
            else:
                reconstructed = self.model.decode(quant)
            
            return reconstructed.clamp(-1, 1)
    
    def get_num_tokens(self, indices: torch.Tensor) -> int:
        """Get number of tokens for compression ratio calculation"""
        return indices.numel()