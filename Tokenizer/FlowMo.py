import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from omegaconf import OmegaConf
from FlowMo.flowmo import train_utils
from typing import Tuple, Any
from .base import Tokenizer


class FlowMo(Tokenizer):
    """FlowMo tokenizer implementation"""
    
    def __init__(self, 
                 model_name: str = "flowmo_lo",
                 num_tokens: int = None,
                 **kwargs):
        self.model_name = model_name
        self.num_tokens = num_tokens  # Manual override if provided
        
        # FlowMo model zoo
        self.zoo = {
            "flowmo_lo": {"context_dim": 18, "ckpt_path": "FlowMo/flowmo_lo.pth"},
            "flowmo_hi": {"context_dim": 56, "ckpt_path": "FlowMo/flowmo_hi.pth"},
        }
        
        # Disable torch.compile if available
        try:
            torch.compiler.set_stance("force_eager")
        except:
            pass
        
        super().__init__(**kwargs)
    
    def _load_model(self) -> None:
        """Load the FlowMo model"""
        # Load config
        config = OmegaConf.load('FlowMo/flowmo/configs/base.yaml')
        config.data.batch_size = 8
        config.data.num_workers = 0
        
        # Set model-specific config
        config.model.context_dim = self.zoo[self.model_name]['context_dim']
        config.model.codebook_size_for_entropy = 1  # don't need this at test time
        
        # Build model
        self.model = train_utils.build_model(config)
        
        # Load checkpoint
        state_dict = torch.load(self.zoo[self.model_name]['ckpt_path'], map_location=self.device)
        self.model.load_state_dict(state_dict['model_ema_state_dict'])
        self.model = self.model.to(self.device)
        
        print(f"Loaded FlowMo model: {self.model_name}")
        print(f"Model config code_length: {self.model.code_length}")
        print(f"Model config context_dim: {self.model.config.model.context_dim}")
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image to tensor format expected by FlowMo"""
        if not image.mode == "RGB":
            image = image.convert("RGB")
        
        # Store original size for later reconstruction
        self.original_size = image.size
        
        # No cropping/resizing - tiler handles this
        # Just convert to numpy and normalize
        image_np = np.array(image)
        image_np = (image_np / 127.5 - 1.0).astype(np.float32)  # FlowMo normalization
        
        # Convert to tensor (C, H, W) format and add batch dimension
        tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return tensor
    
    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """Postprocess tensor back to PIL image"""
        # Tensor is in range [-1, 1], convert back to [0, 255]
        tensor = (tensor + 1.0) * 127.5
        tensor = tensor.clamp(0, 255)
        
        # Convert CHW to HWC and to numpy
        if tensor.dim() == 4:  # Remove batch dimension if present
            tensor = tensor.squeeze(0)
        image_array = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        return Image.fromarray(image_array)
    
    def encode(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Encode tensor to discrete tokens
        Note: FlowMo doesn't expose separate encode method, 
        so this is a placeholder that stores the input for decode
        """
        # FlowMo doesn't expose encode/decode separately, only reconstruct
        # So we store the input tensor and return dummy indices
        self._input_tensor = tensor
        dummy_indices = torch.empty(0)  # Placeholder
        additional_info = {'input_tensor': tensor}
        return dummy_indices, additional_info
    
    def decode(self, indices: torch.Tensor, additional_info: Any = None) -> torch.Tensor:
        """Decode discrete tokens back to tensor
        Note: FlowMo uses reconstruct() directly, so this uses the stored input
        """
        if additional_info is None or 'input_tensor' not in additional_info:
            raise ValueError("FlowMo requires input_tensor in additional_info for reconstruction")
        
        input_tensor = additional_info['input_tensor']
        
        with torch.no_grad():
            # Use FlowMo's reconstruct method
            reconstructed = self.model.reconstruct(input_tensor)
        
        return reconstructed
    
    def get_num_tokens(self, indices: torch.Tensor) -> int:
        """Get number of tokens for compression ratio calculation"""
        # Use manual override if provided, otherwise use model's code_length
        assert self.num_tokens is not None, "num_tokens must be set for FlowMo"
        return self.num_tokens
    
    def reconstruct(self, image: Image.Image) -> Tuple[Image.Image, dict]:
        """Full reconstruction pipeline with metrics"""
        # Use the parent class's reconstruct method which calls encode() then decode()
        return super().reconstruct(image)

