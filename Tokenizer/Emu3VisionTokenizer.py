import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
from typing import Tuple, Any
from transformers import AutoModel, AutoImageProcessor
from .base import Tokenizer 


class Emu3VisionTokenizer(Tokenizer):
    """Emu3 Vision Tokenizer implementation"""
    
    def __init__(self, model_path: str = "BAAI/Emu3-VisionTokenizer", **kwargs):
        self.model_path = model_path
        self.name = "Emu3VisionTokenizer"
        self.processor = None
        super().__init__(**kwargs)
    
    def _load_model(self) -> None:
        """Load the Emu3 vision tokenizer model"""
        print(f"Loading {self.name} from {self.model_path}...")
        try:
            self.model = AutoModel.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            ).eval().to(self.device)
            
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            print(f"✓ {self.name} loaded successfully")
            self.get_params()
            
        except Exception as e:
            print(f"✗ Failed to load {self.name}: {e}")
            raise
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image using the Emu3 processor"""
        # Convert single image to list format expected by processor
        image_tensor = self.processor([image], return_tensors="pt")["pixel_values"]
        return image_tensor.to(self.device)
    
    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """Postprocess tensor back to PIL image using the Emu3 processor"""
        # The processor.postprocess expects the tensor to be on CPU
        tensor_cpu = tensor.cpu()
        result = self.processor.postprocess(tensor_cpu)
        return result["pixel_values"][0]
    
    def encode(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Encode tensor to discrete tokens using Emu3 model"""
        with torch.no_grad():
            codes = self.model.encode(tensor)
            
            # Store additional info needed for decoding
            additional_info = {
                'codes_shape': codes.shape,
                'original_shape': tensor.shape
            }
            
            return codes, additional_info
    
    def decode(self, indices: torch.Tensor, additional_info: Any = None) -> torch.Tensor:
        """Decode discrete tokens back to tensor using Emu3 model"""
        with torch.no_grad():
            reconstructed = self.model.decode(indices)
            return reconstructed
    
    def get_num_tokens(self, indices: torch.Tensor) -> int:
        """Get number of tokens for compression ratio calculation"""
        if len(indices.shape) == 3:  # [batch, height, width]
            return indices.shape[1] * indices.shape[2]
        elif len(indices.shape) == 4:  # [batch, channels, height, width]
            return indices.shape[2] * indices.shape[3]
        else:
            # For other shapes, calculate total elements per batch
            return indices.numel() // indices.shape[0]
    
if __name__ == "__main__":
    # Example usage replicating your original code structure
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from utils import load_all_images
    
    # Initialize tokenizer
    tokenizer = Emu3VisionTokenizer()
    
    # Set up paths (adjust as needed)
    RECONSTRUCTION_PATH = f'/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/{tokenizer.name}'
    os.makedirs(RECONSTRUCTION_PATH, exist_ok=True)
    
    # Load images
    images, image_names, _ = load_all_images("/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/original")
    
    # Process and encode each image individually
    for idx, (image, name) in enumerate(zip(images, image_names)):
        print(f"\nProcessing image {idx+1}: {name}")
        
        # Reconstruct image using the tokenizer
        recon_image, metrics = tokenizer.reconstruct(image)
        
        # Print metrics
        print(f"Input image shape: {metrics['input_shape']}")
        print(f"Encoded codes shape: {metrics['indices_shape']}")
        print(f"Number of tokens: {metrics['num_tokens']}")
        print(f"Original image pixels: {metrics['original_pixels']}")
        print(f"Compression ratio: {metrics['compression_ratio']:.2f}x")
        
        # Save the reconstructed image
        name_without_ext = os.path.splitext(name)[0]
        output_filename = f"{name_without_ext}_{metrics['num_tokens']}.png"
        output_path = os.path.join(RECONSTRUCTION_PATH, output_filename)
        recon_image.save(output_path)
        print(f"Saved: {output_filename}")
        
        # Display comparison
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title(f"Original: {name}")
        axes[0].axis('off')
        
        # Reconstructed image
        axes[1].imshow(recon_image)
        axes[1].set_title(f"Reconstructed: {name}")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()