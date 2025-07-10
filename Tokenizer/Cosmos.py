import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cosmos.cosmos_predict1.tokenizer.inference.image_lib import ImageTokenizer
from cosmos.cosmos_predict1.tokenizer.inference.utils import (
    tensor2numpy,
    numpy2tensor,
)
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any
from Tokenizer.base import Tokenizer
import matplotlib.pyplot as plt
import os

class Cosmos(Tokenizer):
    """Cosmos image tokenizer implementation"""
    
    def __init__(self, model_name: str = "Cosmos-0.1-Tokenizer-DI8x8", 
                 checkpoint_dir: str = "/cosmos/checkpoints", 
                 device: str = "cuda", 
                 dtype: str = "bfloat16", 
                 **kwargs):
        """
        Initialize Cosmos tokenizer
        
        Args:
            model_name: Name of the Cosmos model (e.g., "Cosmos-0.1-Tokenizer-DI8x8")
            checkpoint_dir: Directory containing the model checkpoints
            device: Device to run the model on
            dtype: Data type for model inference
        """
        self.name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.device_str = device
        self.dtype_str = dtype
        
        # Call parent constructor
        super().__init__(model_name=model_name, checkpoint_dir=checkpoint_dir, 
                        device=device, dtype=dtype, **kwargs)
    
    def _load_model(self) -> None:
        """Load the Cosmos tokenizer model"""
        encoder_ckpt = f"{self.checkpoint_dir}/{self.name}/encoder.jit"
        decoder_ckpt = f"{self.checkpoint_dir}/{self.name}/decoder.jit"
        
        self.model = ImageTokenizer(
            checkpoint_enc=encoder_ckpt,
            checkpoint_dec=decoder_ckpt,
            device=self.device_str,
            dtype=self.dtype_str,
        )
        
        print(f"Loaded {self.name} tokenizer")
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image to tensor format expected by Cosmos"""
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert to numpy array
        input_image = np.array(image)
        assert input_image.ndim == 3 and input_image.shape[2] == 3, "Image must have shape H x W x 3"
        
        # Add batch dimension: B x H x W x C
        batched_input_image = np.expand_dims(input_image, axis=0)
        
        # Convert to tensor
        input_tensor = numpy2tensor(
            batched_input_image, 
            dtype=self.model._dtype, 
            device=self.model._device
        )
        
        return input_tensor
    
    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """Postprocess tensor back to PIL image"""
        # Convert tensor to numpy and remove batch dimension
        numpy_image = tensor2numpy(tensor)[0]
        
        # Convert to PIL Image
        reconstructed_image = Image.fromarray(numpy_image.astype(np.uint8))
        
        return reconstructed_image
    
    def encode(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode tensor to discrete tokens
        
        Returns:
            indices: Discrete token indices
            discrete_codes: Continuous latent codes
        """
        with torch.no_grad():
            encoded_output = self.model.encode(tensor)
            indices = encoded_output[0]  # The discrete indices
            discrete_codes = encoded_output[1]  # The continuous codes
            
        return indices, discrete_codes
    
    def decode(self, indices: torch.Tensor, additional_info: Any = None) -> torch.Tensor:
        """Decode discrete tokens back to tensor
        
        Args:
            indices: discrete token indices
            additional_info: discrete codes (not used for basic decoding)
        """
        with torch.no_grad():
            reconstructed_tensor = self.model.decode(indices)
            
        return reconstructed_tensor
    
    def get_num_tokens(self, indices: torch.Tensor) -> int:
        """Get number of tokens for compression ratio calculation"""
        return indices.numel() // indices.shape[0]  # tokens per image
    
    def get_compression_info(self, image: Image.Image) -> Dict[str, Any]:
        """Get detailed compression information for an image"""
        input_tensor = self.preprocess(image)
        indices, discrete_codes = self.encode(input_tensor)
        
        num_tokens = self.get_num_tokens(indices)
        original_pixels = input_tensor.shape[-2] * input_tensor.shape[-1]
        compression_ratio = original_pixels / num_tokens
        
        # Extract patch size from model name
        if "8x8" in self.name:
            patch_size = 8
        elif "16x16" in self.name:
            patch_size = 16
        else:
            patch_size = "unknown"
        
        return {
            'model_name': self.name,
            'patch_size': patch_size,
            'num_tokens': num_tokens,
            'original_pixels': original_pixels,
            'compression_ratio': compression_ratio,
            'input_shape': tuple(input_tensor.shape),
            'indices_shape': tuple(indices.shape),
            'discrete_codes_shape': tuple(discrete_codes.shape),
            'original_size': (image.width, image.height)
        }
    
    def __repr__(self) -> str:
        return f"Cosmos(model_name='{self.name}', device='{self.device_str}', dtype='{self.dtype_str}')"
    

if __name__ == "__main__":
    # Example usage replicating your original code structure
    import sys
    
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from utils import load_all_images, resize_by_ratio

    # Initialize the Cosmos tokenizer
    tokenizer = Cosmos(
        model_name="Cosmos-0.1-Tokenizer-DI8x8",  # or "Cosmos-0.1-Tokenizer-DI16x16"
        checkpoint_dir="/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/cosmos/checkpoints",
        device="cuda", 
        dtype="bfloat16"
    )
    
    # Get model parameters
    tokenizer.get_params()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Process multiple ratios
    processing_ratios = [1.0, 0.9, 0.8, 0.7]
    
    # Load images
    images, image_names, image_paths = load_all_images('/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/original')
    
    print(f"Found {len(images)} images to process")
    print(f"Processing ratios: {processing_ratios}")
    
    # Process each ratio
    for processing_ratio in processing_ratios:
        print(f"\n{'='*80}")
        if processing_ratio == 1.0:
            print(f"PROCESSING WITH FULL RESOLUTION (no resize)")
        else:
            print(f"PROCESSING WITH RATIO: {processing_ratio}")
            print(f"EFFECTIVE AREA RATIO: {processing_ratio**2:.3f}")
        print(f"{'='*80}")
        
        # Setup paths - different naming for ratio 1.0
        if processing_ratio == 1.0:
            RECONSTRUCTION_PATH = f'/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/{tokenizer.name}'
        else:
            RECONSTRUCTION_PATH = f'/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/{tokenizer.name}_ratio_{processing_ratio**2:.3f}'
        os.makedirs(RECONSTRUCTION_PATH, exist_ok=True)
        
        # Process each image with current ratio
        with torch.no_grad():
            for idx, (image, name) in enumerate(zip(images, image_names)):
                print(f"\n{'-'*60}")
                print(f"Processing image {idx+1}/{len(images)}: {name} (ratio {processing_ratio})")
                print(f"{'-'*60}")

                # Use tokenizer's preprocess method but remove batch dimension
                image_tensor = tokenizer.preprocess(image).squeeze(0)
                print(f"Normalized image shape: {image_tensor.shape}")
                
                # Store original size for later restoration
                original_size = (image_tensor.shape[1], image_tensor.shape[2])  # (H, W)
                
                # Step 1: Resize by ratio to processing size (skip if ratio is 1.0)
                if processing_ratio == 1.0:
                    print("📷 Using full resolution (no resize)")
                    resized_tensor = image_tensor
                else:
                    print("📉 Resizing by ratio to processing size...")
                    resized_tensor = resize_by_ratio(image_tensor, processing_ratio)
                
                # Step 2: Process through tokenizer (encode + decode)
                print("🔄 Processing through tokenizer...")
                # Add batch dimension for tokenizer
                batch_tensor = resized_tensor.unsqueeze(0).to(device)
                
                # Encode and decode
                indices, additional_info = tokenizer.encode(batch_tensor)
                reconstructed_batch = tokenizer.decode(indices, additional_info)
                
                # Remove batch dimension and move to CPU
                reconstructed_tensor = reconstructed_batch.squeeze(0).clamp(-1, 1).cpu()
                
                print(f"Encoded codes shape: {indices.shape}")
                
                # Calculate tokens and compression ratio
                total_tokens = indices.numel()
                original_pixels = original_size[0] * original_size[1]  # H * W
                compression_ratio = original_pixels / total_tokens
                
                print(f"Total tokens: {total_tokens}")
                print(f"Original image pixels: {original_pixels}")
                print(f"Compression ratio: {compression_ratio:.2f}x")
                
                # Convert to PIL for display and saving using tokenizer's postprocess
                original_pil = tokenizer.postprocess(image_tensor.unsqueeze(0))
                reconstructed_pil = tokenizer.postprocess(reconstructed_tensor.unsqueeze(0))
                
                # Create filename with ratio and token count
                name_without_ext = os.path.splitext(name)[0]
                output_filename = f"{name_without_ext}_{total_tokens}.png"
                output_path = os.path.join(RECONSTRUCTION_PATH, output_filename)
                
                # Save the reconstructed image
                reconstructed_pil.save(output_path)
                print(f"  💾 Saved: {output_filename}")
                
                # Show results
                if processing_ratio == 1.0:
                    # For full resolution, show only original and reconstructed
                    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
                    
                    # Original image
                    axes[0].imshow(original_pil)
                    axes[0].set_title(f"Original: {name}\n{original_size[0]}×{original_size[1]}")
                    axes[0].axis('off')
                    
                    # Reconstructed image
                    axes[1].imshow(reconstructed_pil)
                    axes[1].set_title(f"Reconstructed: {name_without_ext}\n{total_tokens} tokens, {compression_ratio:.2f}x compression")
                    axes[1].axis('off')
                else:
                    # For resized versions, show all three
                    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
                    
                    # Original image
                    axes[0].imshow(original_pil)
                    axes[0].set_title(f"Original: {name}\n{original_size[0]}×{original_size[1]}")
                    axes[0].axis('off')
                    
                    # Resized version (for reference)
                    resized_pil = tokenizer.postprocess(resized_tensor.unsqueeze(0))
                    axes[1].imshow(resized_pil)
                    axes[1].set_title(f"Resized (ratio {processing_ratio:.3f})\n{resized_tensor.shape[-2]}×{resized_tensor.shape[-1]}")
                    axes[1].axis('off')
                    
                    # Reconstructed image
                    axes[2].imshow(reconstructed_pil)
                    axes[2].set_title(f"Reconstructed: {name_without_ext}\n{total_tokens} tokens, {compression_ratio:.2f}x compression")
                    axes[2].axis('off')
                
                plt.tight_layout()
                plt.show()
                
                # Clean up memory
                del image_tensor, resized_tensor, batch_tensor, indices, reconstructed_batch, reconstructed_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("\n✅ Processing complete for all ratios!")