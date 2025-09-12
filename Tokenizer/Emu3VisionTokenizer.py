import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
from typing import Tuple, Any
from transformers import AutoModel, AutoImageProcessor
from Tokenizer.base import Tokenizer
from Tokenizer.Emu3.emu3.tokenizer.image_processing_emu3visionvq import Emu3VisionVQImageProcessor

class Emu3VisionTokenizer(Tokenizer):
    """Emu3 Vision Tokenizer implementation"""
    
    def __init__(self, 
                 model_path: str = "BAAI/Emu3-VisionTokenizer", 
                 min_pixels: int = None,
                 max_pixels: int = None,
                 **kwargs):
        self.model_path = model_path
        self.name = "Emu3VisionTokenizer"
        self.processor = None
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        super().__init__(**kwargs)
    
    def _load_model(self) -> None:
        """Load the Emu3 vision tokenizer model"""
        print(f"Loading {self.name} from {self.model_path}...")
        try:
            # Uses local cache if available, downloads only if needed
            self.model = AutoModel.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            ).eval().to(self.device)
            
            self.processor = Emu3VisionVQImageProcessor.from_pretrained(
                self.model_path, 
                local_files_only=True,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels
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

    from utils_benchmark import load_all_images, resize_by_ratio

    # Initialize the Emu3 tokenizer
    tokenizer = Emu3VisionTokenizer()
    
    # Get model parameters
    tokenizer.get_params()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Process multiple ratios
    processing_ratios = [1.0]
    
    # Load images
    images, image_names, image_paths = load_all_images('/users/nirmiger/benchmark-image-tokenzier/assets/high_aspect_ratio')
    
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
            RECONSTRUCTION_PATH = f'/users/nirmiger/benchmark-image-tokenzier/assets/high_aspect_ratio_recon_2_{tokenizer.name}'
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
                total_tokens = tokenizer.get_num_tokens(indices)
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