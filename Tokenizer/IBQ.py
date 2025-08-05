import os
import sys
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, "repos", "OpenMAGViT2_IBQ"))

import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional
from src.Open_MAGVIT2.models.lfqgan import VQModel
from evaluation_image import load_config, load_vqgan_new
from Tokenizer.OpenMAGViT2 import OpenMAGViT2

class IBQ(OpenMAGViT2):
    """IBQ discrete image tokenizer inheriting from OpenMAGViT2"""
    
    def __init__(self, 
                 config_file: str,
                 ckpt_path: str,
                 model_type: str = "IBQ",
                 **kwargs):
        self.model_type = model_type
        super().__init__(config_file, ckpt_path, **kwargs)
        self.name = "IBQ_256"
    
    def _load_model(self) -> None:
        """Load the IBQ model using the specific IBQ loading function"""
        # Load model configuration
        config_model = load_config(self.config_file, display=False)
        
        # Use the IBQ-specific loading function
        self.model = load_vqgan_new(config_model, self.model_type, ckpt_path=self.ckpt_path)
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"IBQ model loaded successfully")
        print(f"Model device: {next(self.model.parameters()).device}")
        print(f"Model uses EMA: {self.model.use_ema}")

        self.get_params()
    
    def encode(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Encode tensor to discrete tokens using IBQ-specific API"""
        with torch.no_grad():
            if self.model.use_ema:
                with self.model.ema_scope():
                    quant, qloss, (_, _, indices) = self.model.encode(tensor)
            else:
                quant, qloss, (_, _, indices) = self.model.encode(tensor)
            
            # Store additional info needed for decoding
            additional_info = {
                'quant': quant,
                'qloss': qloss,
                'original_shape': tensor.shape
            }
            
            return indices, additional_info
    
    def decode(self, indices: torch.Tensor, additional_info: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Decode discrete tokens back to tensor using IBQ-specific API"""
        with torch.no_grad():
            
            quant = additional_info['quant'].to(self.device)

            if self.model.use_ema:
                with self.model.ema_scope():
                    reconstructed = self.model.decode(quant)
            else:
                reconstructed = self.model.decode(quant)
            
            return reconstructed.clamp(-1, 1)
    
    def get_compression_info(self, indices: torch.Tensor, original_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Get compression information"""
        total_tokens = indices.numel()
        if len(original_shape) == 4:  # Batch dimension included
            original_pixels = original_shape[-2] * original_shape[-1] * original_shape[0]
        else:  # Single image
            original_pixels = original_shape[-2] * original_shape[-1]
        
        compression_ratio = original_pixels / total_tokens
        
        return {
            'total_tokens': total_tokens,
            'original_pixels': original_pixels,
            'compression_ratio': compression_ratio,
            'tokens_per_image': total_tokens // (original_shape[0] if len(original_shape) == 4 else 1)
        }
    

if __name__ == "__main__":
    # Example usage
    from utils import load_all_images
    from Tiler import Tiler

    # Initialize IBQ tokenizer
    tokenizer = IBQ(
        config_file="/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/OpenMAGViT2_IBQ/configs/IBQ/gpu/imagenet_ibqgan_262144.yaml",
        ckpt_path="/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/OpenMAGViT2_IBQ/upload_ckpts/IBQ/imagenet256_262144.ckpt",
        model_type="IBQ",
        device="cuda"
    )

    processing_ratios = [1.0, 0.9, 0.8, 0.7]
    tile_resize = 256
    batch_size = 8

    # Load images
    images, image_names, image_paths = load_all_images('/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/original')
    
    # Process each ratio
    for processing_ratio in processing_ratios:
        print(f"\n{'='*80}")
        if processing_ratio == 1.0:
            print(f"PROCESSING WITH TILING AT FULL RESOLUTION (no tile resize)")
            tile_orig_size = tile_resize  # No resize, so orig_size = resize
            RECONSTRUCTION_PATH = f'/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/{tokenizer.name}_tiled_{tile_resize}'
        else:
            tile_orig_size = round(tile_resize / processing_ratio)
            print(f"PROCESSING WITH TILING RATIO: {processing_ratio}")
            print(f"TILE ORIG SIZE: {tile_orig_size} -> TILE RESIZE: {tile_resize}")
            print(f"EFFECTIVE AREA RATIO: {processing_ratio**2:.3f}")
            RECONSTRUCTION_PATH = f'/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/{tokenizer.name}_tiled_{tile_resize}_ratio_{processing_ratio**2:.3f}'
        print(f"{'='*80}")
        
        os.makedirs(RECONSTRUCTION_PATH, exist_ok=True)
        
        # Initialize tiler for current ratio
        if processing_ratio == 1.0:
            tiler = Tiler(tile_size=tile_orig_size, pad_value=-1.0)  # No resize
        else:
            tiler = Tiler(tile_size=tile_orig_size, pad_value=-1.0, tile_resize=tile_resize)
        
        print(f"Using tiler with tile_size={tiler.tile_size}, tile_resize={getattr(tiler, 'tile_resize', 'None')}")
        
        # Process each image with current ratio
        with torch.no_grad():
            for idx, (image, name) in enumerate(zip(images, image_names)):
                print(f"\n{'-'*60}")
                print(f"Processing image {idx+1}/{len(images)}: {name} (ratio {processing_ratio})")
                print(f"{'-'*60}")

                # Use tokenizer's preprocess method but remove batch dimension for tiling
                image_tensor = tokenizer.preprocess(image).squeeze(0)
                print(f"Normalized image shape: {image_tensor.shape}")

                # Tile the normalized image
                print("🔲 Tiling image...")
                result = tiler(image_tensor)
                tiles = result['tiles']  # Shape: (total_tiles, C, tile_h, tile_w)
                metadata = result['metadata']

                print(f"Number of tiles: {tiles.shape[0]}")
                print(f"Tile shape: {tiles.shape[1:]}")

                # Process tiles in batches to avoid OOM
                total_tiles = tiles.shape[0]
                reconstructed_tiles_list = []
                all_indices = []

                print(f"🔄 Processing {total_tiles} tiles in batches of {batch_size}")

                for i in range(0, total_tiles, batch_size):
                    end_idx = min(i + batch_size, total_tiles)
                    batch_tiles = tiles[i:end_idx].to(tokenizer.device)

                    print(f"  📦 Processing batch {i//batch_size + 1}/{(total_tiles + batch_size - 1)//batch_size}: tiles {i+1}-{end_idx}")

                    # Process this batch through the tokenizer
                    indices, additional_info = tokenizer.encode(batch_tiles)
                    reconstructed_batch = tokenizer.decode(indices, additional_info)

                     # Clamp and move to CPU to save GPU memory
                    reconstructed_batch = reconstructed_batch.clamp(-1, 1).cpu()
                    reconstructed_tiles_list.append(reconstructed_batch)
                    all_indices.append(indices.cpu())

                    # Clear GPU memory
                    del batch_tiles, reconstructed_batch, indices
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Concatenate all results
                reconstructed_tiles = torch.cat(reconstructed_tiles_list, dim=0)
                all_indices_tensor = torch.cat(all_indices, dim=0)

                # Calculate tokens and compression ratio
                total_tokens = all_indices_tensor.numel()
                original_pixels = image_tensor.shape[-2] * image_tensor.shape[-1]
                compression_ratio = original_pixels / total_tokens

                print(f"Total tokens across all tiles: {total_tokens}")
                print(f"Original image pixels: {original_pixels}")
                print(f"Compression ratio: {compression_ratio:.2f}x")

                # Reconstruct full image from tiles
                reconstructed_full = tiler.full_reconstruct(reconstructed_tiles, metadata)

                print(f"Reconstructed image shape: {reconstructed_full.shape}")

                # Convert to PIL for display and saving using tokenizer's postprocess
                original_pil = tokenizer.postprocess(image_tensor.unsqueeze(0))
                reconstructed_pil = tokenizer.postprocess(reconstructed_full.unsqueeze(0))

                # Create filename with token count
                name_without_ext = os.path.splitext(name)[0]
                output_filename = f"{name_without_ext}_tiled_{total_tokens}.png"
                output_path = os.path.join(RECONSTRUCTION_PATH, output_filename)

                # Save the reconstructed image
                reconstructed_pil.save(output_path)
                print(f"  💾 Saved: {output_filename}")

                # Show results (commented out for performance, uncomment if needed)
                # try:
                #     import matplotlib.pyplot as plt
                #     if processing_ratio == 1.0:
                #         # For full resolution, show only original and reconstructed
                #         fig, axes = plt.subplots(1, 2, figsize=(20, 10))
                        
                #         # Original image
                #         axes[0].imshow(original_pil)
                #         axes[0].set_title(f"Original: {name}\n{image_tensor.shape[-2]}×{image_tensor.shape[-1]}")
                #         axes[0].axis('off')
                        
                #         # Reconstructed image
                #         axes[1].imshow(reconstructed_pil)
                #         axes[1].set_title(f"Reconstructed: {name_without_ext}\n{total_tokens} tokens, {compression_ratio:.2f}x compression")
                #         axes[1].axis('off')
                #     else:
                #         # For resized versions, show original, sample tile, and reconstructed
                #         fig, axes = plt.subplots(1, 3, figsize=(30, 10))
                        
                #         # Original image
                #         axes[0].imshow(original_pil)
                #         axes[0].set_title(f"Original: {name}\n{image_tensor.shape[-2]}×{image_tensor.shape[-1]}")
                #         axes[0].axis('off')
                        
                #         # Sample tile (for reference)
                #         sample_tile_pil = tokenizer.postprocess(tiles[0:1].cpu())
                #         axes[1].imshow(sample_tile_pil)
                #         axes[1].set_title(f"Sample Tile (ratio {processing_ratio:.3f})\n{tiles.shape[-2]}×{tiles.shape[-1]} ({tiles.shape[0]} total tiles)")
                #         axes[1].axis('off')
                        
                #         # Reconstructed image
                #         axes[2].imshow(reconstructed_pil)
                #         axes[2].set_title(f"Reconstructed: {name_without_ext}\n{total_tokens} tokens, {compression_ratio:.2f}x compression")
                #         axes[2].axis('off')
                    
                #     plt.tight_layout()
                #     plt.show()
                # except Exception as e:
                #     print(f"Could not display images: {e}")

                # # Show all tiles in grid format (optional, can be commented out for faster processing)
                # if processing_ratio != 1.0:  # Only show tiles for resized versions
                #     print("Displaying all tiles in grid:")
                #     try:
                #         tiler.show_tiles(tiles.cpu(), metadata)
                #     except Exception as e:
                #         print(f"Could not display tiles: {e}")

                # Clean up memory
                del tiles, reconstructed_tiles, all_indices_tensor, image_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("\n✅ Processing complete for all ratios!")