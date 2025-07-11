import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from LlamaGen.tokenizer.tokenizer_image.vq_model import VQ_models
from typing import Tuple, Any
from Tokenizer.base import Tokenizer

class LlamaGen(Tokenizer):
    """LlamaGen tokenizer_image implementation"""
    
    def __init__(self, 
                 vq_model: str = "VQ-16",
                 vq_ckpt: str = None,
                 codebook_size: int = 16384,
                 codebook_embed_dim: int = 8,
                 seed: int = 0,
                 **kwargs):
        self.vq_model = vq_model
        self.vq_ckpt = vq_ckpt
        self.codebook_size = codebook_size
        self.codebook_embed_dim = codebook_embed_dim
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
        
        # Normalize to [-1, 1] range
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
        # Convert from NCHW to NHWC and remove batch dimension
        output = tensor.permute(0, 2, 3, 1)[0]
        
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


if __name__ == "__main__":
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt
    import sys
    
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from utils import load_all_images
    from Tiler import Tiler
    
    # Initialize tokenizer
    tokenizer = LlamaGen(
        vq_model="VQ-8",
        vq_ckpt="/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/LlamaGen/pretrained_models/vq_ds8_c2i.pt",
        codebook_size=16384,
        codebook_embed_dim=8,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Process multiple ratios
    processing_ratios = [1.0, 0.9, 0.8, 0.7]
    tile_resize = 256
    batch_size = 8
    
    # Load images
    images, image_names, image_paths = load_all_images('/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/original')
    
    print(f"Found {len(images)} images to process")
    print(f"Processing ratios: {processing_ratios}")
    print(f"Tile resize: {tile_resize}")
    print(f"VQ Model: {tokenizer.vq_model}")
    print(f"Codebook size: {tokenizer.codebook_size}")
    
    # Process each ratio
    for processing_ratio in processing_ratios:
        print(f"\n{'='*80}")
        if processing_ratio == 1.0:
            print(f"PROCESSING WITH TILING AT FULL RESOLUTION (no tile resize)")
            tile_orig_size = tile_resize  # No resize, so orig_size = resize
            RECONSTRUCTION_PATH = f'/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/LlamaGen_vq8_{tile_resize}'
        else:
            tile_orig_size = round(tile_resize / processing_ratio)
            print(f"PROCESSING WITH TILING RATIO: {processing_ratio}")
            print(f"TILE ORIG SIZE: {tile_orig_size} -> TILE RESIZE: {tile_resize}")
            print(f"EFFECTIVE AREA RATIO: {processing_ratio**2:.3f}")
            RECONSTRUCTION_PATH = f'/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/LlamaGen_vq8_{tile_resize}_ratio_{processing_ratio**2:.3f}'
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
                    batch_tiles = tiles[i:end_idx].to(device)

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

                print(f"Encoded codes shape: {all_indices_tensor.shape}")

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

                # # Show results
                # if processing_ratio == 1.0:
                #     # For full resolution, show only original and reconstructed
                #     fig, axes = plt.subplots(1, 2, figsize=(20, 10))
                    
                #     # Original image
                #     axes[0].imshow(original_pil)
                #     axes[0].set_title(f"Original: {name}\n{image_tensor.shape[-2]}×{image_tensor.shape[-1]}")
                #     axes[0].axis('off')
                    
                #     # Reconstructed image
                #     axes[1].imshow(reconstructed_pil)
                #     axes[1].set_title(f"Reconstructed: {name_without_ext}\n{total_tokens} tokens, {compression_ratio:.2f}x compression")
                #     axes[1].axis('off')
                # else:
                #     # For resized versions, show original, sample tile, and reconstructed
                #     fig, axes = plt.subplots(1, 3, figsize=(30, 10))
                    
                #     # Original image
                #     axes[0].imshow(original_pil)
                #     axes[0].set_title(f"Original: {name}\n{image_tensor.shape[-2]}×{image_tensor.shape[-1]}")
                #     axes[0].axis('off')
                    
                #     # Sample tile (for reference)
                #     sample_tile_pil = tokenizer.postprocess(tiles[0:1].cpu())
                #     axes[1].imshow(sample_tile_pil)
                #     axes[1].set_title(f"Sample Tile (ratio {processing_ratio:.3f})\n{tiles.shape[-2]}×{tiles.shape[-1]} ({tiles.shape[0]} total tiles)")
                #     axes[1].axis('off')
                    
                #     # Reconstructed image
                #     axes[2].imshow(reconstructed_pil)
                #     axes[2].set_title(f"Reconstructed: {name_without_ext}\n{total_tokens} tokens, {compression_ratio:.2f}x compression")
                #     axes[2].axis('off')
                
                # plt.tight_layout()
                # plt.show()

                # # Show all tiles in grid format (optional, can be commented out for faster processing)
                # if processing_ratio != 1.0:  # Only show tiles for resized versions
                #     print("Displaying all tiles in grid:")
                #     tiler.show_tiles(tiles.cpu(), metadata)

                # # Clean up memory
                # del tiles, reconstructed_tiles, all_indices_tensor, image_tensor
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()

    print("\n✅ Processing complete for all ratios!")