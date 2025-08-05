import os
import sys

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, "repos", "OpenMAGViT2_IBQ"))

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Optional
from evaluation_image import load_config
from Tokenizer.base import Tokenizer
import matplotlib.pyplot as plt

# Model variant configurations
MODEL_MODULES = {
    "OpenMAGViT2_256": "OpenMAGViT2_IBQ.src.Open_MAGVIT2.models.lfqgan",
    "OpenMAGViT2_orig_res": "OpenMAGViT2_IBQ.src.Open_MAGVIT2.models.lfqgan_pretrain"
}


class OpenMAGViT2(Tokenizer):
    """OpenMAGViT2 discrete image tokenizer"""
    
    def __init__(self, 
                 config_file: str,
                 ckpt_path: str,
                 name: str = "OpenMAGViT2_256",
                 **kwargs):
        
        if name not in MODEL_MODULES:
            raise ValueError(f"Unknown model name '{name}'. Available: {list(MODEL_MODULES.keys())}")
        
        module_path = MODEL_MODULES[name]
        module = __import__(module_path, fromlist=['VQModel'])
        self.VQModel = module.VQModel

        self.config_file = config_file
        self.ckpt_path = ckpt_path
        self.name = name
        
        super().__init__(**kwargs)
    
    def _load_model(self) -> None:
        """Load the OpenMAGViT2 model"""
        # Load model configuration
        config_model = load_config(self.config_file, display=False)
        
        # Initialize model
        self.model = self.VQModel(**config_model.model.init_args)
        
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
    

def reconstruction_orig_resolution(tokenizer, images, processing_ratio):
    if processing_ratio == 1.0:
        print(f"PROCESSING WITH TILING AT FULL RESOLUTION (no tile resize)")
        RECONSTRUCTION_PATH = f'/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/{tokenizer.name}_orig'
    else:
        print(f"PROCESSING WITH TILING RATIO: {processing_ratio}")
        print(f"EFFECTIVE AREA RATIO: {processing_ratio**2:.3f}")
        RECONSTRUCTION_PATH = f'/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/{tokenizer.name}_orig_ratio_{processing_ratio**2:.3f}'
    os.makedirs(RECONSTRUCTION_PATH, exist_ok=True)
    print(f"{'='*80}")

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


def reconstruction_tiled(tokenizer, processing_ratio, tile_resize, images, image_names):
    from Tiler import Tiler

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
    tiler = Tiler(tile_size=tile_orig_size, pad_value=-1.0, tile_resize=tile_resize)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 8

    print(f"Found {len(images)} images to process")
    print(f"Using tiler with tile_size={tiler.tile_size}")
    print(f"Processing tiles in batches of {batch_size}")

    # Process each image with tiling
    with torch.no_grad():
        for idx, (image, name) in enumerate(zip(images, image_names)):
            print(f"\n{'='*60}")
            print(f"Processing image {idx+1}/{len(images)}: {name}")
            print(f"{'='*60}")

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

            # Show results
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))

            # Original image
            axes[0].imshow(original_pil)
            axes[0].set_title(f"Original: {name}")
            axes[0].axis('off')

            # Reconstructed image
            axes[1].imshow(reconstructed_pil)
            axes[1].set_title(f"Reconstructed: {name_without_ext}_tiled_{total_tokens}")
            axes[1].axis('off')

            plt.tight_layout()
            plt.show()

            # Show all tiles in grid format
            print("Displaying all tiles in grid:")
            tiler.show_tiles(tiles.cpu(), metadata)

            # Clean up memory
            del tiles, reconstructed_tiles, all_indices_tensor, image_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n✅ Processing complete!")


if __name__ == "__main__":
    from utils import load_all_images, resize_by_ratio

    tokenizer_res_orig = OpenMAGViT2(
        config_file="/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/OpenMAGViT2_IBQ/configs/Open-MAGVIT2/gpu/pretrain_lfqgan_256_262144.yaml",
        ckpt_path="/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/OpenMAGViT2_IBQ/upload_ckpts/Open-MAGVIT2/pretrain_256_262144/pretrain256_262144.ckpt",
        name="OpenMAGViT2_orig_res",
    )

    tokenizer_res_256 = OpenMAGViT2(
        config_file="/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/OpenMAGViT2_IBQ/configs/Open-MAGVIT2/gpu/imagenet_lfqgan_128_L.yaml",
        ckpt_path="/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/OpenMAGViT2_IBQ/upload_ckpts/Open-MAGVIT2/imagenet_128_L.ckpt",
        name="OpenMAGViT2_256",
    )

    # Load images
    images, image_names, image_paths = load_all_images('/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/original')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processing_ratios = [1.0]

    for processing_ratio in processing_ratios:
        
        # reconstruction_orig_resolution(tokenizer_res_orig, images, processing_ratio)
        reconstruction_tiled(tokenizer_res_256, processing_ratio, tile_resize=256, images=images[1:], image_names=image_names[1:])