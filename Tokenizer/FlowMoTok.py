import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add base directory (for utils.py, Tiler.py) and FlowMo directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, "repos", "FlowMo"))

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from omegaconf import OmegaConf
from flowmo import train_utils
from typing import Tuple, Any
from Tokenizer.base import Tokenizer
import matplotlib.pyplot as plt

class FlowMo(Tokenizer):
    """FlowMo tokenizer implementation"""
    
    def __init__(self, 
                 model_name: str = "flowmo_lo",
                 num_tokens: int = None,
                 device: str = "cuda",
                 **kwargs):
        self.model_name = model_name
        self.num_tokens = num_tokens  # Manual override if provided
        self.device = device
        
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
        self._load_model()
    
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

if __name__ == "__main__":
    from utils import load_all_images
    from Tiler import Tiler
    # Example usage
    # tokenizer = FlowMo(model_name="flowmo_hi", device="cuda", num_tokens=1024)
    tokenizer = FlowMo(model_name="flowmo_lo", device="cuda", num_tokens=256)

    processing_ratios = [1.0, 0.9, 0.8, 0.7]
    tile_resize = 256
    batch_size = 8
    
    # Load images
    images, image_names, image_paths = load_all_images('/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/original')
    
    print(f"Found {len(images)} images to process")
    print(f"Processing ratios: {processing_ratios}")
    print(f"Tile resize: {tile_resize}")
    print(f"FlowMo Model: {tokenizer.model_name}")
    print(f"Code length: {tokenizer.model.code_length}")
    
    # Process each ratio
    for processing_ratio in processing_ratios:
        print(f"\n{'='*80}")
        if processing_ratio == 1.0:
            print(f"PROCESSING WITH TILING AT FULL RESOLUTION (no tile resize)")
            tile_orig_size = tile_resize  # No resize, so orig_size = resize
            RECONSTRUCTION_PATH = f'/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/{tokenizer.model_name}_tiled_{tile_resize}'
        else:
            tile_orig_size = round(tile_resize / processing_ratio)
            print(f"PROCESSING WITH TILING RATIO: {processing_ratio}")
            print(f"TILE ORIG SIZE: {tile_orig_size} -> TILE RESIZE: {tile_resize}")
            print(f"EFFECTIVE AREA RATIO: {processing_ratio**2:.3f}")
            RECONSTRUCTION_PATH = f'/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/{tokenizer.model_name}_tiled_{tile_resize}_ratio_{processing_ratio**2:.3f}'
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

                print(f"Encoded codes shape: {all_indices_tensor.shape}")

                # Calculate tokens and compression ratio
                # For FlowMo, use the number of tokens per tile times number of tiles
                tokens_per_tile = tokenizer.get_num_tokens(all_indices_tensor)
                total_tokens = tokens_per_tile * total_tiles
                original_pixels = image_tensor.shape[-2] * image_tensor.shape[-1]
                compression_ratio = original_pixels / total_tokens

                print(f"Tokens per tile: {tokens_per_tile}")
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
                # try:
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


    