import os
import sys
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, "FQGAN"))

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional, Union, Dict, Any
import torchvision.transforms as T
from FQGAN.tokenizer.vq_model_triple import VQ_models
from Tokenizer.base import Tokenizer

class FQGAN(Tokenizer):
    """FQGAN Tokenizer using triple VQ model architecture"""
    
    def __init__(self, 
                 vq_model: str = "VQ-16",
                 vq_ckpt: str = None,
                 codebook_size: int = 16384,
                 codebook_embed_dim: int = 8,
                 image_size: int = 256,
                 with_clip_supervision: bool = False,
                 with_disentanglement: bool = False,
                 disentanglement_ratio: float = 0.1,
                 **kwargs):
        """
        Initialize FQGAN tokenizer
        
        Args:
            vq_model: VQ model architecture name
            vq_ckpt: Path to VQ model checkpoint
            codebook_size: Size of the codebook
            codebook_embed_dim: Embedding dimension of codebook
            image_size: Input image size
            with_clip_supervision: Whether to use CLIP supervision
            with_disentanglement: Whether to use disentanglement
            disentanglement_ratio: Disentanglement ratio
        """
        self.name = f"FQGAN-{vq_model}"  # More descriptive name
        self.vq_model_name = vq_model    # For VQ_models lookup
        self.vq_ckpt = vq_ckpt
        self.codebook_size = codebook_size
        self.codebook_embed_dim = codebook_embed_dim
        self.image_size = image_size
        self.with_clip_supervision = with_clip_supervision
        self.with_disentanglement = with_disentanglement
        self.disentanglement_ratio = disentanglement_ratio
        
        super().__init__(**kwargs)
    
    def _load_model(self) -> None:
        """Load the FQGAN VQ model"""
        
        # Create VQ model and assign to self.model for base class compatibility
        self.model = VQ_models[self.vq_model_name](
            codebook_size=self.codebook_size,
            codebook_embed_dim=self.codebook_embed_dim,
            with_clip_supervision=self.with_clip_supervision,
            with_disentanglement=self.with_disentanglement,
            disentanglement_ratio=self.disentanglement_ratio,
        )

        # Load checkpoint if provided
        if self.vq_ckpt:
            checkpoint = torch.load(self.vq_ckpt, map_location="cpu")
            
            # Handle different checkpoint formats
            if "ema" in checkpoint:  # ema
                model_weight = checkpoint["ema"]
            elif "model" in checkpoint:  # ddp
                model_weight = checkpoint["model"]
            elif "state_dict" in checkpoint:
                model_weight = checkpoint["state_dict"]
            else:
                raise Exception("Checkpoint format not recognized. Expected 'ema', 'model', or 'state_dict' keys.")
            
            self.model.load_state_dict(model_weight)
            del checkpoint
        
        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Disable gradients
        torch.set_grad_enabled(False)

        # Print parameter count automatically when model is loaded
        self.get_params()
       
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image to tensor format expected by FQGAN"""
        # Convert PIL image to numpy array (assuming it's already the right size from your tiler)
        arr = np.array(image)
        
        # Convert to tensor using torch.tensor (avoids numpy dependency issues)
        tensor = torch.tensor(arr, dtype=torch.float32)
        
        # Normalize from [0, 255] to [0, 1] then to [-1, 1]
        tensor = tensor / 255.0
        tensor = (tensor - 0.5) / 0.5
        
        # Add batch dimension and rearrange to CHW format
        if len(tensor.shape) == 3:
            tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor.to(self.device)
    
    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """Postprocess tensor back to PIL image"""
        # Remove batch dimension if present
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Convert from CHW to HWC
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        
        # Clamp and convert to uint8
        tensor = torch.clamp(127.5 * tensor + 128.0, 0, 255)
        tensor = tensor.to(torch.uint8).cpu().numpy()
        
        # Convert to PIL Image
        return Image.fromarray(tensor)
    
    def encode(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Encode tensor to discrete tokens using triple VQ"""
        with torch.no_grad():
            # Encode using triple VQ model (using self.model)
            latent_vis, latent_sem_mid, latent_sem_high, \
            [_, _, indices_vis], [_, _, indices_sem_mid], [_, _, indices_sem_high] = self.model.encode(tensor)
            
            # Combine all indices
            indices = {
                'indices_vis': indices_vis,
                'indices_sem_mid': indices_sem_mid,
                'indices_sem_high': indices_sem_high
            }
            
            # Store additional info needed for decoding
            additional_info = {
                'latent_shapes': {
                    'vis_shape': latent_vis.shape,
                    'sem_mid_shape': latent_sem_mid.shape,
                    'sem_high_shape': latent_sem_high.shape
                }
            }
            
            return indices, additional_info
    
    def decode(self, indices: Union[torch.Tensor, Dict[str, torch.Tensor]], additional_info: Dict[str, Any] = None) -> torch.Tensor:
        """Decode discrete tokens back to tensor"""
        with torch.no_grad():
            if isinstance(indices, dict):
                # Extract individual indices
                indices_vis = indices['indices_vis']
                indices_sem_mid = indices['indices_sem_mid']
                indices_sem_high = indices['indices_sem_high']
                
                # Extract latent shapes
                latent_shapes = additional_info['latent_shapes']
                vis_shape = latent_shapes['vis_shape']
                sem_mid_shape = latent_shapes['sem_mid_shape']
                sem_high_shape = latent_shapes['sem_high_shape']
                
                # Decode using VQ model (using self.model)
                reconstructed = self.model.decode_code(
                    indices_vis, indices_sem_mid, indices_sem_high,
                    vis_shape, sem_mid_shape, sem_high_shape
                )
            else:
                raise ValueError("FQGAN requires indices as dictionary with 'indices_vis', 'indices_sem_mid', 'indices_sem_high' keys")
            
            return reconstructed
    
    def get_num_tokens(self, indices: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> int:
        """Get total number of tokens across all quantizers"""
        if isinstance(indices, dict):
            total_tokens = 0
            for key, idx in indices.items():
                if isinstance(idx, torch.Tensor):
                    total_tokens += idx.numel()
                else:
                    total_tokens += len(idx.flatten()) if hasattr(idx, 'flatten') else len(idx)
            return total_tokens
        else:
            return indices.numel() if isinstance(indices, torch.Tensor) else len(indices)
    
    def get_codebook_usage(self, indices: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, int]]:
        """Get codebook usage statistics for each quantizer"""
        usage_stats = {}
        
        for key, idx in indices.items():
            if isinstance(idx, torch.Tensor):
                idx_flat = idx.flatten()
                unique_tokens = torch.unique(idx_flat)
                usage_stats[key] = {
                    'unique_tokens': len(unique_tokens),
                    'total_tokens': len(idx_flat),
                    'usage_ratio': len(unique_tokens) / self.codebook_size,
                    'most_frequent': torch.mode(idx_flat)[0].item(),
                    'token_range': (idx_flat.min().item(), idx_flat.max().item())
                }
        
        return usage_stats
    
    def reconstruct(self, image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        """Full reconstruction pipeline with FQGAN-specific metrics"""
        # Call parent reconstruct method
        reconstructed_image, metrics = super().reconstruct(image)
        
        # Add FQGAN-specific metrics
        if isinstance(metrics['additional_info'], dict) and 'latent_shapes' in metrics['additional_info']:
            # Get indices for codebook usage analysis
            input_tensor = self.preprocess(image)
            indices, _ = self.encode(input_tensor)
            
            # Add codebook usage statistics
            if isinstance(indices, dict):
                metrics['codebook_usage'] = self.get_codebook_usage(indices)
                
                # Add per-quantizer token counts
                metrics['tokens_per_quantizer'] = {
                    key: idx.numel() if isinstance(idx, torch.Tensor) else len(idx)
                    for key, idx in indices.items()
                }
        
        return reconstructed_image, metrics


def print_metrics(name, total_tokens, total_tokens_vis, total_tokens_sem_mid, total_tokens_sem_high, 
                 compression_ratio, original_pixels, processing_ratio=None):
    """Print comprehensive metrics for the processed image"""
    print(f"\n📊 METRICS for {name}:")
    print(f"  💾 Total tokens: {total_tokens:,}")
    print(f"     ├─ Visual tokens: {total_tokens_vis:,}")
    print(f"     ├─ Semantic mid tokens: {total_tokens_sem_mid:,}")
    print(f"     └─ Semantic high tokens: {total_tokens_sem_high:,}")
    print(f"  🖼️  Original pixels: {original_pixels:,}")
    print(f"  📈 Compression ratio: {compression_ratio:.2f}x")
    if processing_ratio is not None and processing_ratio != 1.0:
        print(f"  🔍 Processing ratio: {processing_ratio} (area: {processing_ratio**2:.3f})")


def setup_environment():
    """Set up environment variables for threading"""
    os.environ['OMP_NUM_THREADS'] = '16'
    os.environ['MKL_NUM_THREADS'] = '16'
    os.environ['OPENBLAS_NUM_THREADS'] = '16'
    os.environ['NUMEXPR_NUM_THREADS'] = '16'


def create_tiler(processing_ratio, tile_resize):
    """Create tiler based on processing ratio"""
    from Tiler import Tiler

    if processing_ratio == 1.0:
        tile_orig_size = tile_resize  # No resize, so orig_size = resize
        return Tiler(tile_size=tile_orig_size, pad_value=-1.0), tile_orig_size  # No resize
    else:
        tile_orig_size = round(tile_resize / processing_ratio)
        return Tiler(tile_size=tile_orig_size, pad_value=-1.0, tile_resize=tile_resize), tile_orig_size


def process_single_image(tokenizer, tiler, image, name, batch_size, processing_ratio, reconstruction_path):
    """Process a single image through the tokenization pipeline"""
    print(f"\n{'-'*60}")
    print(f"Processing image: {name} (ratio {processing_ratio})")
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

    # Process tiles in smaller batches
    total_tiles = tiles.shape[0]
    reconstructed_tiles_list = []
    all_indices_vis = []
    all_indices_sem_mid = []
    all_indices_sem_high = []
    all_additional_info = []

    print(f"🔄 Processing {total_tiles} tiles in batches of {batch_size}")

    for i in range(0, total_tiles, batch_size):
        end_idx = min(i + batch_size, total_tiles)
        batch_tiles = tiles[i:end_idx].to(tokenizer.device)

        print(f"  📦 Processing batch {i//batch_size + 1}/{(total_tiles + batch_size - 1)//batch_size}: tiles {i+1}-{end_idx}")

        # Process this batch through the tokenizer
        indices, additional_info = tokenizer.encode(batch_tiles)
        reconstructed_batch = tokenizer.decode(indices, additional_info)

        # Store indices from all three codebooks
        all_indices_vis.append(indices['indices_vis'].cpu())
        all_indices_sem_mid.append(indices['indices_sem_mid'].cpu())
        all_indices_sem_high.append(indices['indices_sem_high'].cpu())
        all_additional_info.append(additional_info)

        # Move to CPU and clamp
        reconstructed_batch = reconstructed_batch.clamp(-1, 1).cpu()
        reconstructed_tiles_list.append(reconstructed_batch)

        # Clear GPU memory
        del batch_tiles, reconstructed_batch, indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Reconstruct full image
    print("🔧 Reconstructing full image from processed tiles...")
    reconstructed_tiles = torch.cat(reconstructed_tiles_list, dim=0)

    # Combine all indices from all batches
    all_indices_combined = {
        'indices_vis': torch.cat(all_indices_vis, dim=0),
        'indices_sem_mid': torch.cat(all_indices_sem_mid, dim=0),
        'indices_sem_high': torch.cat(all_indices_sem_high, dim=0)
    }

    # Calculate total tokens across all three codebooks
    total_tokens_vis = all_indices_combined['indices_vis'].numel()
    total_tokens_sem_mid = all_indices_combined['indices_sem_mid'].numel()
    total_tokens_sem_high = all_indices_combined['indices_sem_high'].numel()
    total_tokens = total_tokens_vis + total_tokens_sem_mid + total_tokens_sem_high

    original_pixels = image_tensor.shape[-2] * image_tensor.shape[-1]
    compression_ratio = original_pixels / total_tokens

    # Reconstruct and postprocess
    reconstructed_full = tiler.full_reconstruct(reconstructed_tiles, metadata)

    # Convert to PIL for saving using tokenizer's postprocess
    reconstructed_pil = tokenizer.postprocess(reconstructed_full.unsqueeze(0))

    # Create filename with token count
    name_without_ext = os.path.splitext(name)[0]
    output_filename = f"{name_without_ext}_tiled_{total_tokens}.png"
    output_path = os.path.join(reconstruction_path, output_filename)

    # Save the reconstructed image
    reconstructed_pil.save(output_path)
    print(f"  💾 Saved: {output_filename}")
    
    # Print metrics
    print_metrics(name, total_tokens, total_tokens_vis, total_tokens_sem_mid, 
                 total_tokens_sem_high, compression_ratio, original_pixels, processing_ratio)

    # Clean up memory
    del tiles, reconstructed_tiles, all_indices_combined, image_tensor, reconstructed_full
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Setup environment
    setup_environment()
    
    # Import here to avoid circular imports when used as module
    from Tiler import Tiler
    from utils import load_all_images

    # Initialize tokenizer
    tokenizer = FQGAN(
        vq_model="VQ-16",
        vq_ckpt="/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/FQGAN/checkpoints/fqgan_triple_ds16.pt",
        codebook_size=16384,
        codebook_embed_dim=8,
        image_size=256,
        with_clip_supervision=True,
        with_disentanglement=False,
        disentanglement_ratio=0.1
    )

    # Configuration
    processing_ratios = [1.0, 0.9, 0.8, 0.7]
    tile_resize = 256
    batch_size = 8

    # Load images
    image_folder = '/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/original'
    print(f"Loading images from: {image_folder}")
    
    try:
        images, image_names, image_paths = load_all_images(image_folder)
        print(f"Successfully loaded {len(images)} images")
    except Exception as e:
        print(f"Error loading images: {e}")
        sys.exit(1)

    # Process each ratio
    for processing_ratio in processing_ratios:
        print(f"\n{'='*80}")
        if processing_ratio == 1.0:
            print(f"PROCESSING WITH TILING AT FULL RESOLUTION (no tile resize)")
            RECONSTRUCTION_PATH = f'/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/fqgan_triple_tiled_{tile_resize}'
        else:
            print(f"PROCESSING WITH TILING RATIO: {processing_ratio}")
            tile_orig_size = round(tile_resize / processing_ratio)
            print(f"TILE ORIG SIZE: {tile_orig_size} -> TILE RESIZE: {tile_resize}")
            print(f"EFFECTIVE AREA RATIO: {processing_ratio**2:.3f}")
            RECONSTRUCTION_PATH = f'/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/fqgan_triple_tiled_{tile_resize}_ratio_{processing_ratio**2:.3f}'
        print(f"OUTPUT PATH: {RECONSTRUCTION_PATH}")
        print(f"{'='*80}")
        
        # Create output directory
        os.makedirs(RECONSTRUCTION_PATH, exist_ok=True)
        
        # Initialize tiler for current ratio
        tiler, tile_orig_size = create_tiler(processing_ratio, tile_resize)
        
        print(f"Using tiler with tile_size={tiler.tile_size}, tile_resize={getattr(tiler, 'tile_resize', 'None')}")
        
        # Process each image with current ratio
        with torch.no_grad():
            for idx, (image, name) in enumerate(zip(images, image_names)):
                try:
                    process_single_image(
                        tokenizer, tiler, image, name, batch_size, processing_ratio, RECONSTRUCTION_PATH
                    )
                except Exception as e:
                    print(f"❌ Error processing {name}: {e}")
                    continue

    print(f"\n🎉 All processing completed!")
    print(f"Check the output directories for reconstructed images.")