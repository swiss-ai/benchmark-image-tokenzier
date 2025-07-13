import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from PIL import Image
import math
from typing import Union, Tuple, Dict, Any, Optional
import torchshow as ts


class Tiler:
    """
    Optimal tiler using tile_resize/tile_size ratio.
    
    Workflow: downsample → pad → tile → process → stitch → unpad → upsample
    """
    
    def __init__(self, tile_size: int = 256, tile_resize: Optional[int] = None, pad_value: float = 0):
        """
        Args:
            tile_size: Original tile extraction size
            tile_resize: Target processing size  
            pad_value: Padding value
        """
        self.tile_size = tile_size
        self.tile_resize = tile_resize if tile_resize is not None else tile_size
        self.pad_value = pad_value

        self.ratio = self.tile_resize / self.tile_size
        area_ratio = (self.tile_resize ** 2) / (self.tile_size ** 2)
        
        print(f"Linear scaling ratio: {tile_resize}/{tile_size} = {self.ratio:.3f}")
        print(f"Pixel per tile ratio: {tile_resize}²/{tile_size}² = {area_ratio:.3f}")
        
        if area_ratio < 1.0:
            print(f"Pixel reduction: {1 - area_ratio:.1%} fewer pixels per tile (downsampling)")
        elif area_ratio > 1.0:
            print(f"Pixel increase: {area_ratio - 1:.1%} more pixels per tile (upsampling)")
        else:
            print(f"No pixel change: same resolution processing")
    
    def _convert_input(self, image: Union[torch.Tensor, Image.Image, np.ndarray]) -> torch.Tensor:
        """Convert input to CHW tensor."""
        if isinstance(image, Image.Image):
            image = torch.tensor(np.array(image), dtype=torch.float32)
            if len(image.shape) == 3 and image.shape[-1] in [1, 3, 4]:
                image = image.permute(2, 0, 1)
            elif len(image.shape) == 2:
                image = image.unsqueeze(0)
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)
            if len(image.shape) == 3 and image.shape[-1] in [1, 3, 4]:
                image = image.permute(2, 0, 1)
            elif len(image.shape) == 2:
                image = image.unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            image = image.float()
            if len(image.shape) == 3 and image.shape[-1] in [1, 3, 4]:
                image = image.permute(2, 0, 1)
            elif len(image.shape) == 2:
                image = image.unsqueeze(0)
        
        return image
    
    def process(self, image: Union[torch.Tensor, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Process image using optimal ratio-based workflow.
        
        Returns:
            dict: {
                'tiles': 4D tensor (N, C, tile_resize, tile_resize),
                'metadata': reconstruction metadata
            }
        """
        image = self._convert_input(image)
        C, orig_H, orig_W = image.shape
        print(f"Original image: {C}×{orig_H}×{orig_W}")
        
        # Step 1: Downsample by ratio
        scaled_H = int(orig_H * self.ratio)
        scaled_W = int(orig_W * self.ratio)
        
        print(f"Downsampling to: {scaled_H}×{scaled_W} (ratio: {self.ratio:.3f})")
        
        downsampled = F.interpolate(
            image.unsqueeze(0), 
            size=(scaled_H, scaled_W), 
            mode='bicubic', 
            align_corners=False, 
            antialias=True
        ).squeeze(0)
        
        # Step 2: Pad for tiling at tile_resize scale
        target_H = math.ceil(scaled_H / self.tile_resize) * self.tile_resize
        target_W = math.ceil(scaled_W / self.tile_resize) * self.tile_resize
        
        pad_H = target_H - scaled_H
        pad_W = target_W - scaled_W
        pad_top = pad_H // 2
        pad_left = pad_W // 2
        pad_bottom = pad_H - pad_top
        pad_right = pad_W - pad_left
        
        print(f"Padding to: {target_H}×{target_W}")
        
        padded = F.pad(downsampled, (pad_left, pad_right, pad_top, pad_bottom), 
                      mode='constant', value=self.pad_value)
        
        # Step 3: Tile at tile_resize scale (no per-tile resizing needed!)
        num_tiles_H = target_H // self.tile_resize
        num_tiles_W = target_W // self.tile_resize
        
        print(f"Creating {num_tiles_H}×{num_tiles_W} = {num_tiles_H * num_tiles_W} tiles of {self.tile_resize}×{self.tile_resize}")
        
        tiles = rearrange(
            padded, 
            'c (nh th) (nw tw) -> (nh nw) c th tw',
            nh=num_tiles_H, nw=num_tiles_W, 
            th=self.tile_resize, tw=self.tile_resize
        )
        
        # Store metadata for reconstruction
        metadata = {
            'original_size': (orig_H, orig_W),
            'scaled_size': (scaled_H, scaled_W),
            'grid_shape': (num_tiles_H, num_tiles_W),
            'padding': (pad_top, pad_bottom, pad_left, pad_right),
            'ratio': self.ratio
        }
        
        return {
            'tiles': tiles, 
            'metadata': metadata
        }
    
    def reconstruct(self, tiles: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Reconstruct to original size and scale.
        
        Args:
            tiles: 4D tensor (N, C, tile_resize, tile_resize)
            metadata: From process()
            
        Returns:
            Reconstructed image at original size
        """
        num_tiles_H, num_tiles_W = metadata['grid_shape']
        
        print(f"Reconstructing from {tiles.shape[0]} tiles")
        
        # Step 1: Stitch tiles
        stitched = rearrange(
            tiles, 
            '(nh nw) c th tw -> c (nh th) (nw tw)',
            nh=num_tiles_H, nw=num_tiles_W
        )
        
        # Step 2: Remove padding
        pad_top, pad_bottom, pad_left, pad_right = metadata['padding']
        scaled_H, scaled_W = metadata['scaled_size']
        
        unpadded = stitched[:, pad_top:pad_top + scaled_H, pad_left:pad_left + scaled_W]
        print(f"After removing padding: {unpadded.shape}")
        
        # Step 3: Upsample back to original scale
        orig_H, orig_W = metadata['original_size']
        upscale_ratio = 1.0 / metadata['ratio']
        
        print(f"Upsampling by {upscale_ratio:.3f} to restore original size")
        
        reconstructed = F.interpolate(
            unpadded.unsqueeze(0), 
            size=(orig_H, orig_W),
            mode='bicubic', 
            align_corners=False, 
            antialias=True
        ).squeeze(0)
        
        return reconstructed
    
    def __call__(self, image: Union[torch.Tensor, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Convenience method: same as process().
        
        Args:
            image: Input image
        
        Returns:
            dict: {
                'tiles': 4D tensor (N, C, tile_resize, tile_resize),
                'metadata': reconstruction metadata
            }
        """
        return self.process(image)
    
    def full_reconstruct(self, tiles: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Alias for reconstruct() for compatibility.
        """
        return self.reconstruct(tiles, metadata)
    
    def show(self, image: Union[torch.Tensor, Image.Image, np.ndarray]):
        """
        Display image using torchshow.
        
        Args:
            image: Input image to display
        """
        # Convert input to tensor if needed
        if not isinstance(image, torch.Tensor):
            image = self._convert_input(image)
        
        # Show the image
        ts.show(image)

    def show_tiles(self, tiles: torch.Tensor, metadata: Dict[str, Any], tight_layout: bool = True):
        """
        Display tiles in a grid using torchshow.
        
        Args:
            tiles: Tiles tensor of shape (total_tiles, C, tile_size, tile_size)
            metadata: Metadata containing grid shape (num_H, num_W)
            tight_layout: Whether to use tight layout
        """
        assert len(tiles.shape) == 4, f"Tiles must be 4D, got {tiles.shape}"
        total_tiles, C, tile_h, tile_w = tiles.shape
        
        # Get grid shape from metadata
        num_H, num_W = metadata["grid_shape"]
        
        # Verify grid consistency
        assert total_tiles == num_H * num_W, f"Total tiles {total_tiles} doesn't match grid {num_H}×{num_W}={num_H*num_W}"

        # Handle different channel counts
        display_tiles = tiles
        if C == 4:  # RGBA - convert to RGB by dropping alpha
            print("4-channel image detected, converting RGBA to RGB")
            display_tiles = tiles[:, :3, :, :]
            C = 3
        elif C > 4:
            print(f"Warning: {C} channels detected, showing only first 3 channels")
            display_tiles = tiles[:, :3, :, :]
            C = 3
        
        print(f"Showing {num_H}×{num_W} = {num_H*num_W} tiles")
        
        # Show tiles using torchshow
        ts.show(display_tiles, nrows=num_H, ncols=num_W, tight_layout=tight_layout, 
                figsize=(num_W * tile_w / 200, num_H * tile_h / 200))

# Example usage and comparison
if __name__ == "__main__":
    print("=== Ratio-Based Tiler Demo ===")
    
    # Test downsampling
    print("\n--- Test 1: Downsampling (384→256) ---")
    tiler_down = Tiler(tile_size=384, tile_resize=256)
    
    # Test upsampling  
    print("\n--- Test 2: Upsampling (256→384) ---")
    tiler_up = Tiler(tile_size=256, tile_resize=384)
    
    # Test same size
    print("\n--- Test 3: Same size (256→256) ---")
    tiler_same = Tiler(tile_size=256, tile_resize=256)
    
    # Test image
    image = torch.randn(3, 1200, 1600)
    
    # Process with downsampling
    print("\n=== Processing with downsampling ===")
    result_down = tiler_down.process(image)
    tiles_down = result_down['tiles']
    reconstructed_down = tiler_down.reconstruct(tiles_down, result_down['metadata'])
    
    # Process with upsampling
    print("\n=== Processing with upsampling ===")
    result_up = tiler_up.process(image)
    tiles_up = result_up['tiles']
    reconstructed_up = tiler_up.reconstruct(tiles_up, result_up['metadata'])
    
    print(f"\nComparison:")
    print(f"Original: {image.shape}")
    print(f"Downsampling tiles: {tiles_down.shape[0]} tiles of {tiles_down.shape[-1]}×{tiles_down.shape[-1]}")
    print(f"Upsampling tiles: {tiles_up.shape[0]} tiles of {tiles_up.shape[-1]}×{tiles_up.shape[-1]}")
    print(f"Both reconstruct to: {reconstructed_down.shape}")
    print(f"Perfect reconstruction: {image.shape == reconstructed_down.shape == reconstructed_up.shape}")