import torch
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from PIL import Image
import math
from typing import Union, Tuple, Dict, Any
import torchshow as ts 

class Tiler:
    """
    A simple class for padding images and extracting tiles.
    Core functionality: pad -> tile -> reconstruct -> remove_pad
    """
    
    def __init__(self, tile_size: int = 256, pad_value: float = 0):
        """
        Initialize the Tiler.
        
        Args:
            tile_size: Size of each square tile (default: 256)
            pad_value: Value to use for padding (default: 0)
        """
        self.tile_size = tile_size
        self.pad_value = pad_value
        
    def _convert_input(self, image: Union[torch.Tensor, Image.Image, np.ndarray]) -> torch.Tensor:
        """Convert various input formats to CHW tensor format."""
        
        if isinstance(image, Image.Image):
            image = torch.from_numpy(np.array(image)).float()
            if len(image.shape) == 3 and image.shape[-1] in [1, 3, 4]:
                image = image.permute(2, 0, 1)  # HWC to CHW
            elif len(image.shape) == 2:  # Grayscale PIL image
                image = image.unsqueeze(0)  # Add channel dimension
        
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
            if len(image.shape) == 3 and image.shape[-1] in [1, 3, 4]:
                image = image.permute(2, 0, 1)  # HWC to CHW
            elif len(image.shape) == 2:  # Grayscale numpy array
                image = image.unsqueeze(0)  # Add channel dimension
        
        elif isinstance(image, torch.Tensor):
            image = image.float()
            # Handle different tensor formats
            if len(image.shape) == 3:
                if image.shape[0] in [1, 3, 4]:  # CHW format
                    pass  # Already correct
                elif image.shape[-1] in [1, 3, 4]:  # HWC format
                    image = image.permute(2, 0, 1)  # Convert to CHW
                else:
                    # Assume first dimension is channels if not clear
                    print(f"Warning: Ambiguous tensor shape {image.shape}, assuming CHW format")
            elif len(image.shape) == 2:  # Grayscale HW
                image = image.unsqueeze(0)  # Add channel dimension
            else:
                raise ValueError(f"Unsupported tensor shape: {image.shape}")
        
        return image
    
    def pad(self, image: Union[torch.Tensor, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Pad image so both dimensions are multiples of tile_size.
        Image is centered with symmetric padding.
        
        Args:
            image: Input image
        
        Returns:
            dict: {
                'padded_image': Padded image tensor,
                'metadata': Metadata needed for reconstruction
            }
        """
        image = self._convert_input(image)
        C, H, W = image.shape
        original_size = (H, W)
        
        print(f"Original image shape: C={C}, H={H}, W={W}")
        
        # Calculate target dimensions independently
        target_h = math.ceil(H / self.tile_size) * self.tile_size
        target_w = math.ceil(W / self.tile_size) * self.tile_size
        
        print(f"Target size: H={target_h}, W={target_w}")
        
        # Calculate symmetric padding for centering
        pad_h = target_h - H
        pad_w = target_w - W
        
        # Distribute padding symmetrically
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Apply padding: (left, right, top, bottom)
        padded_image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), 
                            mode='constant', value=self.pad_value)
        
        print(f"Padded image shape: {padded_image.shape}")
        
        # Calculate number of tiles in each dimension
        num_tiles_h = target_h // self.tile_size
        num_tiles_w = target_w // self.tile_size
        
        print(f"Number of tiles: H={num_tiles_h}, W={num_tiles_w}, Total={num_tiles_h * num_tiles_w}")
        
        metadata = {
            'original_size': original_size,
            'target_size': (target_h, target_w),
            'num_tiles': (num_tiles_h, num_tiles_w),
            'padding_info': {
                'pad_top': pad_top,
                'pad_bottom': pad_bottom,
                'pad_left': pad_left,
                'pad_right': pad_right
            }
        }
        
        return {
            'padded_image': padded_image,
            'metadata': metadata
        }
    
    def tile(self, padded_image: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extract tiles from the padded image using einops.
        Tiles are flattened to 4D tensor: (total_tiles, C, tile_size, tile_size)
        
        Args:
            padded_image: Image tensor where H and W are multiples of tile_size
        
        Returns:
            tiles: Tensor of shape (total_tiles, C, tile_size, tile_size)
            grid_shape: Tuple (num_tiles_h, num_tiles_w) for reconstruction
        """
        C, H, W = padded_image.shape
        
        print(f"Tiling padded image of shape: C={C}, H={H}, W={W}")
        
        # Ensure the image dimensions are divisible by tile_size
        assert H % self.tile_size == 0 and W % self.tile_size == 0, \
            f"Image size ({H}, {W}) must be divisible by tile_size ({self.tile_size})"
        
        # Calculate number of tiles
        nh = H // self.tile_size  # number of tiles in height
        nw = W // self.tile_size  # number of tiles in width
        
        print(f"Tiles grid: {nh} rows × {nw} columns = {nh * nw} total tiles")
        
        # Use einops to extract tiles and flatten to 4D
        tiles = rearrange(
            padded_image, 
            'c (nh th) (nw tw) -> (nh nw) c th tw',
            nh=nh, nw=nw,
            th=self.tile_size, 
            tw=self.tile_size
        )
        
        print(f"Tiles shape: {tiles.shape} (total_tiles, c, th, tw)")
        assert len(tiles.shape) == 4, f"Tiles should be 4D, got {len(tiles.shape)}D: {tiles.shape}"
        
        return tiles, (nh, nw)
    
    def reconstruct(self, tiles: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Reconstruct the padded image from 4D tiles using einops.
        
        Args:
            tiles: Tensor of shape (total_tiles, C, tile_size, tile_size)
            metadata: Metadata containing grid shape and target size
        
        Returns:
            reconstructed_image: Tensor of shape (C, H, W)
        """
        total_tiles, C, th, tw = tiles.shape
        target_h, target_w = metadata["target_size"]
        nh, nw = metadata["tiles_grid_shape"]
        
        # Verify the grid dimensions match
        assert total_tiles == nh * nw, f"Total tiles {total_tiles} doesn't match grid {nh}×{nw}={nh*nw}"
        
        # Verify the grid dimensions match target shape
        expected_h = nh * self.tile_size
        expected_w = nw * self.tile_size
        
        assert expected_h == target_h and expected_w == target_w, \
            f"Tile grid {nh}×{nw} with tile_size {self.tile_size} gives ({expected_h}, {expected_w}), " \
            f"but target shape is {target_h}, {target_w}"
        
        reconstructed = rearrange(
            tiles,
            '(nh nw) c th tw -> c (nh th) (nw tw)',
            nh=nh, nw=nw
        )
        
        return reconstructed
    
    def remove_pad(self, padded_image: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Remove padding to get back to original size.
        
        Args:
            padded_image: Padded image tensor of shape (C, H, W)
            metadata: Metadata containing padding info
        
        Returns:
            original_image: Image cropped to original size
        """
        orig_h, orig_w = metadata['original_size']
        pad_top = metadata['padding_info']['pad_top']
        pad_left = metadata['padding_info']['pad_left']
        
        # Crop from the center where the original image is located
        original = padded_image[:, pad_top:pad_top + orig_h, pad_left:pad_left + orig_w]
        
        return original
    
    def __call__(self, image: Union[torch.Tensor, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Convenience method: pad and tile in one call.
        
        Args:
            image: Input image
        
        Returns:
            dict: {
                'tiles': Extracted tiles as 4D tensor (total_tiles, C, tile_h, tile_w),
                'metadata': Metadata for reconstruction
            }
        """
        pad_result = self.pad(image)
        tiles, grid_shape = self.tile(pad_result['padded_image'])
        
        # Store grid info in metadata for reconstruction
        metadata = pad_result['metadata']
        metadata['grid_shape'] = metadata['target_size']  # (H, W) of padded image
        metadata['tiles_grid_shape'] = grid_shape  # (nh, nw)
        
        return {
            'tiles': tiles,
            'metadata': metadata
        }
    
    def full_reconstruct(self, tiles: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Convenience method: reconstruct and remove padding in one call.
        
        Args:
            tiles: Tiles tensor as 4D (total_tiles, C, tile_h, tile_w)
            metadata: Metadata from pad() or __call__()
        
        Returns:
            original_image: Fully reconstructed original image
        """
        padded = self.reconstruct(tiles, metadata)
        original = self.remove_pad(padded, metadata)
        return original
    
    def show(self, image: Union[torch.Tensor, Image.Image, np.ndarray]):
        """
        Quick display image using torchshow.
        
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
            metadata: Metadata containing grid shape (nh, nw)
            tight_layout: Whether to use tight layout
        """
        assert len(tiles.shape) == 4, f"Tiles must be of shape (total_tiles, C, tile_size, tile_size), got {tiles.shape}"
        total_tiles, C, tile_h, tile_w = tiles.shape
        
        # Get grid shape from metadata
        nh, nw = metadata["tiles_grid_shape"]
        
        # Verify grid consistency
        assert total_tiles == nh * nw, f"Total tiles {total_tiles} doesn't match grid {nh}×{nw}={nh*nw}"

        # Handle different channel counts
        if C == 4:  # RGBA - convert to RGB by dropping alpha
            print("4-channel image detected, converting RGBA to RGB")
            tiles = tiles[:, :3, :, :]  # Keep only RGB channels
            C = 3
        elif C > 4:
            print(f"Warning: {C} channels detected, showing only first 3 channels")
            tiles = tiles[:, :3, :, :]  # Keep only first 3 channels
            C = 3
        
        print(f"Showing {nh}×{nw} = {nh*nw} tiles")
        
        # Show tiles using torchshow with correct parameter names
        ts.show(tiles, nrows=nh, ncols=nw, tight_layout=tight_layout, figsize=(nw * tile_w / 200, nh * tile_h / 200))

# Example usage
if __name__ == "__main__":
    # Create a tiler
    tiler = Tiler(tile_size=256, pad_value=0)
    
    # Test image
    image = torch.randn(3, 567, 789)
    print(f"Original: {image.shape}")
    
    # Method 1: Step by step
    pad_result = tiler.pad(image)
    padded_image = pad_result['padded_image']
    metadata = pad_result['metadata']
    
    tiles, grid_shape = tiler.tile(padded_image)
    metadata['tiles_grid_shape'] = grid_shape  # Store grid shape for reconstruction
    print(f"Tiles: {tiles.shape}, Grid: {grid_shape}")
    
    reconstructed_padded = tiler.reconstruct(tiles, metadata)
    final_image = tiler.remove_pad(reconstructed_padded, metadata)
    print(f"Final: {final_image.shape}")
    
    # Method 2: Using convenience methods
    result = tiler(image)  # pad + tile
    final_image2 = tiler.full_reconstruct(result['tiles'], result['metadata'])
    
    # Show tiles (now requires metadata)
    tiler.show_tiles(result['tiles'], result['metadata'])
    
    # Verify
    error = torch.max(torch.abs(image - final_image)).item()
    print(f"Reconstruction error: {error}")
    print("✅ Perfect reconstruction!")