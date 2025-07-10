import cv2
import numpy as np
import os
from PIL import Image
from typing import Tuple, Any
import torch
from torchvision import transforms

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils_benchmark import load_all_images
from pathlib import Path

from Tiler import Tiler
from Tokenizer.base import Tokenizer

os.chdir('/users/nirmiger/vila-u')
sys.path.append('/users/nirmiger/vila-u')

import vila_u

TOKENIZER_PATH = '/iopsstor/scratch/cscs/nirmiger/vila-u-7b-256'
TOKENIZER = 'vila-u-7b-256'

IMAGE_SIZE = 256
TILE_SIZE = 256

RECONSTRUCTION_PATH = f'/users/nirmiger/benchmark-image-tokenzier/assets/{TOKENIZER}_ratio_{(IMAGE_SIZE/TILE_SIZE)*(IMAGE_SIZE/TILE_SIZE)}'

class VILA_U_Tokenizer(Tokenizer):
    """UniTok tokenizer implementation"""

    def __init__(self,
                 model_path: str,
                 device: str = "cuda",
                 image_size: int = 256,
                 seed: int = 0,
                 **kwargs):
        self.model_path = model_path
        self.device = device
        self.image_size = image_size
        self.seed = seed

        torch.manual_seed(seed)
        torch.set_grad_enabled(False)

        super().__init__(**kwargs)

    def _load_model(self) -> None:
        """Load VILA-U model from checkpoint"""
        self.model = vila_u.load(self.model_path)
        self.vision_tower = self.model.get_vision_tower()

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image to tensor format expected by VILA-U"""
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to tensor and scales to [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
        tensor = transform(image).unsqueeze(0)
        return tensor

    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor output back to a PIL image"""
        img = tensor.to(torch.float32).add_(1).mul_(127.5).clamp_(0, 255)
        img = img.chunk(2)[0].detach().to(torch.float32).cpu().squeeze(0)
        img = img.permute(1, 2, 0).numpy().astype(np.uint8)
        return Image.fromarray(img)

    def encode(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Encode tensor into discrete token indices"""
        tensor = tensor.to(self.model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            code, tokens = self.vision_tower.vision_tower.rqvaesiglip.encode_image(tensor)
        return tokens, {}  # Additional info unused here

    def decode(self, indices: torch.Tensor, additional_info: Any = None) -> torch.Tensor:
        """Decode discrete tokens back into an image tensor"""
        with torch.no_grad():
            img_tensor = self.vision_tower.vision_tower.rqvaesiglip.decode(indices)
        return img_tensor

    def get_num_tokens(self, indices: torch.Tensor) -> int:
        """Get total number of tokens"""
        return int(indices.numel()/256)
    

if __name__ == "__main__":
    # Example usage
    tokenizer = VILA_U_Tokenizer(model_path=TOKENIZER_PATH, device='cuda', image_size=256)
    tiler = Tiler(tile_size=TILE_SIZE, pad_value=-1.0, tile_resize=IMAGE_SIZE)
    images, _, image_paths = load_all_images('/users/nirmiger/benchmark-image-tokenzier/assets/original')
    batch_size = 8  # Adjust based on GPU memory
    os.makedirs(RECONSTRUCTION_PATH, exist_ok=True)
    for idx, image_path in enumerate(image_paths):
        name = Path(image_path).name
        print(f"\nProcessing image {idx+1}: {name}")
        
        # Load and normalize image manually
        image = Image.open(image_path).convert("RGB")
        image_tensor = tokenizer.preprocess(image).squeeze(0)  # Shape: (3, H, W)
        print(f"Normalized image shape: {image_tensor.shape}")

        # Tile the image
        result = tiler(image_tensor)
        tiles = result['tiles']  # (N, C, H, W)
        metadata = result['metadata']
        print(f"Number of tiles: {tiles.shape[0]} | Tile shape: {tiles.shape[1:]}")

        total_tiles = tiles.shape[0]
        reconstructed_tiles_list = []
        all_indices = []

        print(f"Processing {total_tiles} tiles in batches of {batch_size}")

        for i in range(0, total_tiles, batch_size):
            end_idx = min(i + batch_size, total_tiles)
            batch_tiles = tiles[i:end_idx].to(tokenizer.device)

            print(f"Batch {i//batch_size + 1}: tiles {i}-{end_idx - 1}")

            with torch.no_grad():
                indices, additional_info = tokenizer.encode(batch_tiles)
                reconstructed_batch = tokenizer.decode(indices, additional_info)

            reconstructed_tiles_list.append(reconstructed_batch.cpu())
            all_indices.append(indices.cpu())

            del batch_tiles, reconstructed_batch, indices
            torch.cuda.empty_cache()

        # Combine reconstructed tiles and indices
        reconstructed_tiles = torch.cat(reconstructed_tiles_list, dim=0)
        all_indices_tensor = torch.cat(all_indices, dim=0)

        # Reconstruct full image
        reconstructed_tiles = reconstructed_tiles.to(torch.float32)
        reconstructed_full = tiler.full_reconstruct(reconstructed_tiles, metadata)

        # Convert to PIL
        reconstructed_image = tokenizer.postprocess(reconstructed_full.unsqueeze(0))

        # Metrics
        total_tokens = tokenizer.get_num_tokens(all_indices_tensor)
        original_pixels = image_tensor.shape[-2] * image_tensor.shape[-1]
        compression_ratio = original_pixels / total_tokens

        metrics = {
            'input_shape': image_tensor.shape,
            'num_tokens': total_tokens,
            'original_pixels': original_pixels,
            'compression_ratio': compression_ratio,
            'indices_shape': all_indices_tensor.shape,
            'num_tiles': total_tiles,
            'tile_size': tiler.tile_size
        }

        # Log
        print(f"Input image shape: {metrics['input_shape']}")
        print(f"Number of tokens: {metrics['num_tokens']}")
        print(f"Original image pixels: {metrics['original_pixels']}")
        print(f"Compression ratio: {metrics['compression_ratio']:.2f}x")
        print(f"Indices shape: {metrics['indices_shape']}")
        print(f"Number of tiles: {metrics['num_tiles']}")

        # Save output
        name_without_ext = Path(name).stem
        output_filename = f"{name_without_ext}_tiled_{metrics['num_tokens']}.png"
        output_path = os.path.join(RECONSTRUCTION_PATH, output_filename)
        reconstructed_image.save(output_path)
        print(f"Saved: {output_filename}")