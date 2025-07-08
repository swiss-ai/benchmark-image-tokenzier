import torch
import numpy as np
from PIL import Image
from typing import Tuple, Any
from torchvision import transforms
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils_benchmark import load_all_images
from pathlib import Path
from torchvision.utils import make_grid


from Tiler import Tiler
from base import Tokenizer

os.chdir('/users/nirmiger/DetailFlow')
sys.path.append('/users/nirmiger/DetailFlow')

from inference.load_vq import load_vq_model

TOKENIZER = 'datailflow_256'
if TOKENIZER == 'datailflow':
    TOKENIZER_PATH = '/iopsstor/scratch/cscs/nirmiger/512.pt'
    CONFIG_PATH = '/users/nirmiger/DetailFlow/config_512.json'
    YAML_PATH = '/users/nirmiger/DetailFlow/config_512.yaml'
elif TOKENIZER == 'datailflow_256':
    TOKENIZER_PATH = '/iopsstor/scratch/cscs/nirmiger/256.pt'
    CONFIG_PATH = '/users/nirmiger/DetailFlow/config.json'
    YAML_PATH = '/users/nirmiger/DetailFlow/config.yaml'

RECONSTRUCTION_PATH = f'/users/nirmiger/benchmark-image-tokenzier/assets/{TOKENIZER}'

class DetailFlowTokenizer(Tokenizer):
    """DetailFlow tokenizer implementation"""

    def __init__(self,
                 ckpt_path: str,
                 device: str = "cuda",
                 image_size: int = 256,
                 seed: int = 0,
                 use_ema: bool = False,
                 config_path: str = CONFIG_PATH,
                 yaml_path: str = YAML_PATH,
                 **kwargs):
        self.device = device
        self.image_size = image_size
        self.vq_model_path = ckpt_path
        self.use_ema = use_ema
        self.seed = seed
        self.config_path = config_path
        self.yaml_path = yaml_path

        torch.manual_seed(seed)
        torch.set_grad_enabled(False)

        # Preprocessing transform
        self.preprocess_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])

        super().__init__(**kwargs)

    def _load_model(self) -> None:
        """Load DetailFlow VQ model and config"""
        self.vq_model, self.config, self.config_yaml, self.res_deg = load_vq_model(
            self.vq_model_path,
            ema=self.use_ema,
            device=self.device,
            eval_mode=True,
            config_path=self.config_path,
            yaml_path=self.yaml_path,
        )
        # Move to GPU
        self.vq_model = self.vq_model.to(self.device)
        # Convert to fp16 if supported
        self.vq_model = self.vq_model.half()


    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image to tensor format"""
        tensor = self.preprocess_transform(image).unsqueeze(0).to(self.device)  # Shape: (1, 3, H, W)
        return tensor

    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor output back to a PIL image"""
        tensor = tensor.squeeze(0)  # Remove batch dimension
        img = torch.clamp(127.5 * tensor + 128, 0, 255).to(torch.uint8)
        img = img.permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(img)

    def encode(self, tensor: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Encode tensor into token indices"""
        with torch.no_grad():
            tokens, _, _ = self.vq_model.encode(tensor)
        return tokens, {}  # Add any metadata if needed

    def decode(self, indices: torch.Tensor, additional_info=None) -> torch.Tensor:
        """Decode token indices back into image tensor"""
        with torch.no_grad():
            output = self.vq_model.decode(indices)
        img_tensor = output.pixel_value
        return img_tensor

    def get_num_tokens(self, indices: torch.Tensor) -> int:
        """Get total number of tokens"""
        return int(indices.numel()/8) # 8 is the embedding dimension for DetailFlow

if __name__ == "__main__":
    # Example usage
    tokenizer = DetailFlowTokenizer(ckpt_path=TOKENIZER_PATH, config_path=CONFIG_PATH, yaml_path=YAML_PATH)
    tiler = Tiler(tile_size=256, pad_value=-1.0)
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

            reconstructed_tiles_list.append(reconstructed_batch)
            all_indices.append(indices.cpu())

            del batch_tiles, reconstructed_batch, indices
            torch.cuda.empty_cache()

        # Combine reconstructed tiles and indices
        reconstructed_tiles = torch.cat(reconstructed_tiles_list, dim=0)
        all_indices_tensor = torch.cat(all_indices, dim=0)

        # Reconstruct full image
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