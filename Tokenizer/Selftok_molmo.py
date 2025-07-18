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


from molmo_tiler import MultiModalPreprocessor
from base import Tokenizer

os.chdir('/users/nirmiger/SelftokTokenizer')
sys.path.append('/users/nirmiger/SelftokTokenizer')

from mimogpt.infer.SelftokPipeline import SelftokPipeline, NormalizeToTensor
from mimogpt.infer.infer_utils import parse_args_from_yaml

TOKENIZER = 'selftok_1024'
if TOKENIZER == 'selftok_512':
    TOKENIZER_PATH = '/iopsstor/scratch/cscs/nirmiger/renderer_512_ckpt.pth'
    CONFIG_PATH = '/users/nirmiger/SelftokTokenizer/configs/renderer/renderer-eval.yml'
elif TOKENIZER == 'selftok_1024':
    TOKENIZER_PATH = '/iopsstor/scratch/cscs/nirmiger/renderer_1024_ckpt.pth'
    CONFIG_PATH = '/users/nirmiger/SelftokTokenizer/configs/renderer/renderer-eval_1024.yml'

SD3_PATH = '/iopsstor/scratch/cscs/nirmiger/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671'
IMAGE_SIZE = 256
TILE_SIZE = 256

RECONSTRUCTION_PATH = f'/users/nirmiger/benchmark-image-tokenzier/assets/{TOKENIZER}_molmo'

class SelftokTokenizer(Tokenizer):
    """Selftok tokenizer implementation"""

    def __init__(self,
                 yml_path: str,
                 ckpt_path: str,
                 sd3_path: str,
                 device: str = "cuda",
                 image_size: int = 256,
                 seed: int = 0,
                 **kwargs):
        self.yml_path = yml_path
        self.ckpt_path = ckpt_path
        self.sd3_path = sd3_path
        self.device = device
        self.image_size = image_size
        self.seed = seed

        torch.manual_seed(seed)
        torch.set_grad_enabled(False)

        # Preprocessing transform
        self.preprocess_transform = transforms.Compose([
            NormalizeToTensor(),
        ])

        super().__init__(**kwargs)

    def _load_model(self) -> None:
        """Load Selftok model from checkpoint"""
        cfg = parse_args_from_yaml(self.yml_path)
        self.model = SelftokPipeline(cfg=cfg,
                                     ckpt_path=self.ckpt_path,
                                     sd3_path=self.sd3_path,
                                     datasize=self.image_size,
                                     device=self.device)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image to tensor format"""
        image = image.convert("RGB")
        tensor = self.preprocess_transform(image).unsqueeze(0).to(self.device)  # Shape: (1, 3, H, W)
        return tensor

    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor output back to a PIL image"""
        grid = make_grid(tensor)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        return Image.fromarray(ndarr)

    def encode(self, tensor: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Encode tensor into token indices"""
        with torch.no_grad():
            tokens = self.model.encoding(tensor, device=self.device)
        return tokens, {}  # Add any metadata if needed

    def decode(self, indices: torch.Tensor, additional_info=None) -> torch.Tensor:
        """Decode token indices back into image tensor"""
        with torch.no_grad():
            if isinstance(indices, torch.Tensor):
                idx_np = indices.cpu().numpy()
            else:
                idx_np = indices
            img_tensor = self.model.decoding_with_renderer(idx_np, device=self.device)
        return img_tensor

    def get_num_tokens(self, indices: torch.Tensor) -> int:
        """Get total number of tokens"""
        return indices.numel()

if __name__ == "__main__":
    # Example usage
    tokenizer = SelftokTokenizer(yml_path=CONFIG_PATH, ckpt_path=TOKENIZER_PATH, sd3_path=SD3_PATH, device='cuda', image_size=256)
    tiler = MultiModalPreprocessor(pad_value=-1.0, overlap_margins=(0, 0), base_image_input_size=(IMAGE_SIZE, IMAGE_SIZE), image_token_length_h=32, image_token_length_w=32, image_patch_size=8, max_crops=16)
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
        image_tensor = image_tensor.permute(1, 2, 0).cpu().numpy()  # Change to (H, W, C) for processing
        crops, tiling, patch_ordering, masks = tiler.image_to_patches_and_tokens(image_tensor, is_training=True, rng=np.random.default_rng())
        print(f"Number of tiles: {crops.shape[0]} | Tile shape: {crops.shape[1:]}")
        crops = crops.permute(0, 3, 1, 2)  # Change to (N, C, H, W) for processing

        total_tiles = crops.shape[0]
        reconstructed_tiles_list = []
        all_indices = []

        print(f"Processing {total_tiles} tiles in batches of {batch_size}")

        for i in range(0, total_tiles, batch_size):
            end_idx = min(i + batch_size, total_tiles)
            batch_tiles = crops[i:end_idx].to(tokenizer.device)

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
        reconstructed_tiles = reconstructed_tiles.to(torch.float32).permute(0, 2, 3, 1).cpu().numpy()

        reconstructed_full, first_image = tiler.reconstruct(reconstructed_tiles, tiling, patch_ordering, masks)

        # Convert to PIL
        reconstructed_full = torch.tensor(reconstructed_full).permute(2, 0, 1)  # Change to (C, H, W)
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
        }

        # Log
        print(f"Input image shape: {metrics['input_shape']}")
        print(f"Number of tokens: {metrics['num_tokens']}")
        print(f"Original image pixels: {metrics['original_pixels']}")
        print(f"Compression ratio: {metrics['compression_ratio']:.2f}x")
        print(f"Indices shape: {metrics['indices_shape']}")

        # Save output
        name_without_ext = Path(name).stem
        output_filename = f"{name_without_ext}_tiled_{metrics['num_tokens']}.png"
        output_path = os.path.join(RECONSTRUCTION_PATH, output_filename)

        reconstructed_image.save(output_path)
        print(f"Saved: {output_filename}")