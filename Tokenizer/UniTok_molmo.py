import os
import sys
from typing import Any, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

from molmo_tiler import MultiModalPreprocessor
from Tokenizer.base import Tokenizer
from utils_benchmark import load_all_images

os.chdir("/users/nirmiger/UniTok")
sys.path.append("/users/nirmiger/UniTok")

from models.unitok import UniTok
from utils.config import Args
from utils.data import normalize_01_into_pm1

TOKENIZER_PATH = "/iopsstor/scratch/cscs/nirmiger/unitok_tokenizer.pth"
TOKENIZER = "unitok"
IMAGE_SIZE = 256

RECONSTRUCTION_PATH = f"/users/nirmiger/benchmark-image-tokenzier/assets/{TOKENIZER}_molmo"


class UniTokTokenizer(Tokenizer):
    """UniTok tokenizer implementation"""

    def __init__(self, ckpt_path: str, device: str = "cuda", image_size: int = 256, seed: int = 0, **kwargs):
        self.ckpt_path = ckpt_path
        self.device = device
        self.image_size = image_size
        self.seed = seed

        torch.manual_seed(seed)
        torch.set_grad_enabled(False)

        super().__init__(**kwargs)

    def _load_model(self) -> None:
        """Load UniTok model from checkpoint"""
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        unitok_cfg = Args()
        unitok_cfg.load_state_dict(ckpt["args"])

        self.model = UniTok(unitok_cfg)
        self.model.load_state_dict(ckpt["trainer"]["unitok"])
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image to tensor format expected by UniTok"""
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize_01_into_pm1,
            ]
        )

        tensor = preprocess(image).unsqueeze(0).to(self.device)  # Shape: (1, 3, H, W)
        return tensor

    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor output back to a PIL image"""
        img = tensor.add(1).mul_(0.5 * 255).round().nan_to_num_(128, 0, 255).clamp_(0, 255)
        img = img.to(dtype=torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]  # Shape: HWC
        return Image.fromarray(img)

    def encode(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Encode tensor into discrete token indices"""
        with torch.no_grad():
            code_idx = self.model.img_to_idx(tensor)
        return code_idx, {}  # Additional info unused here

    def decode(self, indices: torch.Tensor, additional_info: Any = None) -> torch.Tensor:
        """Decode discrete tokens back into an image tensor"""
        with torch.no_grad():
            img_tensor = self.model.idx_to_img(indices)
        return img_tensor

    def get_num_tokens(self, indices: torch.Tensor) -> int:
        """Get total number of tokens"""
        return indices.numel()


if __name__ == "__main__":
    # Example usage
    tokenizer = UniTokTokenizer(ckpt_path=TOKENIZER_PATH, device="cuda", image_size=IMAGE_SIZE)
    tiler = MultiModalPreprocessor(pad_value=-1.0, max_crops=16)
    images, _, image_paths = load_all_images("/users/nirmiger/benchmark-image-tokenzier/assets/original")
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
        crops, tiling, patch_ordering, masks = tiler.image_to_patches_and_tokens(
            image_tensor, is_training=True, rng=np.random.default_rng()
        )
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
        reconstructed_tiles = reconstructed_tiles.permute(0, 2, 3, 1).cpu().numpy()

        reconstructed_full, first_image = tiler.reconstruct(reconstructed_tiles, tiling, patch_ordering, masks)

        # Convert to PIL
        reconstructed_full = torch.tensor(reconstructed_full).permute(2, 0, 1)  # Change to (C, H, W)
        reconstructed_image = tokenizer.postprocess(reconstructed_full.unsqueeze(0))

        # Metrics
        total_tokens = all_indices_tensor.numel()
        original_pixels = image_tensor.shape[-2] * image_tensor.shape[-1]
        compression_ratio = original_pixels / total_tokens

        metrics = {
            "input_shape": image_tensor.shape,
            "num_tokens": total_tokens,
            "original_pixels": original_pixels,
            "compression_ratio": compression_ratio,
            "indices_shape": all_indices_tensor.shape,
            "num_tiles": total_tiles,
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
