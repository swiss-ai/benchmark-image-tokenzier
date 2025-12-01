import os
import sys

LLAMAGEN_DIR = os.path.join(os.path.dirname(__file__), "..", "repos", "LlamaGen")
if os.path.exists(LLAMAGEN_DIR):
    sys.path.insert(0, LLAMAGEN_DIR)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from Tokenizer.base import Tokenizer
from tokenizer.vqgan.model import VQGAN_FROM_TAMING, VQModel


class VQGAN(Tokenizer):
    """VQGAN tokenizer implementation"""

    def __init__(self, vqgan_model: str = "vqgan_openimage_f8_16384", image_size: int = 512, seed: int = 0, **kwargs):
        self.vqgan_model = vqgan_model
        self.image_size = image_size
        self.seed = seed

        # Set seed for reproducibility
        torch.manual_seed(self.seed)
        torch.set_grad_enabled(False)

        super().__init__(**kwargs)

    def _load_model(self) -> None:
        """Load the LlamaGen VQGAN model"""
        # Get config and checkpoint paths
        cfg, ckpt = VQGAN_FROM_TAMING[self.vqgan_model]

        LLAMAGEN_DIR = "/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/LlamaGen"
        cfg_path = os.path.join(LLAMAGEN_DIR, cfg)
        ckpt_path = os.path.join(LLAMAGEN_DIR, ckpt)

        print(f"Loading config from: {cfg_path}")
        print(f"Loading checkpoint from: {ckpt_path}")

        # Load config and create model
        config = OmegaConf.load(cfg_path)
        self.model = VQModel(**config.model.get("params", dict()))

        # Load checkpoint and setup
        self.model.init_from_ckpt(ckpt_path)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image to tensor format expected by LlamaGen VQGAN"""
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # Store original size for later reconstruction
        self.original_size = image.size

        # Resize to model input size
        # image = image.resize((self.image_size, self.image_size))

        # Normalize to [-1, 1] range as expected by VQGAN
        x = np.array(image) / 127.5 - 1.0

        # Convert to tensor and rearrange dimensions
        x = torch.tensor(x, dtype=torch.float32)
        x = x.unsqueeze(dim=0)  # Add batch dimension
        x = torch.einsum("nhwc->nchw", x)  # Convert to NCHW format

        # Move to device
        x_input = x.to(self.device)

        return x_input

    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """Postprocess tensor back to PIL image"""
        # Interpolate back to original size if needed
        # if hasattr(self, 'original_size') and self.original_size:
        #     # Note: PIL size is (width, height), but interpolate expects (height, width)
        #     target_size = [self.original_size[1], self.original_size[0]]
        #     tensor = F.interpolate(tensor, size=target_size, mode='bilinear')

        # Convert from NCHW to NHWC and remove batch dimension
        output = tensor.permute(0, 2, 3, 1)[0]

        # Convert from [-1, 1] range to [0, 255] uint8
        sample = torch.clamp(127.5 * output + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

        return Image.fromarray(sample)

    def encode(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Encode tensor to discrete tokens"""
        with torch.no_grad():
            # LlamaGen VQGAN encode returns: latent, *, [*, *, indices]
            latent, _, [_, _, indices] = self.model.encode(tensor)

        # Store latent shape for decoding
        additional_info = {"latent_shape": latent.shape, "latent": latent}

        return indices, additional_info

    def decode(self, indices: torch.Tensor, additional_info: Any = None) -> torch.Tensor:
        """Decode discrete tokens back to tensor"""
        if additional_info is None:
            raise ValueError("additional_info with latent_shape is required for LlamaGen decoding")

        latent_shape = additional_info["latent_shape"]

        with torch.no_grad():
            # Use decode_code method which takes indices and latent shape
            output = self.model.decode_code(indices, latent_shape)

        return output

    def get_num_tokens(self, indices: torch.Tensor) -> int:
        """Get number of tokens for compression ratio calculation"""
        return indices.numel()


if __name__ == "__main__":

    # Example usage replicating your original code structure
    from utils import load_all_images, resize_by_ratio

    images, image_names, image_paths = load_all_images(
        "/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/original"
    )

    TOKENIZER = "vqgan_openimage_f8_256"  # vqgan_openimage_f8_256 , vqgan_openimage_f8_16384
    RECONSTRUCTION_PATH = f"/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/{TOKENIZER}"
    os.makedirs(RECONSTRUCTION_PATH, exist_ok=True)

    # Initialize tokenizer
    tokenizer = VQGAN(
        vqgan_model=TOKENIZER,
    )

    for idx, image_path in enumerate(image_paths):
        name = Path(image_path).name
        print(f"\nProcessing image {idx+1}: {name}")

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Use tokenizer's reconstruct method
        reconstructed_image, metrics = tokenizer.reconstruct(image)

        # Print metrics
        print(f"Input image shape: {metrics['input_shape']}")
        print(f"Number of tokens: {metrics['num_tokens']}")
        print(f"Original image pixels: {metrics['original_pixels']}")
        print(f"Compression ratio: {metrics['compression_ratio']:.2f}x")
        print(f"Indices shape: {metrics['indices_shape']}")

        # Save reconstructed image with number of tokens
        name_without_ext = Path(name).stem
        output_filename = f"{name_without_ext}_{metrics['num_tokens']}.png"
        output_path = os.path.join(RECONSTRUCTION_PATH, output_filename)
        reconstructed_image.save(output_path)
        print(f"Saved: {output_filename}")

        # # Display comparison
        # fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # # Original image
        # axes[0].imshow(image)
        # axes[0].set_title(f"Original: {name}")
        # axes[0].axis('off')

        # # Reconstructed image
        # axes[1].imshow(reconstructed_image)
        # axes[1].set_title(f"Reconstructed: {name} ({metrics['num_tokens']} tokens)")
        # axes[1].axis('off')

        # plt.tight_layout()
        # plt.show()
