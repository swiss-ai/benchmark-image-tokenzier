import torch
import numpy as np
from PIL import Image
from typing import Tuple, Any
from torchvision import transforms
import os
import sys
from utils_benchmark import load_all_images
from pathlib import Path

from Tiler import Tiler
from Tokenizer import Tokenizer

os.chdir('/users/nirmiger/TokenFlow')
sys.path.append('/users/nirmiger/TokenFlow')

from tokenflow.tokenizer.vq_model import VQ_models

TOKENIZER_PATH = '/iopsstor/scratch/cscs/nirmiger/tokenflow_siglip_32k.pt'
TOKENIZER = 'tokenflow_384'
RECONSTRUCTION_PATH = f'/users/nirmiger/benchmark-image-tokenzier/assets/{TOKENIZER}'

class TokenFlowTokenizer(Tokenizer):
    """UniTok tokenizer implementation"""
    def __init__(self,
                 ckpt_path: str,
                 vq_model_name: str = "TokenFlow",
                 teacher: str = "siglip_384",
                 codebook_size: int = 32768,
                 codebook_embed_dim: int = 8,
                 semantic_code_dim: int = 32,
                 image_size: int = 384,
                 enhanced_decoder: bool = False,
                 infer_interpolate: bool = False,
                 device: str = "cuda",
                 seed: int = 0,
                 **kwargs):

        self.device = device
        self.image_size = image_size
        self.ckpt_path = ckpt_path
        self.vq_model_name = vq_model_name
        self.teacher = teacher
        self.codebook_size = codebook_size
        self.codebook_embed_dim = codebook_embed_dim
        self.semantic_code_dim = semantic_code_dim
        self.enhanced_decoder = enhanced_decoder
        self.infer_interpolate = infer_interpolate
        self.seed = seed
        super().__init__(**kwargs)

        torch.manual_seed(seed)
        torch.set_grad_enabled(False)

    def _load_model(self) -> None:
        """Load TokenFlow VQ model from checkpoint."""
        self.model = VQ_models[self.vq_model_name](
            codebook_size=self.codebook_size,
            codebook_embed_dim=self.codebook_embed_dim,
            semantic_code_dim=self.semantic_code_dim,
            teacher=self.teacher,
            enhanced_decoder=self.enhanced_decoder,
            infer_interpolate=self.infer_interpolate
        ).to(self.device).eval()

        checkpoint = torch.load(self.ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("ema") or checkpoint.get("model") or checkpoint.get("state_dict")
        if state_dict is None:
            raise ValueError("Checkpoint does not contain valid model weights.")
        self.model.load_state_dict(state_dict)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image (PIL → normalized tensor)"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),  # [0,1] → [-1,1]
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """Convert output tensor back to PIL Image"""
        tensor = tensor.squeeze(0).cpu()
        tensor = tensor * 0.5 + 0.5  # [-1, 1] → [0, 1]
        tensor = tensor.mul(255).add(0.5).clamp(0, 255).to(torch.uint8)
        return Image.fromarray(tensor.permute(1, 2, 0).numpy())

    def encode(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Encode tensor into discrete tokens"""
        with torch.no_grad():
            latent, _, _ = self.model.encode(tensor)
        return latent, {}  # TokenFlow doesn’t use auxiliary info

    def decode(self, indices: torch.Tensor, additional_info: dict = None) -> torch.Tensor:
        """Decode discrete tokens back into image tensor"""
        with torch.no_grad():
            output = self.model.decode(indices)
            if isinstance(output, tuple):
                output = output[1]
        return output

    def get_num_tokens(self, indices: torch.Tensor) -> int:
        """Return number of tokens in flattened index tensor"""
        return int(indices.numel()/40) # 40 is the embedding dimension for TokenFlow
    

if __name__ == "__main__":
    # Example usage
    tokenizer = TokenFlowTokenizer(ckpt_path=TOKENIZER_PATH)
    tiler = Tiler(tile_size=384, pad_value=-1.0)
    images, _, image_paths = load_all_images('/users/nirmiger/benchmark-image-tokenzier/assets/original')
    batch_size = 8  # Adjust based on GPU memory
    os.makedirs(RECONSTRUCTION_PATH, exist_ok=True)
    for idx, image_path in enumerate(image_paths):
        name = Path(image_path).name
        print(f"\nProcessing image {idx+1}: {name}")
        
        # Load and normalize image manually
        image = Image.open(image_path).convert("RGB")
        image_np = (np.array(image) / 127.5 - 1.0).astype(np.float32)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # (C, H, W)
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