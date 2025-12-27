import os
import sys

# Add base directory to path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_dir)

# Add Emu3.5 submodule to path
emu3_path = os.path.join(os.path.dirname(__file__), "submodules", "Emu3.5", "src")
sys.path.insert(0, emu3_path)

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

# Import from Emu3.5 submodule
from vision_tokenizer.ibq import IBQ as Emu3IBQModel

# Import base class
from Tokenizer.base import Tokenizer


class Emu3_5_IBQ(Tokenizer):
    """Emu3.5 IBQ discrete image tokenizer"""

    def __init__(
        self,
        model_path: str,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        metadata_only: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize Emu3.5 IBQ tokenizer

        Args:
            model_path: Path to the Emu3.5 IBQ model checkpoint directory
                       Should contain config.yaml and model.ckpt
            min_pixels: Minimum number of pixels after resizing (if None, no resizing)
            max_pixels: Maximum number of pixels after resizing (if None, no resizing)
            metadata_only: If True, only load metadata (codebook_size, name) without model weights
            verbose: If True, print detailed information during processing (default: False)
        """
        self.model_path = model_path
        self.name = "Emu3_5_IBQ"
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.spatial_factor = 16  # Emu3.5 uses 16x downsampling
        self.verbose = verbose
        self.dtype = None  # Will be set during model loading

        # If metadata_only, just load config and return
        if metadata_only:
            config_path = os.path.join(model_path, "config.yaml")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")

            cfg = OmegaConf.load(config_path)

            if "n_embed" not in cfg or "embed_dim" not in cfg:
                raise ValueError(f"Config must contain 'n_embed' and 'embed_dim'. Found keys: {list(cfg.keys())}")

            self.codebook_size = cfg["n_embed"]
            self.codebook_dim = cfg["embed_dim"]
            return

        super().__init__(**kwargs)

    def smart_resize(self, height: int, width: int) -> Tuple[int, int]:
        """
        Smart resize following Emu3's approach - maintains aspect ratio and keeps
        pixels within [min_pixels, max_pixels] range while ensuring divisibility by spatial_factor.

        Args:
            height: Original image height
            width: Original image width

        Returns:
            Tuple of (new_height, new_width)
        """
        if self.min_pixels is None or self.max_pixels is None:
            # No resizing if pixels bounds not set
            return height, width

        factor = self.spatial_factor

        if height < factor or width < factor:
            raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")

        # First, round to nearest multiple of factor
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor

        # Check if we need to scale to fit within pixel bounds
        if h_bar * w_bar > self.max_pixels:
            # Scale down to fit max_pixels
            beta = math.sqrt((height * width) / self.max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < self.min_pixels:
            # Scale up to meet min_pixels
            beta = math.sqrt(self.min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor

        return h_bar, w_bar

    def _load_model(self) -> None:
        """Load the Emu3.5 IBQ model"""
        print(f"Loading {self.name} from {self.model_path}...")

        try:
            # Load configuration
            config_path = os.path.join(self.model_path, "config.yaml")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")

            cfg = OmegaConf.load(config_path)

            # Initialize the model
            self.model = Emu3IBQModel(**cfg)

            # Load checkpoint
            ckpt_path = os.path.join(self.model_path, "model.ckpt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(ckpt)

            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()

            print(f"✓ {self.name} loaded successfully")
            print(f"Model device: {self.device}")

            # Get codebook size and embedding dimension from config
            if "n_embed" not in cfg or "embed_dim" not in cfg:
                raise ValueError(f"Config must contain 'n_embed' and 'embed_dim'. Found keys: {list(cfg.keys())}")

            self.codebook_size = cfg["n_embed"]
            self.codebook_dim = cfg["embed_dim"]
            print(f"Codebook size: {self.codebook_size}")
            print(f"Codebook dimension: {self.codebook_dim}")

            # Cache dtype for efficient preprocessing
            self.dtype = next(self.model.parameters()).dtype
            if self.verbose:
                print(f"Model dtype: {self.dtype}")

            self.get_params()

        except Exception as e:
            print(f"✗ Failed to load {self.name}: {e}")
            raise

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL image to tensor format expected by the model
        Following Emu3.5's preprocessing from src/utils/input_utils.py
        Optionally applies smart resizing if min_pixels/max_pixels are set
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply smart resize if pixel bounds are set (following Emu3's approach)
        if self.min_pixels is not None and self.max_pixels is not None:
            width, height = image.size
            new_height, new_width = self.smart_resize(height, width)

            # Only resize if dimensions changed
            if (new_height, new_width) != (height, width):
                image = image.resize((new_width, new_height), Image.BICUBIC)
                if self.verbose:
                    print(
                        f"Resized image from {width}x{height} to {new_width}x{new_height} "
                        f"({width*height} -> {new_width*new_height} pixels)"
                    )

        # Following exact preprocessing from Emu3.5's build_image function
        # Use cached dtype for efficiency (avoids iterating through parameters on every image)
        image_tensor = torch.tensor((np.array(image) / 127.5 - 1.0)).to(self.device, self.dtype).permute(2, 0, 1)

        # Note: NOT adding batch dimension here to be consistent with other tokenizers
        # Batch dimension will be added in encode() method when needed

        return image_tensor

    def preprocess_batch(self, images: List[Image.Image], resize_size: Tuple[int, int]) -> torch.Tensor:
        """
        Preprocess batch of PIL images to tensor format with specific resize dimensions.

        Args:
            images: List of PIL Images
            resize_size: Target (height, width) for resizing all images

        Returns:
            Batched tensor of shape [B, C, H, W]
        """
        batch_tensors = []
        for image in images:
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize to specified size
            image = image.resize((resize_size[1], resize_size[0]), Image.BICUBIC)

            # Convert to tensor and normalize
            image_tensor = torch.tensor((np.array(image) / 127.5 - 1.0)).to(self.device, self.dtype).permute(2, 0, 1)
            batch_tensors.append(image_tensor)

        # Stack into batch
        return torch.stack(batch_tensors, dim=0)

    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """
        Postprocess tensor back to PIL image
        Expects tensor in range [-1, 1]
        """
        # Remove batch dimension if present
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # Clamp to [-1, 1] and convert to [0, 1]
        tensor = tensor.clamp(-1, 1)
        tensor = (tensor + 1.0) / 2.0

        # Convert to numpy (HWC format) and scale to [0, 255]
        image_np = tensor.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)

        # Convert to PIL Image
        return Image.fromarray(image_np)

    def encode(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode tensor to discrete tokens using Emu3.5 IBQ

        Args:
            tensor: Input tensor in shape (C, H, W) or (B, C, H, W) normalized to [-1, 1]

        Returns:
            indices: Token indices
            additional_info: Dictionary containing latent shape info for decoding
        """
        with torch.no_grad():
            # Add batch dimension if not present
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)

            # Encode using Emu3.5 IBQ
            quant, emb_loss, info = self.model.encode(tensor)

            # Extract indices from info tuple
            # info is typically (perplexity, min_encodings, indices)
            if isinstance(info, tuple) and len(info) > 2:
                indices = info[2]
            else:
                # If info structure is different, assume last element is indices
                indices = info[-1] if isinstance(info, tuple) else info

            # Note: After modifying Emu3.5 source, indices are now spatial [B, H, W]
            # instead of flattened, matching Emu3's behavior

            # Store additional info needed for decoding
            additional_info = {
                "latent_shape": quant.shape,  # Shape of the latent representation [B, C, H, W]
                "original_shape": tensor.shape,
            }

            return indices, additional_info

    def decode(self, indices: torch.Tensor, additional_info: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Decode discrete tokens back to tensor using Emu3.5 IBQ

        Args:
            indices: Token indices
            additional_info: Dictionary containing latent shape info

        Returns:
            Reconstructed tensor in range [-1, 1]
        """
        with torch.no_grad():
            # Get the shape for proper reconstruction
            if additional_info and "latent_shape" in additional_info:
                latent_shape = additional_info["latent_shape"]
                # latent_shape is (B, C, H, W) format
                # decode_code expects shape as (batch, height, width, channel)
                shape = (latent_shape[0], latent_shape[2], latent_shape[3], latent_shape[1])
            else:
                # Estimate shape from indices
                # Assuming indices shape is [batch, num_tokens] or [batch, height, width]
                if indices.ndim == 3:
                    # Already in [batch, height, width] format
                    batch_size, h, w = indices.shape
                    shape = (batch_size, h, w, self.codebook_dim)
                elif indices.ndim == 2:
                    # [batch, num_tokens] format - need to reshape
                    batch_size = indices.shape[0]
                    num_tokens = indices.shape[1]
                    # Assume square spatial dimensions
                    spatial_size = int(np.sqrt(num_tokens))
                    shape = (batch_size, spatial_size, spatial_size, self.codebook_dim)
                else:
                    # Single batch, flatten format
                    num_tokens = indices.numel()
                    spatial_size = int(np.sqrt(num_tokens))
                    shape = (1, spatial_size, spatial_size, self.codebook_dim)

            # Decode using codebook lookup with indices only
            reconstructed = self.model.decode_code(indices, shape=shape)

            # Clamp to valid range
            return reconstructed.clamp(-1, 1)

    def get_num_tokens(self, indices: torch.Tensor) -> int:
        """Get number of tokens for compression ratio calculation"""
        # After modifying Emu3.5 source, indices are spatial [B, H, W]
        if len(indices.shape) == 3:  # [batch, height, width]
            return indices.shape[1] * indices.shape[2]
        elif len(indices.shape) == 2:  # [height, width] for single image
            return indices.shape[0] * indices.shape[1]
        else:
            # For other shapes, return total elements per batch
            return indices.numel() // (indices.shape[0] if indices.shape else 1)


if __name__ == "__main__":
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from utils_benchmark import load_all_images

    # Model path - shared location for all users
    model_path = "/capstor/store/cscs/swissai/infra01/MLLM/Emu3.5-VisionTokenizer"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("This is the shared model location.")
        exit(1)

    # Initialize the Emu3.5 IBQ tokenizer with smart resizing
    print("Initializing Emu3.5 IBQ tokenizer...")
    tokenizer = Emu3_5_IBQ(
        model_path=model_path, min_pixels=512 * 512, max_pixels=1024 * 1024  # Same as Emu3  # Same as Emu3
    )

    # Get model parameters
    tokenizer.get_params()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load images from original assets folder
    images, image_names, image_paths = load_all_images(
        "/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/original"
    )

    print(f"Found {len(images)} images to process")

    # Setup output path
    RECONSTRUCTION_PATH = f"/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/{tokenizer.name}"
    os.makedirs(RECONSTRUCTION_PATH, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"PROCESSING WITH {tokenizer.name}")
    print(f"Min pixels: {tokenizer.min_pixels}, Max pixels: {tokenizer.max_pixels}")
    print(f"Output path: {RECONSTRUCTION_PATH}")
    print(f"{'='*80}")

    # Process each image
    with torch.no_grad():
        for idx, (image, name) in enumerate(zip(images, image_names)):
            print(f"\n{'-'*60}")
            print(f"Processing image {idx+1}/{len(images)}: {name}")
            print(f"{'-'*60}")

            try:
                # Preprocess (includes smart resizing)
                image_tensor = tokenizer.preprocess(image)
                print(f"Preprocessed shape: {image_tensor.shape}")

                # Add batch dimension for tokenizer
                batch_tensor = image_tensor.unsqueeze(0) if image_tensor.ndim == 3 else image_tensor

                # Encode and decode
                indices, additional_info = tokenizer.encode(batch_tensor)
                reconstructed_batch = tokenizer.decode(indices, additional_info)

                # Remove batch dimension and move to CPU
                reconstructed_tensor = reconstructed_batch.squeeze(0).clamp(-1, 1).cpu()

                print(f"Encoded indices shape: {indices.shape}")

                # Calculate tokens and compression ratio
                total_tokens = tokenizer.get_num_tokens(indices)
                original_pixels = image.width * image.height
                compression_ratio = original_pixels / total_tokens

                print(f"Total tokens: {total_tokens}")
                print(f"Original image pixels: {original_pixels}")
                print(f"Compression ratio: {compression_ratio:.2f}x")

                # Convert to PIL for saving
                reconstructed_pil = tokenizer.postprocess(reconstructed_tensor.unsqueeze(0))

                # Create filename with token count
                name_without_ext = os.path.splitext(name)[0]
                output_filename = f"{name_without_ext}_{total_tokens}.png"
                output_path = os.path.join(RECONSTRUCTION_PATH, output_filename)

                # Save the reconstructed image
                reconstructed_pil.save(output_path)
                print(f"  💾 Saved: {output_filename}")

            except Exception as e:
                print(f"  ❌ Error processing {name}: {e}")
                import traceback

                traceback.print_exc()
                continue

            # Clean up memory
            del image_tensor, batch_tensor, indices, reconstructed_batch, reconstructed_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print(f"✅ Processing complete!")
    print(f"Reconstructions saved to: {RECONSTRUCTION_PATH}")
    print(f"{'='*80}")
