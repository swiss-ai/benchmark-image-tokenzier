from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# Simple function to load all images from a folder
def load_all_images(folder_path):
    """Load all image files from the specified folder."""
    folder = Path(folder_path)
    image_extensions = {".png", ".jpg", ".jpeg"}

    images = []
    image_names = []
    image_paths = []

    # Get all files and filter for images
    for file_path in sorted(folder.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            try:
                images.append(Image.open(file_path).convert("RGB"))
                image_names.append(file_path.stem)  # filename without extension
                image_paths.append(str(file_path))
                print(f"Loaded: {file_path.name}")
            except Exception as e:
                print(f"Failed to load {file_path.name}: {e}")

    return images, image_names, image_paths


def _convert_input(image: Union[torch.Tensor, Image.Image, np.ndarray]) -> torch.Tensor:
    """Convert input to CHW tensor format."""
    if isinstance(image, Image.Image):
        image = torch.tensor(np.array(image), dtype=torch.float32)
        if len(image.shape) == 3 and image.shape[-1] in [1, 3, 4]:
            image = image.permute(2, 0, 1)  # HWC -> CHW
        elif len(image.shape) == 2:
            image = image.unsqueeze(0)  # HW -> CHW
    elif isinstance(image, np.ndarray):
        image = torch.tensor(image, dtype=torch.float32)
        if len(image.shape) == 3 and image.shape[-1] in [1, 3, 4]:
            image = image.permute(2, 0, 1)  # HWC -> CHW
        elif len(image.shape) == 2:
            image = image.unsqueeze(0)  # HW -> CHW
    elif isinstance(image, torch.Tensor):
        image = image.float()
        if len(image.shape) == 3 and image.shape[-1] in [1, 3, 4]:
            image = image.permute(2, 0, 1)  # HWC -> CHW
        elif len(image.shape) == 2:
            image = image.unsqueeze(0)  # HW -> CHW

    return image


def resize_by_ratio(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    ratio: float,
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Resize image by a scaling ratio.

    Args:
        image: Input image
        ratio: Scaling ratio (>1 for upscaling, <1 for downscaling)
        interpolation_mode: Interpolation method ('nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area')
        align_corners: Whether to align corners in interpolation
        antialias: Whether to use antialiasing (only for bicubic/bilinear)

    Returns:
        Resized image as torch.Tensor in CHW format
    """
    tensor = _convert_input(image)
    C, H, W = tensor.shape

    target_H = int(H * ratio)
    target_W = int(W * ratio)

    print(f"Resizing by ratio {ratio:.3f}: {H}×{W} → {target_H}×{target_W}")

    # Resize using interpolation
    resized = F.interpolate(
        tensor.unsqueeze(0),  # Add batch dimension
        size=(target_H, target_W),
        mode="bicubic",
        # align_corners=align_corners,
        antialias=True,
    ).squeeze(
        0
    )  # Remove batch dimension

    return resized
