"""
Perceptual quality metrics for image completion benchmarks.

Computes PSNR, SSIM, and LPIPS between completed images and reference images,
both for the full image and for the generated region only.
"""

from typing import ClassVar, Dict, Set

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from vision_tokenization.qualitative_benchmark.metrics import register_metric
from vision_tokenization.qualitative_benchmark.metrics.base import BaseMetric

# Lazy-loaded global LPIPS model (shared across instances for efficiency)
_lpips_model = None
_lpips_device = None


def _get_lpips_model(device):
    """Lazy-load and cache the LPIPS model."""
    global _lpips_model, _lpips_device
    if _lpips_model is None or _lpips_device != device:
        import warnings

        from lpips import LPIPS

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            _lpips_model = LPIPS(net="alex").to(device)
            _lpips_model.eval()
        _lpips_device = device
    return _lpips_model


def _pil_to_lpips_tensor(img, device):
    """Convert PIL image to normalized tensor for LPIPS ([-1, 1] range)."""
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )
    return transform(img).unsqueeze(0).to(device)


def _resize_to_match(img, target_size):
    """Resize image to target (width, height), matching calculate_metrics.py pattern."""
    if img.size != target_size:
        return img.resize(target_size, Image.BICUBIC)
    return img


def _compute_psnr_ssim_lpips(img_a, img_b, device):
    """Compute PSNR, SSIM, LPIPS between two same-sized PIL images."""
    arr_a = np.array(img_a)
    arr_b = np.array(img_b)

    psnr_val = float(psnr(arr_a, arr_b))
    ssim_val = float(ssim(arr_a, arr_b, channel_axis=-1))

    model = _get_lpips_model(device)
    with torch.no_grad():
        tensor_a = _pil_to_lpips_tensor(img_a, device)
        tensor_b = _pil_to_lpips_tensor(img_b, device)
        lpips_val = float(model(tensor_a, tensor_b).item())

    return psnr_val, ssim_val, lpips_val


@register_metric("completion_quality")
class CompletionQualityMetric(BaseMetric):
    """
    Perceptual quality metrics for image completion benchmarks.

    Computes PSNR, SSIM, and LPIPS between completed images and reference images,
    both for the full image and for the generated region only.

    Required input keys:
        completed_image: PIL Image of the completed result (given + generated)
        reference_image: PIL Image of the reconstructed reference (encode -> decode full original)
        boundary_y: Pixel y-coordinate where given region ends and generated region begins
        expected_generated_height: Expected pixel height of the generated region
        actual_generated_height: Actual pixel height of the generated region

    Output keys:
        psnr_full, ssim_full, lpips_full: Full image metrics
        psnr_generated, ssim_generated, lpips_generated: Generated region metrics
        row_count_matched: Whether actual == expected generated height
        overlap_height: Height of overlapping region used for generated metrics
    """

    name: ClassVar[str] = "completion_quality"
    REQUIRED_KEYS: ClassVar[Set[str]] = {
        "completed_image",
        "reference_image",
        "boundary_y",
        "expected_generated_height",
        "actual_generated_height",
    }

    def _setup(self, **kwargs):
        """Pre-load the LPIPS model for efficiency."""
        _get_lpips_model(self.device)

    def __call__(self, data: dict) -> dict:
        """
        Compute completion quality metrics.

        Args:
            data: Dictionary with required keys (see class docstring)

        Returns:
            Flat dictionary with metric key-value pairs
        """
        data = self.validate_input(data)

        completed_image = data["completed_image"]
        reference_image = data["reference_image"]
        boundary_y = data["boundary_y"]
        expected_generated_height = data["expected_generated_height"]
        actual_generated_height = data["actual_generated_height"]

        row_count_matched = actual_generated_height == expected_generated_height

        # --- Full image metrics (resize-to-match) ---
        target_w = min(completed_image.width, reference_image.width)
        target_h = min(completed_image.height, reference_image.height)
        target_size = (target_w, target_h)

        comp_full = _resize_to_match(completed_image, target_size)
        ref_full = _resize_to_match(reference_image, target_size)

        psnr_full, ssim_full, lpips_full = _compute_psnr_ssim_lpips(comp_full, ref_full, self.device)

        # --- Generated region metrics (crop-to-overlap) ---
        overlap_height = min(actual_generated_height, expected_generated_height)

        psnr_gen, ssim_gen, lpips_gen = None, None, None

        if overlap_height > 0 and boundary_y < completed_image.height and boundary_y < reference_image.height:
            # Crop generated region from both images
            comp_crop = completed_image.crop((0, boundary_y, completed_image.width, boundary_y + overlap_height))
            ref_crop = reference_image.crop((0, boundary_y, reference_image.width, boundary_y + overlap_height))

            # Ensure same width (should already match for valid completions)
            crop_w = min(comp_crop.width, ref_crop.width)
            crop_h = min(comp_crop.height, ref_crop.height)
            if crop_w > 0 and crop_h > 0:
                comp_crop = _resize_to_match(comp_crop, (crop_w, crop_h))
                ref_crop = _resize_to_match(ref_crop, (crop_w, crop_h))

                psnr_gen, ssim_gen, lpips_gen = _compute_psnr_ssim_lpips(comp_crop, ref_crop, self.device)

        return {
            "psnr_full": psnr_full,
            "ssim_full": ssim_full,
            "lpips_full": lpips_full,
            "psnr_generated": psnr_gen,
            "ssim_generated": ssim_gen,
            "lpips_generated": lpips_gen,
            "row_count_matched": row_count_matched,
            "overlap_height": overlap_height,
        }
