"""
Perceptual quality metrics for image completion benchmarks.

Computes PSNR, SSIM, and LPIPS between completed images and reference images,
both for the full image and for the generated region only.
"""

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Lazy-loaded global LPIPS model
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


def compute_completion_metrics(
    completed_image,
    reference_image,
    boundary_y,
    expected_generated_height,
    actual_generated_height,
    device="cpu",
):
    """
    Compute perceptual quality metrics for an image completion result.

    Computes metrics in two scopes:
    - *_full: Full image (given + generated) vs full reference. Uses resize-to-match
      if the completed image has a different height than the reference.
    - *_generated: Generated region only (cropped at boundary_y). Uses crop-to-overlap
      (min of actual vs expected generated height) to handle row count mismatches.

    Args:
        completed_image: PIL Image of the completed result (given + generated).
        reference_image: PIL Image of the reconstructed reference (encode -> decode full original).
        boundary_y: Pixel y-coordinate where the given region ends and generated region begins.
        expected_generated_height: Expected pixel height of the generated region.
        actual_generated_height: Actual pixel height of the generated region in completed_image.
        device: Torch device for LPIPS computation.

    Returns:
        Dictionary with keys:
            psnr_full, ssim_full, lpips_full,
            psnr_generated, ssim_generated, lpips_generated,
            row_count_matched, overlap_height
    """
    row_count_matched = actual_generated_height == expected_generated_height

    # --- Full image metrics (resize-to-match) ---
    target_w = min(completed_image.width, reference_image.width)
    target_h = min(completed_image.height, reference_image.height)
    target_size = (target_w, target_h)

    comp_full = _resize_to_match(completed_image, target_size)
    ref_full = _resize_to_match(reference_image, target_size)

    psnr_full, ssim_full, lpips_full = _compute_psnr_ssim_lpips(comp_full, ref_full, device)

    # --- Generated region metrics (crop-to-overlap) ---
    overlap_height = min(actual_generated_height, expected_generated_height)

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

            psnr_gen, ssim_gen, lpips_gen = _compute_psnr_ssim_lpips(comp_crop, ref_crop, device)
        else:
            psnr_gen, ssim_gen, lpips_gen = None, None, None
    else:
        psnr_gen, ssim_gen, lpips_gen = None, None, None

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
