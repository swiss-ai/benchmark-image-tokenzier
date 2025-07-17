import dataclasses
import math
import warnings
from typing import List, Optional, Union, Any, Tuple

import PIL
from PIL import ImageFile
from PIL import ImageOps


def setup_pil():
    PIL.Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import torch
import torchvision.transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import convert_image_dtype
import torchvision.transforms.functional as TF

from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
)

def load_image(image_path):
    setup_pil()  # Call here so the setting is applied in multi-processing contexts
    if isinstance(image_path, PIL.Image.Image):
        # Avoid annoying palette transparency warnings filling up the logs
        with warnings.catch_warnings(record=True) as w:
            image = image_path.convert("RGB")
        try:
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            pass
        return np.array(image)
    elif isinstance(image_path, np.ndarray):
        assert len(image_path.shape) == 3, "Image should have 3 dimensions"
        assert image_path.shape[2] == 3, "Image should have 3 channels"
        assert image_path.dtype == np.uint8, "Image should have uint8 type"
        return image_path
    else:
        with PIL.Image.open(image_path) as image:
            return load_image(image)


def resize_and_pad(
    image,
    desired_output_size,
    is_training=False,
    pad_value=0,
    rng=np.random
):
    """Resize an image while padding to preserve its aspect ratio."""
    desired_height, desired_width = desired_output_size
    height, width = image.shape[:2]

    # Cast into float32 since the training code did this in float32 and it (very rarely) effects
    # the results after rounding.
    image_scale_y = np.array(desired_height, np.float32) / np.array(height, np.float32)
    image_scale_x = np.array(desired_width, np.float32) / np.array(width, np.float32)
    image_scale = min(image_scale_x, image_scale_y)
    scaled_height = int(np.array(height, np.float32) * image_scale)
    scaled_width = int(np.array(width, np.float32) * image_scale)

    image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    image = convert_image_dtype(image)  # resize in float32 to match the training code
    mode = InterpolationMode.BICUBIC
    image = torchvision.transforms.Resize([scaled_height, scaled_width], mode, antialias=True)(image)
    image = torch.clip(image, 0.0, 1.0)
    image = torch.permute(image, [1, 2, 0]).numpy()

    top_pad = (desired_height - scaled_height) // 2
    left_pad = (desired_width - scaled_width) // 2
    padding = [
        [top_pad, desired_height - scaled_height - top_pad],
        [left_pad, desired_width - scaled_width - left_pad],
        [0, 0]
    ]
    image_mask = np.pad(np.ones_like(image[:, :, 0], dtype=bool), padding[:2])
    image = np.pad(image, padding, constant_values=pad_value)
    return image, image_mask

def resize_and_pad_tensor(image, desired_output_size, pad_value=0):
    """Resize a tensor image while padding to preserve its aspect ratio."""
    desired_height, desired_width = desired_output_size
    c, h, w = image.shape

    # Convert to float32 NumPy array in HWC format
    image = image.to(torch.float32)
    image = image.permute(1, 2, 0).cpu().numpy()  # HWC, float32

    # Calculate scaling
    scale_y = np.float32(desired_height) / np.float32(h)
    scale_x = np.float32(desired_width) / np.float32(w)
    scale = min(scale_x, scale_y)

    scaled_height = int(np.round(h * scale))
    scaled_width = int(np.round(w * scale))

    # Resize using torchvision (convert back to tensor temporarily)
    image_t = TF.resize(torch.from_numpy(image).permute(2, 0, 1), [scaled_height, scaled_width],
                        interpolation=InterpolationMode.BICUBIC, antialias=True)
    image_t = torch.clamp(image_t, 0.0, 1.0)  # In case of overflows

    # Back to NumPy HWC
    image = image_t.permute(1, 2, 0).numpy()

    # Calculate padding
    top_pad = (desired_height - scaled_height) // 2
    left_pad = (desired_width - scaled_width) // 2
    padding = [
        [top_pad, desired_height - scaled_height - top_pad],
        [left_pad, desired_width - scaled_width - left_pad],
        [0, 0],
    ]

    # Mask is True where image is present
    image_mask = np.pad(
        np.ones((scaled_height, scaled_width), dtype=bool),
        padding[:2],
        mode='constant',
        constant_values=False
    )

    # Pad image with constant value
    image = np.pad(image, padding, mode='constant', constant_values=pad_value)
    return image, image_mask
# The resize and pad functions are currently unused, keep here for reference
# def metaclip_resize(image, desired_output_size):
#     image = torch.permute(torch.from_numpy(image), [2, 0, 1])
#     if torch.is_floating_point(image):
#         image = torchvision.transforms.Resize(
#             desired_output_size, InterpolationMode.BICUBIC, antialias=True)(image)
#         image = torch.clip(image, 0.0, 1.0)
#     else:
#         assert image.dtype == torch.uint8, "Expected float images or uint8 images, but got {}".format(image.dtype)
#         image = torchvision.transforms.Resize(
#             desired_output_size, InterpolationMode.BICUBIC, antialias=True)(image)
#         image = image.to(torch.float32)
#         image = torch.clip(image, 0, 255)
#         image = image / 255.0
#     resized = torch.permute(image, [1, 2, 0]).numpy()
#     image_mask = np.ones_like(resized[:, :, 0], dtype=np.bool_)
#     return resized, image_mask


# def siglip_resize_and_pad(
#     image: np.ndarray,
#     desired_output_size: Tuple[int, int],
# ) -> Tuple[np.ndarray, np.ndarray]:
#     image = torch.permute(torch.from_numpy(image), [2, 0, 1])
#     dtype = image.dtype
#     if torch.is_floating_point(image):
#         in_min = 0.0
#         in_max = 1.0
#         resized = torchvision.transforms.Resize(
#             desired_output_size,
#             InterpolationMode.BILINEAR,
#             antialias=False,
#         )(image)
#         resized = torch.clip(resized, 0.0, 1.0).to(dtype)
#     else:
#         assert image.dtype == torch.uint8, "SigLIP expects float images or uint8 images, but got {}".format(image.dtype)
#         in_min = 0.0
#         in_max = 255.0
#         resized = torchvision.transforms.Resize(
#             desired_output_size,
#             InterpolationMode.BILINEAR,
#             antialias=False,
#         )(image)
#         resized = torch.clip(resized, 0, 255).to(dtype)

#     resized = resized.to(torch.float32)
#     resized = (resized - in_min) / (in_max - in_min)

#     resized = torch.permute(resized, [1, 2, 0]).numpy()
#     image_mask = np.ones_like(resized[:, :, 0], dtype=np.bool_)
    
#     return resized, image_mask


# def dino_resize_and_pad(
#     image: np.ndarray,
#     desired_output_size: Tuple[int, int],
# ) -> Tuple[np.ndarray, np.ndarray]:
#     image = torch.permute(torch.from_numpy(image), [2, 0, 1])
#     dtype = image.dtype
#     if torch.is_floating_point(image):
#         resized = torchvision.transforms.Resize(
#             desired_output_size,
#             InterpolationMode.BICUBIC,
#             antialias=True,
#         )(image)
#         resized = torch.clip(resized, 0.0, 1.0).to(torch.float32)
#     else:
#         assert image.dtype == torch.uint8, "DINOv2 expects float images or uint8 images, but got {}".format(image.dtype)
#         resized = torchvision.transforms.Resize(
#             desired_output_size,
#             InterpolationMode.BICUBIC,
#             antialias=True,
#         )(image)
#         resized = torch.clip(resized, 0, 255).to(torch.float32)
#         resized = resized / 255.0
    
#     resized = torch.permute(resized, [1, 2, 0]).numpy()
#     image_mask = np.ones_like(resized[:, :, 0], dtype=np.bool_)

#     return resized, image_mask


def select_tiling(h, w, patch_size, max_num_crops):
    """Divide in image of size [w, h] in up to max_num_patches of size patch_size"""
    original_size = np.stack([h, w])  # [1, 2]
    original_res = h * w
    tilings = []
    for i in range(1, max_num_crops + 1):
        for j in range(1, max_num_crops + 1):
            if i*j <= max_num_crops:
                tilings.append((i, j))
    # sort so argmin and argmax favour smaller tilings in the event of a tie
    tilings.sort(key=lambda x: (x[0]*x[1], x[0]))
    candidate_tilings = np.array(tilings, dtype=np.int32)  # [n_resolutions, 2]
    candidate_resolutions = candidate_tilings * patch_size  # [n_resolutions, 2]

    # How much we would need to scale the image to fit exactly in each tiling
    original_size = np.stack([h, w], dtype=np.float32)  # [1, 2]

    # The original size can be zero in rare cases if the image is smaller than the margin
    # In those cases letting the scale become infinite means the tiling is based on the
    # other side, or falls back to the smallest tiling
    with np.errstate(divide='ignore'):
        required_scale_d = candidate_resolutions.astype(np.float32) / original_size,
    required_scale = np.min(required_scale_d, axis=-1, keepdims=True)  # [n_resolutions, 1]
    if np.all(required_scale < 1):
        # We are forced to downscale, so try to minimize the amount of downscaling
        ix = np.argmax(required_scale)
    else:
        # Pick the resolution that required the least upscaling so that it most closely fits the image
        required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
        ix = np.argmin(required_scale)
    return candidate_tilings[ix]

# We currently do not want to reshape the images into patches, but keep the code here for reference
# def pixels_to_patches(array, patch_size):
#     """Reshape an image of [h, w, 3] -> [n_patches, pixels_per_patch]"""
#     if len(array.shape) == 3:
#         w, h, c = array.shape
#         h_patches = h//patch_size
#         w_patches = w//patch_size
#         array = np.reshape(array, [h_patches, patch_size, w_patches, patch_size, c])
#         array = np.transpose(array, [0, 2, 1, 3, 4])
#         array = np.reshape(array, [h_patches*w_patches, patch_size*patch_size*c])
#     else:
#         w, h = array.shape
#         h_patches = h//patch_size
#         w_patches = w//patch_size
#         array = np.reshape(array, [h_patches, patch_size, w_patches, patch_size])
#         array = np.transpose(array, [0, 2, 1, 3])
#         array = np.reshape(array, [h_patches*w_patches, patch_size*patch_size])
#     return array


# def batch_pixels_to_patches(array, patch_size):
#     """Reshape images of [n_images, h, w, 3] -> [n_images, n_patches, pixels_per_patch]"""
#     if len(array.shape) == 3:
#         n_crops, w, h = array.shape
#         h_patches = h//patch_size
#         w_patches = w//patch_size
#         array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size])
#         array = np.transpose(array, [0, 1, 3, 2, 4])
#         array = np.reshape(array, [n_crops, h_patches*w_patches, patch_size*patch_size])
#         return array
#     else:
#         n_crops, w, h, c = array.shape
#         h_patches = h//patch_size
#         w_patches = w//patch_size
#         array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size, c])
#         array = np.transpose(array, [0, 1, 3, 2, 4, 5])
#         array = np.reshape(array, [n_crops, h_patches*w_patches, patch_size*patch_size*c])
#         return array


@dataclasses.dataclass
class MultiModalPreprocessor:
    """
    Converts text/images inputs into tensors that can be used in the forward method
    for the a model
    """
    tokenizer: Optional[Any] = None
    loss_token_weighting: Optional[str] = None

    # How to crops/resize images
    crop_mode: str = "overlap-and-resize-c2"
    max_crops: int = 10
    overlap_margins: Tuple[int, int] = (2, 2)
    resize: str = "default"
    use_col_tokens: bool = True

    # Data about the ViT and connector we need when deciding the crops
    base_image_input_size: Tuple[int, int] = (256, 256)
    image_pooling_w: int = 1
    image_pooling_h: int = 1
    image_token_length_w: int = 16
    image_token_length_h: int = 16
    image_patch_size: int = 16
    image_padding_mask: Union[bool, int] = False
    pad_value: float = 0

    image_patch_token_id: int = dataclasses.field(init=False)
    image_col_token_id: int = dataclasses.field(init=False)
    image_start_token_id: int = dataclasses.field(init=False)
    image_end_token_id: int = dataclasses.field(init=False)

    # Keep here for compatibility with the old code for now
    def __post_init__(self):
        self.image_end_token_id = 0
        self.image_start_token_id = 0
        self.image_col_token_id = 0
        self.image_patch_token_id = 0
        self.image_prompt_token_id = 0

    # Not used at the moment
    # def _normalize(self, image):
    #     if self.normalize == "openai":
    #         image -= np.array(OPENAI_CLIP_MEAN, dtype=np.float32)[None, None, :]
    #         image /= np.array(OPENAI_CLIP_STD, dtype=np.float32)[None, None, :]
    #     elif self.normalize == "siglip":
    #         image = np.asarray(-1.0, dtype=np.float32) + image * np.asarray(2.0, dtype=np.float32)
    #     elif self.normalize == "dino":
    #         image -= np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
    #         image /= np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]
    #     else:
    #         raise NotImplementedError(self.normalize)
    #     return image

    def resize_image(self, image, output_size, is_training, rng):
        if isinstance(image, torch.Tensor):
            return resize_and_pad_tensor(image, output_size, pad_value=self.pad_value)
        elif isinstance(image, np.ndarray):
            return resize_and_pad(
                image, output_size, pad_value=self.pad_value, rng=rng, is_training=is_training)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}. Expected np.ndarray or torch.Tensor.")


    def image_to_patches_and_tokens(
        self,
        image: ImageInput,
        is_training=False,
        rng=None
    ):
        max_crops = self.max_crops
        overlap_margins = self.overlap_margins
        base_image_input_size = self.base_image_input_size
        image_token_length_w = self.image_token_length_w
        image_token_length_h = self.image_token_length_h
        image_patch_size = self.image_patch_size

        if isinstance(base_image_input_size, int):
            base_image_input_size = (base_image_input_size, base_image_input_size)

        base_image_input_d = image_patch_size
        tokens_per_image = image_token_length_w * image_token_length_h
        image_base_patch_w = base_image_input_size[1] // base_image_input_d
        image_base_patch_h = base_image_input_size[0] // base_image_input_d

        original_image_h, original_image_w = image.shape[:2]
        crop_size = base_image_input_size[0]

        # This part is not important for us at the moment
        # if self.crop_mode == "resize":
        #     resized, img_mask = self.resize_image(image, base_image_input_size, is_training, rng)
        #     resized = self._normalize(resized)
        #     patches = pixels_to_patches(resized, image_patch_size)
        #     img_mask = pixels_to_patches(img_mask, image_patch_size)

        #     per_row = np.full(
        #         (image_token_length_w,),
        #         self.image_patch_token_id,
        #         dtype=np.int32
        #     )
        #     if self.use_col_tokens:
        #         per_row = np.concatenate([per_row, [self.image_col_token_id]], 0, dtype=np.int32)
        #     extra_tokens = np.tile(per_row, [image_token_length_h])
        #     joint = [
        #         [self.image_start_token_id],
        #         extra_tokens,
        #         [self.image_end_token_id],
        #     ]
        #     joint = np.concatenate(joint, 0, dtype=np.int32)
        #     return np.expand_dims(patches, 0), joint, None, img_mask

        if self.crop_mode == "overlap-and-resize-c2":
            # Discard this many patches from the (left/top, right/bottom) of crops
            left_margin, right_margin = overlap_margins
            # Required for compatibility with image pooling
            assert left_margin % self.image_pooling_w == 0 and right_margin % self.image_pooling_w == 0
            assert left_margin % self.image_pooling_h == 0 and right_margin % self.image_pooling_h == 0
            total_margin_pixels = base_image_input_d*(right_margin + left_margin)  # pixels removed per dim
            crop_patches = base_image_input_size[0] // base_image_input_d  # patches per crop dim
            crop_window_patches = crop_patches - (right_margin + left_margin)  # usable patches
            crop_window_size = crop_window_patches * base_image_input_d

            # Decide how to tile the image, to account for the overlap margins we compute the tiling
            # as if we had an image without the margins and were using a crop size without the margins
            tiling = select_tiling(
                original_image_h - total_margin_pixels,
                original_image_w - total_margin_pixels,
                crop_window_size,
                max_crops
            )

            src, img_mask = self.resize_image(
                image,
                [tiling[0]*crop_window_size+total_margin_pixels, tiling[1]*crop_window_size+total_margin_pixels],
                is_training,
                rng
            )
            # src = self._normalize(src)

            # Now we have to split the image into crops, while keeping track of how each patch in the
            # each crop should be ordered in the global image, this require a lot of tricky booking
            n_crops = tiling[0] * tiling[1]
            patches_arr = []
            mask_arr = []
            patch_ordering_arr = []

            # We assume hxw pooling, but can allow padding the right/bottom with extra
            # patches if the number of patches per side is not divisible by h/w
            assert (crop_patches + self.image_pooling_h - 1) // self.image_pooling_h == image_token_length_h
            assert (crop_patches + self.image_pooling_w - 1) // self.image_pooling_w == image_token_length_w
            on = 0
            on_patch = 0
            for i in range(tiling[0]):
                y0 = i*crop_window_size
                if i == 0:
                    crop_y0 = 0
                else:
                    crop_y0 = left_margin // self.image_pooling_h

                crop_h = image_base_patch_h - (right_margin + left_margin)
                if i == 0:
                    crop_h += left_margin
                if i == (tiling[0]-1):
                    crop_h += right_margin
                for j in range(tiling[1]):
                    x0 = j*crop_window_size
                    if j == 0:
                        crop_x0 = 0
                    else:
                        crop_x0 = left_margin // self.image_pooling_w

                    crop_w = image_base_patch_w - (right_margin + left_margin)
                    if j == 0:
                        crop_w += left_margin
                    if j == (tiling[1]-1):
                        crop_w += right_margin

                    pooled_w = (crop_w + self.image_pooling_w - 1) // self.image_pooling_w
                    pooled_h = (crop_h + self.image_pooling_h - 1) // self.image_pooling_h
                    after_padding_width = image_token_length_w - pooled_w - crop_x0
                    after_padding_height = image_token_length_h - pooled_h - crop_y0
                    patch_ordering_arr.append(
                        np.pad(
                            np.reshape(
                                np.arange(on, on+pooled_h*pooled_w, dtype=np.int32),
                                (pooled_h, pooled_w)),
                            [[crop_y0, after_padding_height], [crop_x0, after_padding_width]],
                            constant_values=-1, mode='constant'
                        )
                    )
                    patches_arr.append(src[y0:y0+crop_size, x0:x0+crop_size])
                    mask_arr.append(img_mask[y0:y0+crop_size, x0:x0+crop_size])

                    on += pooled_h*pooled_w
                    on_patch += 1
            patches = np.stack(patches_arr)
            patch_ordering = np.stack(patch_ordering_arr)
            img_mask = np.stack(mask_arr)
            mask_arr = img_mask

            # Switch to [n_crops, n_patches, pixels_per_patch] format
            image_layout_impatch_w, image_layout_impatch_h = tiling[0], tiling[1]

            # patches = batch_pixels_to_patches(patches, image_patch_size)
            # img_mask = batch_pixels_to_patches(img_mask, image_patch_size)
            img_mask = img_mask.astype(np.float32).mean(axis=-1)

            # For now i do not do patch ordering, so we skip this part
            # patch_ordering = np.reshape(patch_ordering, [-1])
            # valid = patch_ordering >= 0

            # # Patch order numbers the patches crop-by-crop, here we transpose
            # # it to get left-to-right order
            # patch_ordering_rh = np.reshape(
            #     patch_ordering,
            #     [tiling[0], tiling[1], image_token_length_h, image_token_length_w]
            # )
            # patch_ordering_rh = np.transpose(patch_ordering_rh, [0, 2, 1, 3])
            # patch_ordering_rh = np.reshape(patch_ordering_rh, [-1])

            # # The transpose will screw up which patches are masked, project the
            # # new order into sparse structure of `patch_ordering` to fix it
            # patch_ordering[valid] = patch_ordering_rh[patch_ordering_rh >= 0]
            def get_num_patches(num_tiles: int, pooling_size: int) -> int:
                if num_tiles > 1:
                    left_crop_window_patches = (crop_window_patches + left_margin + pooling_size - 1) // pooling_size * pooling_size
                    middle_crop_window_patches = (crop_window_patches + pooling_size - 1) // pooling_size * pooling_size
                    right_crop_window_patches = (crop_window_patches + right_margin + pooling_size - 1) // pooling_size * pooling_size
                    return left_crop_window_patches + (num_tiles - 2) * middle_crop_window_patches + right_crop_window_patches
                else:
                    single_crop_window_patches = (crop_patches + pooling_size - 1) // pooling_size * pooling_size
                    return single_crop_window_patches

            # Now build the output tokens
            h = get_num_patches(tiling[0], self.image_pooling_h)
            w = get_num_patches(tiling[1], self.image_pooling_w)
            per_row = np.full(
                (w // self.image_pooling_w,),
                self.image_patch_token_id,
                dtype=np.int32
            )
            if self.use_col_tokens:
                per_row = np.concatenate([per_row, [self.image_col_token_id]], 0)

            joint = np.tile(per_row, [h // self.image_pooling_h])
            joint = [
                [self.image_start_token_id],
                joint,
                [self.image_end_token_id]
            ]

            # Finally do the same for the global image
            resized, mask = self.resize_image(image, base_image_input_size, is_training, rng)
            # resized = self._normalize(resized)
            # resized = pixels_to_patches(resized, image_patch_size)
            patches = np.concatenate([np.expand_dims(resized, 0), patches], 0)

            # Global image goes first, so the order of patches in previous crops gets increased
            patch_ordering = np.where(
                patch_ordering >= 0,
                patch_ordering + tokens_per_image,
                -1
            )
            base_ordering = np.arange(tokens_per_image).reshape(16, 16)
            patch_ordering = np.concatenate([base_ordering[None], patch_ordering], axis=0)
            # Old way to do it
            # patch_ordering = np.concatenate([np.arange(0, tokens_per_image), patch_ordering], 0)
            per_row = np.full(
                (image_token_length_w,),
                self.image_patch_token_id,
                dtype=np.int32
            )
            if self.use_col_tokens:
                per_row = np.concatenate([per_row, [self.image_col_token_id]], 0)
            extra_tokens = np.tile(per_row, [image_token_length_h])
            joint = [
                        [self.image_start_token_id],
                        extra_tokens,
                        [self.image_end_token_id],
                    ] + joint

            joint = np.concatenate(joint, 0)
            img_mask = np.pad(img_mask, [[0, 1], [0, 0]], constant_values=-1)
            mask_arr = np.concatenate([np.expand_dims(mask, 0), mask_arr], 0)
            return patches, tiling, patch_ordering, mask_arr
            # Original return statement
            # return patches, joint, patch_ordering, img_mask
        else:
            raise NotImplementedError(self.crop_mode)

    def build_image_input_idx(
        self,
        image_tokens: np.ndarray,
        patch_order: np.ndarray,
    ):
        """Converts `patch_order` into an array mapping patch_id -> token_position"""
        tokens_per_image = self.image_token_length_w * self.image_token_length_h

        image_input_idx = image_tokens == self.image_patch_token_id
        image_input_idx = np.nonzero(image_input_idx)[0].astype(np.int32)

        n_tokens = image_input_idx.shape[0]

        if patch_order is not None:
            patch_order = np.reshape(patch_order, [-1])
            n_patches = patch_order.shape[0]

            valid = patch_order >= 0
            n_valid_patches = valid.sum()
            assert len(image_input_idx) == n_valid_patches

            # Get the reversed mapping of patch order (so instead of sorted position->patch_idx we
            # want patch_idx->sorted position)
            # We have to be careful to preserve the sparse structure of `patch_order` where -1 means
            # a patch is skipped
            sorted_patch_ixs = np.zeros([n_tokens], np.int32)
            sorted_patch_ixs[patch_order[valid]] = np.arange(n_valid_patches, dtype=np.int32)
            sorted_patch_ixs_ex = np.full(np.shape(patch_order), -1)
            sorted_patch_ixs_ex[valid] = sorted_patch_ixs

            # Now go from patch_idx->sorted position to patch_idx->tokens position, we need to do
            # this since the `image_tokens`` will contain special tokens interleave with the
            # tokens that will become image features
            valid = (sorted_patch_ixs_ex >= 0).astype(np.int32)
            image_input_idx = image_input_idx[sorted_patch_ixs_ex*valid]
            image_input_idx = image_input_idx*valid - 100*(1 - valid)

        image_input_idx = np.reshape(image_input_idx, [-1, tokens_per_image])
        return image_input_idx

    def preprocess(self, image, is_training: bool, rng=None):
        """Preprocesses a single image

        Returns:
            crops: (n_crops, n_patches, patch_dim) individual crops, `n_crops` might
                   change between images but the other dimension are fixed
            tokens: (n_tokens,) int32 tokens, pad tokens indicate where to insert the
                                patch features, might include other special tokens as well
            image_idx: (n_crops, n_patches) index in `tokens` to put the patch features from the
                       crops after pooling, negative values indicates patches features to exclude
            padding_mask: (n_crops, n_patches) what percent of each crop is padding, can be None
                          if the image mask is not being used.
        """
        crops, image_tokens, patch_ordering, img_mask = self.image_to_patches_and_tokens(
            image, is_training, rng)
        # I need to fix this function       
        patch_idx = self.build_image_input_idx(
            image_tokens,
            patch_ordering,
        )
        return crops, image_tokens, patch_idx, img_mask

    def reconstruct(self, reconstructeed_patches, tiling, patch_orderings, masks):
        """Reconstructs the original image from the patches"""
        # trim the crops to remove the overlaps between them
        trims_overlap = self.count_invalid_edges(patch_orderings, invalid_value=-1)
        trims_overlap = trims_overlap * self.image_patch_size  # Convert to pixels
        crops_no_overlap = self.apply_trims_to_crops(reconstructeed_patches, trims_overlap)

        # The masks are in the same format as the patch orderings, so we can use the same trimming
        # logic to trim the padding from the edges of the crops
        trims_padding = self.count_invalid_edges(masks, invalid_value=False)
        trimmed_crops = self.apply_trims_to_crops(crops_no_overlap, trims_padding)

        tiles_x, tiles_y = tiling
        # reshape the crops into a grid
        # The first crop is the global image, the rest are the crops
        grid = np.array(trimmed_crops[1:], dtype=object).reshape(tiles_x, tiles_y)
        rows = [np.concatenate(grid[y, :], axis=1) for y in range(grid.shape[0])]
        full_image = np.concatenate(rows, axis=0)

        return full_image, trimmed_crops[0]  # Return the full image and the first crop (global image)
    
    def count_invalid_edges(self, masks, invalid_value=-1):
        """
        Counts how many full rows/columns can be trimmed from the edges of each crop,
        based on a boolean or binary mask where `invalid_value` indicates trim regions.

        Args:
            masks: np.ndarray of shape (N, H, W), typically:
                - patch_orderings != -1 → (N, 16, 16)
                - pixel masks (e.g., masks != 0) → (N, 256, 256)
            invalid_value: the value considered as 'invalid' (default: False)

        Returns:
            trims: np.ndarray of shape (N, 4), with (left, top, right, bottom) per crop
        """
        N, H, W = masks.shape
        trims = np.zeros((N, 4), dtype=int)

        for i in range(N):
            valid = masks[i] != invalid_value  # shape: (H, W)
            # Count left
            for x in range(W):
                if np.any(valid[:, x]):
                    break
                trims[i, 0] += 1  # left
            # Count top
            for y in range(H):
                if np.any(valid[y, :]):
                    break
                trims[i, 1] += 1  # top
            # Count right
            for x in reversed(range(W)):
                if np.any(valid[:, x]):
                    break
                trims[i, 2] += 1  # right
            # Count bottom
            for y in reversed(range(H)):
                if np.any(valid[y, :]):
                    break
                trims[i, 3] += 1  # bottom

        return trims
    
    def apply_trims_to_crops(self, crops, trims):
        """
        Trim each crop according to the corresponding trims.

        Args:
            crops: list or np.array of shape (N, H, W, C)
            trims: np.array of shape (N, 4) with (left, top, right, bottom) pixel counts to trim

        Returns:
            trimmed_crops: list of cropped numpy arrays
        """
        trimmed_crops = []
        for i in range(len(crops)):
            left, top, right, bottom = trims[i]
            crop = crops[i]
            trimmed_crop = crop[
                top : crop.shape[0] - bottom,
                left : crop.shape[1] - right,
                :
            ]
            trimmed_crops.append(trimmed_crop)
        return trimmed_crops

# Example usage of the MultiModalPreprocessor class
if __name__ == "__main__":
    preprocessor = MultiModalPreprocessor(
        tokenizer=None,  # Replace with actual tokenizer
        crop_mode="overlap-and-resize-c2",
        max_crops=12,
        overlap_margins=(2, 2),
        resize="default",
        use_col_tokens=True,
        base_image_input_size=(256, 256),
        image_pooling_w=1,
        image_pooling_h=1,
        image_token_length_w=16,
        image_token_length_h=16,
        image_patch_size=16,
        image_padding_mask=False,
        pad_value=0.0
    )

    # Example usage
    image = load_image("/users/nirmiger/benchmark-image-tokenzier/assets/original/physics1.png")
    crops, tiling, patch_ordering, masks = preprocessor.image_to_patches_and_tokens(image, is_training=True, rng=np.random.default_rng())
    print("Crops shape:", crops.shape)
    print("Tiling:", tiling)
    print("Patch ordering shape:", patch_ordering.shape)
    print("Image mask shape:", masks.shape)
    breakpoint()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    images_to_plot = (crops * 255).astype(np.uint8)

    # Plot
    fig, axes = plt.subplots(1, len(crops), figsize=(20, 3))
    for i in range(len(crops)):
        axes[i].imshow(images_to_plot[i])
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('crops_plot.png', dpi=150)
    plt.close()

    full_grid, first_crop = preprocessor.reconstruct(crops, tiling, patch_ordering, masks)
    first_image = (first_crop * 255).clip(0, 255).astype(np.uint8)
    PIL.Image.fromarray(first_image).save("first_crop_image.png")
    full_image = (full_grid * 255).clip(0, 255).astype(np.uint8)
    PIL.Image.fromarray(full_image).save("reconstructed_image.png")

