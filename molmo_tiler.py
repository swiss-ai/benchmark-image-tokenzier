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
    pad_value=-1,
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
    mode = InterpolationMode.BICUBIC
    image = torchvision.transforms.Resize([scaled_height, scaled_width], mode, antialias=True)(image)
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

    def resize_image(self, image, output_size, is_training, rng):
        return resize_and_pad(
            image, output_size, pad_value=self.pad_value, rng=rng, is_training=is_training)


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

            # Correction for non-spatial tokenizers where the numbers do not fully add up
            if overlap_margins == (0,0):
                crop_window_size = crop_size

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
            # Now we have to split the image into crops, while keeping track of how each patch in the
            # each crop should be ordered in the global image, this require a lot of tricky booking
            n_crops = tiling[0] * tiling[1]
            patches_arr = []
            mask_arr = []
            patch_ordering_arr = []

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

            img_mask = img_mask.astype(np.float32).mean(axis=-1)

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

            patches = np.concatenate([np.expand_dims(resized, 0), patches], 0)

            # Global image goes first, so the order of patches in previous crops gets increased
            patch_ordering = np.where(
                patch_ordering >= 0,
                patch_ordering + tokens_per_image,
                -1
            )
            base_ordering = np.arange(tokens_per_image).reshape(self.image_token_length_h, self.image_token_length_w)
            patch_ordering = np.concatenate([base_ordering[None], patch_ordering], axis=0)

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
            return torch.tensor(patches), tiling, patch_ordering, mask_arr
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
        crops_grid = [
            trimmed_crops[1 + y * tiles_y : 1 + (y + 1) * tiles_y]
            for y in range(tiles_x)
        ]
        rows = [np.concatenate(row, axis=1) for row in crops_grid]
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
    image = load_image("/users/nirmiger/benchmark-image-tokenzier/assets/original/math_draft1.png")
    crops, tiling, patch_ordering, masks = preprocessor.image_to_patches_and_tokens(image, is_training=True, rng=np.random.default_rng())
    print("Crops shape:", crops.shape)
    print("Tiling:", tiling)
    print("Patch ordering shape:", patch_ordering.shape)
    print("Image mask shape:", masks.shape)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    crops = crops.numpy()
    images_to_plot = crops.astype(np.uint8)

    patch_size = 16  # Patch size assumed from earlier
    tiling_rows, tiling_cols = tiling  # e.g., 3, 2
    crop_count = len(crops)

    # Plot setup: one figure, but we manually position axes
    fig = plt.figure(figsize=(tiling_cols * 4, (tiling_rows + 1) * 4))  # +1 row for first crop

    # Function to plot a single crop at a given location in the figure
    def plot_crop(ax, img, patch_mask):
        ax.imshow(img)
        ax.axis('off')

        h_patches, w_patches = patch_mask.shape
        for y in range(h_patches):
            for x in range(w_patches):
                if patch_mask[y, x] == -1:
                    rect = patches.Rectangle(
                        (x * patch_size, y * patch_size), patch_size, patch_size,
                        linewidth=0, edgecolor=None, facecolor='black', alpha=0.3
                    )
                    ax.add_patch(rect)

    # --- First crop (plotted alone at the top) ---
    ax_first = fig.add_subplot(tiling_rows + 1, tiling_cols, 1)  # First cell in the top row
    plot_crop(ax_first, images_to_plot[0], patch_ordering[0])
    fig.text(0.65, 0.8, "Global overview (left) and corresponding crops (below).\nOverlapping regions are shaded", fontsize=16, ha='center', va='center')

    # --- Rest of the crops (laid out in tiling grid, starting from the second one) ---
    index = 1  # Start from second image
    for r in range(tiling_rows):
        for c in range(tiling_cols):
            if index >= crop_count:
                break
            # Offset by tiling_cols to start from second row in figure grid
            ax = fig.add_subplot(tiling_rows + 1, tiling_cols, (r + 1) * tiling_cols + c + 1)
            plot_crop(ax, images_to_plot[index], patch_ordering[index])
            index += 1

    plt.tight_layout(pad=2.0)  # Add spacing between subplots
    plt.savefig('rearranged_crops_plot.png', dpi=150)
    plt.close()

    full_grid, first_crop = preprocessor.reconstruct(crops, tiling, patch_ordering, masks)
    full_image = full_grid.clip(0, 255).astype(np.uint8)
    PIL.Image.fromarray(full_image).save("reconstructed_image.png")

