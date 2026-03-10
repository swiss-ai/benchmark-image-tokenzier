#!/usr/bin/env python3
"""
Test conditional image generation with EMU3
Provide first quarter of real image tokens and let model complete the rest
"""

import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from Tokenizer.Emu3VisionTokenizer import Emu3VisionTokenizer


def tokenize_real_image(image_path, device="cuda:1"):
    """
    Tokenize a real image to get ground truth tokens and dimensions
    """
    print(f"Loading and tokenizing image: {image_path}")

    # Load image
    img = Image.open(image_path).convert("RGB")
    print(f"  Original image size: {img.size}")

    # Initialize tokenizer on specified device
    tokenizer = Emu3VisionTokenizer()
    tokenizer.model = tokenizer.model.to(device)
    tokenizer.device = device

    # Preprocess and encode (EMU3 will handle the sizing)
    img_tensor = tokenizer.preprocess(img)
    img_tensor = img_tensor.to(device)

    # Encode to get token indices
    with torch.no_grad():
        indices, _ = tokenizer.encode(img_tensor)

    # Get actual token dimensions
    # indices shape: [batch, height, width]
    token_height = indices.shape[1]
    token_width = indices.shape[2]
    print(f"  Token dimensions: {token_height}×{token_width}")

    visual_indices = indices[0].flatten().cpu().tolist()

    # Clean up
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return visual_indices, img, token_height, token_width


def create_partial_prompt(visual_indices, height, width, given_rows):
    """
    Create a prompt with first few rows of the image
    """
    prompt_parts = []

    # Add metadata
    prompt_parts.append(f"<|begin_of_text|><|img_start|>{height}*{width}<|img_token_start|>")

    # Add first quarter of visual tokens with proper EOL structure
    for row in range(given_rows):
        row_start = row * width
        row_end = row_start + width
        row_tokens = visual_indices[row_start:row_end]

        # Add visual tokens for this row
        for token_idx in row_tokens:
            prompt_parts.append(f"<|visual token {token_idx:06d}|>")

        # Add EOL after each row
        prompt_parts.append("<|img_end_of_row|>")

    prompt = "".join(prompt_parts)

    # Calculate expected remaining tokens
    remaining_rows = height - given_rows
    expected_remaining = remaining_rows * width

    return prompt, expected_remaining


def test_conditional_generation(
    inferencer,
    image_path="/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/assets/original/logo1.png",
    given_rows=3,
):
    """
    Test conditional generation with real image tokens

    Args:
        inferencer: EMU3 inferencer
        image_path: Path to image to use as reference
        given_rows: Number of rows to provide as context (default 3)
    """
    print("=" * 80)
    print("CONDITIONAL IMAGE GENERATION TEST")
    print("=" * 80)

    # 1. Tokenize the real image
    print("\n1. Tokenizing real image...")
    try:
        visual_indices, original_img, token_height, token_width = tokenize_real_image(image_path, device="cuda:1")
        print(f"✓ Got {len(visual_indices)} tokens from real image")
        print(f"  Token dimensions: {token_height}×{token_width}")
        print(f"  Token range: {min(visual_indices)} - {max(visual_indices)}")
        print(f"  Unique tokens: {len(set(visual_indices))}")
    except Exception as e:
        print(f"❌ Failed to tokenize image: {e}")
        return

    # 2. Create prompt with first few rows
    print(f"\n2. Creating prompt with first {given_rows} rows ({given_rows * token_width} tokens)...")
    prompt, expected_remaining = create_partial_prompt(visual_indices, token_height, token_width, given_rows)
    print(f"✓ Prompt created with {given_rows * token_width} given tokens")
    print(f"  Expected remaining: {expected_remaining} tokens")

    # 3. Generate completion
    print("\n3. Generating completion...")
    result = inferencer.generate(
        prompt,
        max_tokens=expected_remaining + 200,  # Allow some extra for structure tokens
        temperature=0.3,  # Low temperature for more deterministic completion
    )

    print(f"✓ Generated {result['num_visual_tokens']} visual tokens")
    print(f"  Structure tokens: {result.get('structure_tokens', {})}")

    # 4. Extract and analyze the completed tokens
    from emu3_reconstruct_helper import extract_visual_tokens_by_row

    # Combine given and generated tokens
    given_indices = visual_indices[: given_rows * token_width]  # First N rows of tokens

    # Extract generated visual tokens
    rows_generated, _ = extract_visual_tokens_by_row(
        result["generated_token_ids"], inferencer.vision_mapping, inferencer.special_token_ids
    )

    generated_indices = []
    for row in rows_generated:
        generated_indices.extend(row)

    # Combine all tokens
    all_indices = given_indices + generated_indices

    print(f"\n4. Token Analysis:")
    print(f"  Given tokens: {len(given_indices)}")
    print(f"  Generated tokens: {len(generated_indices)}")
    print(f"  Total tokens: {len(all_indices)}")

    expected_total = token_height * token_width
    if len(all_indices) == expected_total:
        print(f"  ✅ Correct total count ({expected_total})")
    else:
        print(f"  ⚠️ Token count mismatch (expected {expected_total}, got {len(all_indices)})")

    # 5. Reconstruct and visualize
    if len(all_indices) >= expected_total:
        print("\n5. Reconstructing completed image...")
        all_indices = all_indices[:expected_total]  # Ensure exact token count

        try:
            # Initialize tokenizer for reconstruction
            tokenizer = Emu3VisionTokenizer()
            tokenizer.model = tokenizer.model.to("cuda:1")
            tokenizer.device = "cuda:1"

            # Create tensor and decode
            indices_tensor = torch.tensor(all_indices, dtype=torch.long, device="cuda:1")
            indices_tensor = indices_tensor.reshape(1, token_height, token_width)

            with torch.no_grad():
                reconstructed_tensor = tokenizer.decode(indices_tensor)

            reconstructed_pil = tokenizer.postprocess(reconstructed_tensor)

            # Clean up
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()

            # Visualize results
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Row 1: Original and token analysis
            axes[0, 0].imshow(original_img)
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis("off")

            # Show given vs generated boundary
            token_grid = np.array(all_indices).reshape(token_height, token_width)
            masked_grid = token_grid.copy()
            masked_grid[given_rows:, :] = masked_grid[given_rows:, :] * 0.5  # Dim the generated part

            axes[0, 1].imshow(masked_grid, cmap="viridis")
            axes[0, 1].axhline(y=given_rows, color="red", linewidth=2, label="Given/Generated boundary")
            axes[0, 1].set_title(f"Token Map (top {given_rows} rows given)")
            axes[0, 1].legend()

            axes[0, 2].imshow(reconstructed_pil)
            axes[0, 2].set_title("Reconstructed (Given + Generated)")
            axes[0, 2].axis("off")

            # Row 2: Token statistics
            axes[1, 0].hist(given_indices, bins=30, alpha=0.7, label="Given tokens", color="blue")
            axes[1, 0].set_title("Given Token Distribution")
            axes[1, 0].set_xlabel("Token Index")
            axes[1, 0].set_ylabel("Frequency")

            axes[1, 1].hist(generated_indices, bins=30, alpha=0.7, label="Generated tokens", color="orange")
            axes[1, 1].set_title("Generated Token Distribution")
            axes[1, 1].set_xlabel("Token Index")
            axes[1, 1].set_ylabel("Frequency")

            # Compare distributions
            axes[1, 2].hist(given_indices, bins=30, alpha=0.5, label="Given", color="blue")
            axes[1, 2].hist(generated_indices, bins=30, alpha=0.5, label="Generated", color="orange")
            axes[1, 2].set_title("Token Distribution Comparison")
            axes[1, 2].set_xlabel("Token Index")
            axes[1, 2].set_ylabel("Frequency")
            axes[1, 2].legend()

            plt.suptitle("Conditional Image Generation Results", fontsize=16)
            plt.tight_layout()

            # Save results
            save_path = "conditional_generation_result.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved visualization to {save_path}")

            plt.show()

            print("\n✅ Conditional generation test complete!")

        except Exception as e:
            print(f"❌ Reconstruction failed: {e}")
            import traceback

            traceback.print_exc()

    return result


if __name__ == "__main__":
    print("This script should be imported and used in a notebook with an active inferencer")
    print("Example usage:")
    print("  from test_conditional_generation import test_conditional_generation")
    print("  result = test_conditional_generation(inferencer)")
