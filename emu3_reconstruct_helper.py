#!/usr/bin/env python3
"""
EMU3 Image Reconstruction Helper
Validates and reconstructs images from generated visual tokens
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel


def validate_token_structure(token_ids: List[int], special_token_ids: Dict[str, int]) -> Dict:
    """
    Validate the structure of generated tokens including checking row consistency.

    Args:
        token_ids: List of token IDs from model generation
        special_token_ids: Dictionary mapping special token names to IDs

    Returns:
        Dictionary with validation results
    """
    eol_id = special_token_ids.get("img_end_of_row", -1)
    eof_id = special_token_ids.get("img_end_of_frame", -1)
    eoi_id = special_token_ids.get("img_end", -1)

    # Extract visual token IDs from vision_mapping
    # This would need the reverse mapping from the inferencer

    rows = []
    current_row = []
    visual_token_count = 0

    for token_id in token_ids:
        if token_id == eol_id:
            # End of row
            rows.append(len(current_row))
            current_row = []
        elif token_id == eof_id or token_id == eoi_id:
            # End of frame/image
            if current_row:
                rows.append(len(current_row))
            break
        elif token_id not in special_token_ids.values():
            # Likely a visual token (would need proper check with vision_mapping)
            current_row.append(token_id)
            visual_token_count += 1

    # Check row consistency
    rows_consistent = len(set(rows)) <= 1 if rows else False

    validation = {
        "total_tokens": len(token_ids),
        "visual_tokens": visual_token_count,
        "num_rows": len(rows),
        "tokens_per_row": rows,
        "rows_consistent": rows_consistent,
        "row_width": rows[0] if rows and rows_consistent else None,
        "inferred_shape": (len(rows), rows[0]) if rows and rows_consistent else None,
    }

    if not rows_consistent and rows:
        validation["row_variance"] = {"min": min(rows), "max": max(rows), "mean": np.mean(rows), "std": np.std(rows)}

    return validation


def extract_visual_tokens_by_row(
    token_ids: List[int], vision_mapping: Dict[int, int], special_token_ids: Dict[str, int]
) -> Tuple[List[List[int]], Dict]:
    """
    Extract visual tokens organized by rows.

    Args:
        token_ids: List of token IDs from model
        vision_mapping: Mapping from visual index to token ID
        special_token_ids: Special token IDs

    Returns:
        (rows_of_visual_indices, statistics)
    """
    # Create reverse mapping
    reverse_mapping = {v: k for k, v in vision_mapping.items()}

    eol_id = special_token_ids.get("img_end_of_row", -1)
    eof_id = special_token_ids.get("img_end_of_frame", -1)
    eoi_id = special_token_ids.get("img_end", -1)

    rows = []
    current_row = []

    for token_id in token_ids:
        if token_id == eol_id:
            if current_row:
                rows.append(current_row)
                current_row = []
        elif token_id == eof_id or token_id == eoi_id:
            if current_row:
                rows.append(current_row)
            break
        elif token_id in reverse_mapping:
            # It's a visual token
            visual_idx = reverse_mapping[token_id]
            current_row.append(visual_idx)

    # Calculate statistics
    row_lengths = [len(row) for row in rows]
    stats = {
        "num_rows": len(rows),
        "row_lengths": row_lengths,
        "consistent": len(set(row_lengths)) <= 1,
        "total_visual_tokens": sum(row_lengths),
        "shape": (len(rows), row_lengths[0]) if row_lengths and len(set(row_lengths)) == 1 else None,
    }

    return rows, stats


def reconstruct_with_emu3_tokenizer(
    visual_indices: List[int],
    height: int,
    width: int,
    tokenizer_model_path: str = "BAAI/Emu3-VisionTokenizer",
    device: str = "cuda",
) -> torch.Tensor:
    """
    Reconstruct image using EMU3 vision tokenizer's decode method.

    Args:
        visual_indices: List of visual token indices
        height: Height in tokens
        width: Width in tokens
        tokenizer_model_path: Path to EMU3 vision tokenizer
        device: Device to use

    Returns:
        Reconstructed image tensor
    """
    print(f"Loading EMU3 Vision Tokenizer from {tokenizer_model_path}")

    # Load the vision tokenizer model
    model = AutoModel.from_pretrained(tokenizer_model_path, trust_remote_code=True).eval().to(device)

    # Convert indices to tensor and reshape
    indices_tensor = torch.tensor(visual_indices, dtype=torch.long).to(device)

    # Reshape to [batch, height, width] format expected by decoder
    if len(visual_indices) != height * width:
        print(f"Warning: Expected {height*width} tokens, got {len(visual_indices)}")
        # Pad or truncate as needed
        if len(visual_indices) < height * width:
            # Pad with zeros
            padding = [0] * (height * width - len(visual_indices))
            visual_indices = visual_indices + padding
        else:
            # Truncate
            visual_indices = visual_indices[: height * width]
        indices_tensor = torch.tensor(visual_indices, dtype=torch.long).to(device)

    indices_tensor = indices_tensor.reshape(1, height, width)  # Add batch dimension

    print(f"Decoding indices tensor of shape: {indices_tensor.shape}")

    # Decode using the model
    with torch.no_grad():
        reconstructed = model.decode(indices_tensor)

    print(f"Reconstructed tensor shape: {reconstructed.shape}")

    return reconstructed


def visualize_reconstruction(
    reconstructed_tensor: torch.Tensor,
    visual_indices: List[int],
    height: int,
    width: int,
    save_path: Optional[str] = None,
):
    """
    Visualize reconstructed image alongside token distribution.

    Args:
        reconstructed_tensor: Reconstructed image tensor from decoder
        visual_indices: Original visual token indices
        height: Height in tokens
        width: Width in tokens
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Convert tensor to displayable format
    if reconstructed_tensor.dim() == 4:  # [batch, channels, height, width]
        img = reconstructed_tensor[0].permute(1, 2, 0).cpu().numpy()
    elif reconstructed_tensor.dim() == 3:  # [channels, height, width]
        img = reconstructed_tensor.permute(1, 2, 0).cpu().numpy()
    else:
        img = reconstructed_tensor.cpu().numpy()

    # Normalize to [0, 1] for display
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Left: Reconstructed image
    axes[0].imshow(img)
    axes[0].set_title(f"Reconstructed Image\n({height}×{width} tokens)")
    axes[0].axis("off")

    # Middle: Token index heatmap
    token_grid = np.array(visual_indices[: height * width]).reshape(height, width)
    im = axes[1].imshow(token_grid, cmap="viridis", aspect="auto")
    axes[1].set_title("Visual Token Indices")
    axes[1].set_xlabel("Width (tokens)")
    axes[1].set_ylabel("Height (tokens)")
    plt.colorbar(im, ax=axes[1])

    # Right: Token distribution
    axes[2].hist(visual_indices, bins=50, edgecolor="black", alpha=0.7)
    axes[2].set_title("Token Index Distribution")
    axes[2].set_xlabel("Token Index")
    axes[2].set_ylabel("Frequency")

    # Add statistics
    unique_tokens = len(set(visual_indices))
    axes[2].text(
        0.65,
        0.95,
        f"Total: {len(visual_indices)}\nUnique: {unique_tokens}\n"
        f"Min: {min(visual_indices)}\nMax: {max(visual_indices)}",
        transform=axes[2].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    plt.show()

    return fig


def analyze_generation_pattern(rows_of_indices: List[List[int]]) -> Dict:
    """
    Analyze the pattern of generated visual tokens across rows.

    Args:
        rows_of_indices: List of rows, each containing visual token indices

    Returns:
        Dictionary with pattern analysis
    """
    analysis = {}

    # Row-wise statistics
    row_lengths = [len(row) for row in rows_of_indices]
    analysis["row_count"] = len(rows_of_indices)
    analysis["row_lengths"] = {
        "values": row_lengths,
        "consistent": len(set(row_lengths)) == 1,
        "min": min(row_lengths) if row_lengths else 0,
        "max": max(row_lengths) if row_lengths else 0,
        "mean": np.mean(row_lengths) if row_lengths else 0,
        "std": np.std(row_lengths) if row_lengths else 0,
    }

    # Token distribution across rows
    all_tokens = [token for row in rows_of_indices for token in row]
    if all_tokens:
        analysis["token_distribution"] = {
            "total": len(all_tokens),
            "unique": len(set(all_tokens)),
            "min": min(all_tokens),
            "max": max(all_tokens),
            "mean": np.mean(all_tokens),
            "std": np.std(all_tokens),
        }

        # Check for patterns
        # Are tokens increasing/decreasing?
        diffs = np.diff(all_tokens)
        analysis["token_progression"] = {
            "mostly_increasing": np.sum(diffs > 0) > len(diffs) * 0.7,
            "mostly_decreasing": np.sum(diffs < 0) > len(diffs) * 0.7,
            "mostly_constant": np.sum(diffs == 0) > len(diffs) * 0.5,
        }

    return analysis


# Example usage
if __name__ == "__main__":
    # This would be used with your inference results
    print("EMU3 Reconstruction Helper loaded")
    print("Use the functions to:")
    print("1. validate_token_structure() - Check if rows are consistent")
    print("2. extract_visual_tokens_by_row() - Organize tokens by rows")
    print("3. reconstruct_with_emu3_tokenizer() - Decode tokens to image")
    print("4. visualize_reconstruction() - Show reconstructed image")
    print("5. analyze_generation_pattern() - Analyze token patterns")
