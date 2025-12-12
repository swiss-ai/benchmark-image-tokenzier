import argparse
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from lpips import LPIPS
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms

# Path to the assets folder
ASSETS_FOLDER = "assets"

# Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform for SSIM (preserve original size)
basic_transform = transforms.ToTensor()

lpips_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),  # scales [0,1] to [-1,1]
    ]
)

# Transform specifically for FID (resize to 299x299 for InceptionV3)
fid_transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.PILToTensor(),
    ]
)

# Initialize LPIPS model
warnings.filterwarnings("ignore", category=UserWarning)
lpips_model = LPIPS(net="alex").to(device)


# Function to round ratio values in folder names
def round_ratio_in_name(folder_name):
    """Round ratio values in folder names to 2 decimal places"""
    if "ratio" in folder_name.lower():
        # Find all decimal numbers in the string
        def round_match(match):
            number = float(match.group())
            return f"{number:.2f}"

        # Replace decimal numbers with rounded versions
        rounded_name = re.sub(r"\d+\.\d+", round_match, folder_name)
        return rounded_name
    return folder_name


# Function to calculate FID
def calculate_fid(original_images, generated_images):
    fid = FrechetInceptionDistance(feature=64).to(device)
    for img in original_images:
        fid.update(fid_transform(img).unsqueeze(0).to(device), real=True)
    for img in generated_images:
        fid.update(fid_transform(img).unsqueeze(0).to(device), real=False)
    return fid.compute().item()


# Function to load images from a folder
def load_images_from_folder(folder_path, original=False):
    images = []
    numbers = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            images.append(Image.open(img_path).convert("RGB"))
            if not original:
                base_name = os.path.splitext(filename)[0]
                number_str = base_name.split("_")[-1]
                number = int(number_str)
                numbers.append(number)
    return images, numbers


# Helper function for output filenames
def get_output_filename(base_name, prefix=""):
    """Generate output filename with optional prefix."""
    return f"{prefix}{base_name}" if prefix else base_name


# Function to detect mode and generate prefix
def detect_mode_and_prefix(comparison_path, output_prefix=None):
    """
    Detect single vs multi-folder mode and generate appropriate output prefix.

    Args:
        comparison_path: Path to comparison folder(s)
        output_prefix: Optional manual prefix override

    Returns:
        tuple: (folders_to_process, final_output_prefix)
            - folders_to_process: List of (folder_name, folder_path) tuples
            - final_output_prefix: Prefix for output files
    """
    # Check if comparison_path contains images directly
    try:
        items = os.listdir(comparison_path)
    except (FileNotFoundError, NotADirectoryError) as e:
        raise ValueError(f"Comparison path does not exist: {comparison_path}") from e

    image_files = [f for f in items if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if image_files:
        # Single folder mode: process comparison_path directly
        folder_name = os.path.basename(comparison_path.rstrip("/\\"))
        folders_to_process = [(folder_name, comparison_path)]

        # Auto-generate prefix if not provided
        if output_prefix is None:
            # Sanitize folder name: lowercase, replace special chars with underscores
            sanitized = re.sub(r"[^a-z0-9]+", "_", folder_name.lower()).strip("_")
            final_output_prefix = f"{sanitized}_"
        else:
            final_output_prefix = output_prefix

    else:
        # Multi-folder mode: process all subdirectories
        folders_to_process = [
            (folder_name, os.path.join(comparison_path, folder_name))
            for folder_name in items
            if os.path.isdir(os.path.join(comparison_path, folder_name)) and folder_name != "original"
        ]

        # No auto-prefix in multi-folder mode
        final_output_prefix = output_prefix if output_prefix else ""

    return folders_to_process, final_output_prefix


# Main function to calculate metrics
def calculate_metrics(original_path="assets/original", comparison_path="assets", output_prefix=None):
    """
    Calculate reconstruction metrics between original and comparison images.

    Args:
        original_path: Path to folder containing original images
        comparison_path: Path to comparison folder(s) - can be parent folder or single folder
        output_prefix: Optional prefix for output files (auto-generated in single folder mode)
    """
    if not os.path.exists(original_path):
        print(f"Error: Original path does not exist: {original_path}")
        return

    if not os.path.exists(comparison_path):
        print(f"Error: Comparison path does not exist: {comparison_path}")
        return

    original_images, _ = load_images_from_folder(original_path, original=True)
    if not original_images:
        print(f"No original images found in: {original_path}")
        return

    # Detect mode and generate prefix
    folders_to_process, output_prefix = detect_mode_and_prefix(comparison_path, output_prefix)

    if not folders_to_process:
        print(f"No comparison folders found in: {comparison_path}")
        return

    results = []

    for folder_name, folder_path in folders_to_process:
        print(f"\nCalculating metrics for folder: {folder_name}")
        generated_images, numbers = load_images_from_folder(folder_path)

        if len(original_images) != len(generated_images):
            print(f"  Skipping folder {folder_name}: mismatched image counts.")
            continue

        psnr_values = []
        ssim_values = []
        lpips_values = []
        used_tokens = []

        original_images_resized = []
        generated_images_resized = []

        for orig, gen in zip(original_images, generated_images):
            ow, oh = orig.size
            gw, gh = gen.size

            # Determine target size
            target_w = min(ow, gw)
            target_h = min(oh, gh)

            # Resize whichever image is larger
            if (ow, oh) != (target_w, target_h):
                orig = orig.resize((target_w, target_h), Image.BICUBIC)
            if (gw, gh) != (target_w, target_h):
                gen = gen.resize((target_w, target_h), Image.BICUBIC)

            original_images_resized.append(orig)
            generated_images_resized.append(gen)

        for orig_resized, gen_resized, tokens in zip(original_images_resized, generated_images_resized, numbers):
            # Convert to numpy arrays for PSNR and SSIM
            orig_np = np.array(orig_resized)
            gen_np = np.array(gen_resized)
            if "unitok" in folder_name:
                tokens = tokens / 8
            used_tokens.append(tokens)
            # PSNR
            psnr_values.append(psnr(orig_np, gen_np))
            # SSIM
            ssim_values.append(ssim(orig_np, gen_np, channel_axis=-1))
            # LPIPS
            orig_tensor = lpips_transform(orig_resized).unsqueeze(0).to(device)
            gen_tensor = lpips_transform(gen_resized).unsqueeze(0).to(device)
            lpips_score = lpips_model(orig_tensor, gen_tensor).item()
            lpips_values.append(lpips_score)

        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        avg_lpips = np.mean(lpips_values)
        avg_tokens = np.mean(used_tokens)

        # FID remains unchanged, original and generated images as is
        fid_value = calculate_fid(original_images_resized, generated_images_resized)

        # Round ratio values in folder name for consistent display
        display_folder_name = round_ratio_in_name(folder_name)

        results.append(
                {
                "Folder": display_folder_name,
                "PSNR": avg_psnr,
                "SSIM": avg_ssim,
                "LPIPS": avg_lpips,
                "#Tokens": avg_tokens,
            }
        )

        # Results
        print(f"  Average PSNR:  {avg_psnr:.2f}")
        print(f"  Average SSIM:  {avg_ssim:.4f}")
        print(f"  Average LPIPS: {avg_lpips:.4f}")
        print(f"  FID:           {fid_value:.2f}")
        print(
            "We currently do not have enough data to calculate the true FID or rFID. Do not use this value for any serious evaluation."
        )

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df_sorted = df.sort_values(by="LPIPS")
        output_csv = get_output_filename("metrics_results.csv", output_prefix)
        df_sorted.to_csv(output_csv, index=False)
        print(f"\nSaved metrics to {output_csv}")

        # Save pretty Markdown
        output_md = get_output_filename("metrics_results.md", output_prefix)
        with open(output_md, "w") as f:
            f.write(df_sorted.to_markdown(index=False))

        sns.set_theme(style="whitegrid", font_scale=1.1)

        # Use a larger, more diverse color palette (tab20 has 20 colors, extend if needed)
        unique_folders = list(df["Folder"].unique())
        palette = sns.color_palette("tab20", n_colors=len(unique_folders))
        palette_dict = dict(zip(unique_folders, palette))

        # --- Sorted LPIPS barplot ---
        plt.figure(figsize=(14, 6))
        sorted_df = df.sort_values(by="LPIPS", ascending=True)

        ax = sns.barplot(data=sorted_df, x="Folder", y="LPIPS", palette=[palette_dict[f] for f in sorted_df["Folder"]])
        ax.set_title("LPIPS Scores per Tokenizer\n(Lower is Better)", fontsize=16, weight="bold")
        ax.set_xlabel("Tokenizer", fontsize=12)
        ax.set_ylabel("LPIPS", fontsize=12)
        plt.xticks(rotation=40, ha="right", fontsize=9)

        # Annotate LPIPS values on bars
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.3f}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                xytext=(0, 3),
                textcoords="offset points",
            )

        plt.tight_layout()
        output_plot1 = get_output_filename("lpips_comparison_plot.png", output_prefix)
        plt.savefig(output_plot1, dpi=300, bbox_inches="tight")
        plt.close()

        # --- LPIPS vs #Tokens (lowest LPIPS) ---
        top20 = df.nsmallest(20, "LPIPS")

        plt.figure(figsize=(14, 6))
        ax = sns.scatterplot(
            data=top20,
            x="#Tokens",
            y="LPIPS",
            hue="Folder",
            palette=palette_dict,
            s=120,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_title("LPIPS vs Average #Tokens", fontsize=16, weight="bold")
        ax.set_xlabel("#Tokens", fontsize=12)
        ax.set_ylabel("LPIPS", fontsize=12)

        leg = ax.legend(
            title="Tokenizer",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=8,
            markerscale=1.2,
            labelspacing=1.25,
        )
        if leg is not None:
            leg.get_frame().set_alpha(0.95)

        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        output_plot2 = get_output_filename("lpips_vs_tokens_scatter.png", output_prefix)
        plt.savefig(output_plot2, dpi=300, bbox_inches="tight")
        plt.close()

        # --- LPIPS vs #Tokens (lowest Tokens × LPIPS) ---
        df["Composite"] = df["#Tokens"] * df["LPIPS"]
        top20_composite = df.nsmallest(20, "Composite")

        plt.figure(figsize=(14, 6))
        ax = sns.scatterplot(
            data=top20_composite,
            x="#Tokens",
            y="LPIPS",
            hue="Folder",
            palette=palette_dict,
            s=120,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_title("LPIPS vs Average #Tokens/Image", fontsize=16, weight="bold")
        ax.set_xlabel("#Tokens", fontsize=12)
        ax.set_ylabel("LPIPS", fontsize=12)

        leg = ax.legend(
            title="Tokenizer",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=8,
            markerscale=1.2,
            labelspacing=1.1,
        )
        if leg is not None:
            leg.get_frame().set_alpha(0.95)

        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        output_plot3 = get_output_filename("lpips_vs_tokens_composite_plot.png", output_prefix)
        plt.savefig(output_plot3, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate image reconstruction metrics (PSNR, SSIM, LPIPS, FID)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default behavior (backwards compatible, multi-folder mode)
  python calculate_metrics.py

  # Custom original path only
  python calculate_metrics.py --original-path /path/to/originals

  # Single comparison folder (auto-prefix enabled)
  python calculate_metrics.py --comparison-path assets/Emu3_5_IBQ

  # Custom both paths (multi-folder mode)
  python calculate_metrics.py --original-path /data/originals --comparison-path /data/reconstructions

  # Override auto-prefix
  python calculate_metrics.py --comparison-path assets/Emu3_5_IBQ --output-prefix custom_

  # Multi-folder with custom prefix
  python calculate_metrics.py --comparison-path /data/reconstructions --output-prefix experiment1_
        """,
    )

    parser.add_argument(
        "--original-path",
        type=str,
        default="assets/original",
        help="Path to folder containing original images (default: assets/original)",
    )

    parser.add_argument(
        "--comparison-path",
        type=str,
        default="assets",
        help="Path to comparison folder(s) - single folder or parent folder (default: assets)",
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output files (default: auto-generated for single folder, none for multi-folder)",
    )

    args = parser.parse_args()

    calculate_metrics(
        original_path=args.original_path,
        comparison_path=args.comparison_path,
        output_prefix=args.output_prefix,
    )