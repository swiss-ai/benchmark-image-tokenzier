import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from lpips import LPIPS
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the assets folder
ASSETS_FOLDER = "assets"

# Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform for SSIM (preserve original size)
basic_transform = transforms.ToTensor()

lpips_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),  # scales [0,1] to [-1,1]
])

# Transform specifically for FID (resize to 299x299 for InceptionV3)
fid_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.PILToTensor(),
])

# Initialize LPIPS model
warnings.filterwarnings("ignore", category=UserWarning)
lpips_model = LPIPS(net='alex').to(device)

# Function to calculate FID
def calculate_fid(original_images, generated_images):
    fid = FrechetInceptionDistance(feature=64).to(device)
    for img in original_images:
        fid.update(fid_transform(img).unsqueeze(0).to(device), real=True)
    for img in generated_images:
        fid.update(fid_transform(img).unsqueeze(0).to(device), real=False)
    return fid.compute().item()

# Function to load images from a folder
def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            images.append(Image.open(img_path).convert("RGB"))
    return images

# Main function to calculate metrics
def calculate_metrics():
    original_folder = os.path.join(ASSETS_FOLDER, "original")
    if not os.path.exists(original_folder):
        print(f"Original folder not found: {original_folder}")
        return

    original_images = load_images_from_folder(original_folder)
    if not original_images:
        print("No original images found.")
        return

    results = []

    for folder_name in os.listdir(ASSETS_FOLDER):
        folder_path = os.path.join(ASSETS_FOLDER, folder_name)
        if os.path.isdir(folder_path) and folder_name != "original":
            print(f"\nCalculating metrics for folder: {folder_name}")
            generated_images = load_images_from_folder(folder_path)

            if len(original_images) != len(generated_images):
                print(f"  Skipping folder {folder_name}: mismatched image counts.")
                continue

            psnr_values = []
            ssim_values = []
            lpips_values = []

            # Resize generated images to match original resolution
            resized_generated_images = [
                gen.resize(orig.size, Image.LANCZOS)
                for orig, gen in zip(original_images, generated_images)
            ]

            for orig, gen_resized in zip(original_images, resized_generated_images):
                # Convert to numpy arrays for PSNR and SSIM
                orig_np = np.array(orig)
                gen_np = np.array(gen_resized)

                # PSNR
                psnr_values.append(psnr(orig_np, gen_np))
                # SSIM
                ssim_values.append(ssim(orig_np, gen_np, channel_axis=-1))
                # LPIPS
                orig_tensor = lpips_transform(orig).unsqueeze(0).to(device)
                gen_tensor = lpips_transform(gen_resized).unsqueeze(0).to(device)
                lpips_score = lpips_model(orig_tensor, gen_tensor).item()
                lpips_values.append(lpips_score)

            avg_psnr = np.mean(psnr_values)
            avg_ssim = np.mean(ssim_values)
            avg_lpips = np.mean(lpips_values)

            # FID remains unchanged, original and generated images as is
            fid_value = calculate_fid(original_images, resized_generated_images)

            results.append({
                "Folder": folder_name,
                "PSNR": avg_psnr,
                "SSIM": avg_ssim,
                "LPIPS": avg_lpips,
            })

            # Results
            print(f"  Average PSNR:  {avg_psnr:.2f}")
            print(f"  Average SSIM:  {avg_ssim:.4f}")
            print(f"  Average LPIPS: {avg_lpips:.4f}")
            print(f"  FID:           {fid_value:.2f}")
            print("We currently do not have enough data to calculate the true FID or rFID. Do not use this value for any serious evaluation.")

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df_sorted = df.sort_values(by="LPIPS")
        df_sorted.to_csv("metrics_results.csv", index=False)
        print("\nSaved metrics to metrics_results.csv")

        # Save pretty Markdown
        with open("metrics_results.md", "w") as f:
            f.write(df_sorted.to_markdown(index=False))
            
        # --- Plot LPIPS Scores ---
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df_sorted, x="Folder", y="LPIPS", palette="viridis")
        plt.title("LPIPS Scores per Tokenizer (Lower is Better)")
        plt.ylabel("LPIPS")
        plt.xlabel("Tokenizer")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("lpips_comparison_plot.png", dpi=300)
        print("Saved LPIPS plot to lpips_comparison_plot.png")

if __name__ == "__main__":
    calculate_metrics()
