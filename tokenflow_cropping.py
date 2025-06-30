import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import argparse
import numpy as np

from tokenflow.dataset.augmentation import center_crop_arr
from tokenflow.tokenizer.vq_model import VQ_models

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def crop_center_square(img: Image.Image) -> Image.Image:
    """Crop the largest possible center square from the image."""
    W, H = img.size
    square_size = min(W, H)
    left = (W - square_size) // 2
    top = (H - square_size) // 2
    return img.crop((left, top, left + square_size, top + square_size))


def split_into_quadrants(img: Image.Image) -> list:
    """Split a square image into 4 equal quadrants."""
    W, H = img.size
    half_W, half_H = W // 2, H // 2
    return [
        img.crop((0, 0, half_W, half_H)),               # Top-left
        img.crop((half_W, 0, W, half_H)),               # Top-right
        img.crop((0, half_H, half_W, H)),               # Bottom-left
        img.crop((half_W, half_H, W, H))                # Bottom-right
    ]

def process_patch(img: Image.Image, transform, vq_model, device):
    """Encode and decode one image patch."""
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        latent, _, _ = vq_model.encode(tensor)
        output = vq_model.decode(latent)
        if isinstance(output, tuple):
            output = output[1]
    return output.squeeze(0).cpu()  # [C, H, W]

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        semantic_code_dim=args.semantic_code_dim,
        teacher=args.teacher,
        enhanced_decoder=args.enhanced_decoder,
        infer_interpolate=args.infer_interpolate
    ).to(device).eval()

    # Load weights
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    state_dict = checkpoint.get("ema") or checkpoint.get("model") or checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint does not contain valid model weights.")
    vq_model.load_state_dict(state_dict)

    # Preprocess input image
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    image = Image.open(args.input_image).convert("RGB")
    square_image = crop_center_square(image)
    os.makedirs(args.output_dir, exist_ok=True)
    square_path = os.path.join(args.output_dir, "square_input.png")
    square_image.save(square_path)

    quadrants = split_into_quadrants(square_image)

    recon_tiles = [process_patch(q, transform, vq_model, device) for q in quadrants]

    top = torch.cat([recon_tiles[0], recon_tiles[1]], dim=2)     # width
    bottom = torch.cat([recon_tiles[2], recon_tiles[3]], dim=2)  # width
    stitched = torch.cat([top, bottom], dim=1)                   # height, final shape: [C, H, W]

    # --- Step 6: Final post-processing and save ---
    stitched = torch.clamp(stitched * 127.5 + 128.0, 0, 255)
    stitched = stitched.byte().permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # [H, W, C]
    
    stitched_path = os.path.join(args.output_dir, "stitched_recon.png")
    Image.fromarray(stitched).save(stitched_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size-eval", type=int, default=384, help="Eval image size (can differ from training size)")
    parser.add_argument("--input-image", type=str, required=True, help="Path to input image (PNG/JPG)")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="Path to pretrained VQ model checkpoint")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="TokenFlow")
    parser.add_argument("--teacher", type=str, default="clipb_224", choices=["clipb_224", "vitamin_xlarge_256", "siglip_384"])
    parser.add_argument("--codebook-size", type=int, default=32768)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--semantic-code-dim", type=int, default=32)
    parser.add_argument("--image-size", type=int, choices=[224, 256, 384], default=256)
    parser.add_argument("--enhanced_decoder", action="store_true")
    parser.add_argument("--infer_interpolate", action="store_true")
    parser.add_argument("--output-dir", type=str, default="recon_output")
    args = parser.parse_args()

    main(args)
