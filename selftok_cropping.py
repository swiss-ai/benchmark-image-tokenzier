import os
import sys
print(sys.path)
sys.path.append(".")

import argparse
from mimogpt.infer.infer_utils import parse_args_from_yaml
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from mimogpt.infer.SelftokPipeline import SelftokPipeline
from mimogpt.infer.SelftokPipeline import NormalizeToTensor
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("--yml-path", type=str, default="./configs/renderer/renderer-eval.yml") # download from https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium.safetensors?download=true, require huggingface login, you have to change the format to .pt with safetensor_to_pt.py
parser.add_argument("--pretrained", type=str, default="/users/nirmiger/SelftokTokenizer/renderer_1024_ckpt.pth") 
parser.add_argument("--sd3_pretrained", type=str, default="/users/nirmiger/SelftokTokenizer/weight/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671") 
parser.add_argument("--data_size", type=int, default=256)

args = parser.parse_args()

cfg = parse_args_from_yaml(args.yml_path)
model = SelftokPipeline(cfg=cfg, ckpt_path=args.pretrained, sd3_path=args.sd3_pretrained, datasize=args.data_size, device='cuda')

img_transform = transforms.Compose([
    transforms.Resize(args.data_size),
    transforms.CenterCrop(args.data_size),
    NormalizeToTensor(),
])

# --- Step 1: Load and crop image to square ---
original_img = Image.open('./image.png').convert("RGB")
W, H = original_img.size
square_size = min(W, H)

# Crop to centered square
left = (W - square_size) // 2
top = (H - square_size) // 2
right = left + square_size
bottom = top + square_size

square_img = original_img.crop((left, top, right, bottom))  # square crop

# Save cropped version
square_img.save('./square_image.png')

# Calculate crop coordinates
W, H = square_img.size
W_half, H_half = W // 2, H // 2

# Crop 4 quadrants
quadrants = [
    square_img.crop((0, 0, W_half, H_half)),           # Top-left
    square_img.crop((W_half, 0, W, H_half)),           # Top-right
    square_img.crop((0, H_half, W_half, H)),           # Bottom-left
    square_img.crop((W_half, H_half, W, H)),           # Bottom-right
]

# Process each quadrant
processed_quadrants = []
for i, quadrant in enumerate(quadrants):
    # Apply resizing + normalization
    img = img_transform(quadrant)  # Shape: [C, H', W']
    processed_quadrants.append(img)

# Stack into batch
images = torch.stack(processed_quadrants).to('cuda')  # [4, C, H', W']

tokens = model.encoding(images, device='cuda')
np.save('./token.npy', tokens.detach().cpu().numpy())
tokens = np.load('./token.npy')
    
decoded = model.decoding_with_renderer(tokens, device='cuda')
# Resize decoded patches back to common size (if needed)
H_, W_ = decoded.shape[-2], decoded.shape[-1]
decoded = [decoded[i] for i in range(4)]

# Stitch decoded tiles back together
top = torch.cat([decoded[0], decoded[1]], dim=2)    # width
bottom = torch.cat([decoded[2], decoded[3]], dim=2)
final_image = torch.cat([top, bottom], dim=1)        # height

# Save final result
save_image(final_image, f'./reconstructed_{args.data_size}_full.png')
