#!/usr/bin/env python3
"""
Example script to prepare WebDataset shards with EMU3 tokenization.
This demonstrates the full pipeline from raw images to tokenized data.
"""

import os
import sys
import torch
import webdataset as wds
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import pickle
from typing import Dict, List, Tuple
import io

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from tokenization_emu3_image_only import EMU3ImageOnlyTokenizer
from add_special_tokens_emu3_style import add_emu3_special_tokens


def create_dummy_vision_tokenizer():
    """Create a dummy vision tokenizer for demonstration."""
    class DummyVisionTokenizer:
        def encode(self, image: Image.Image) -> Tuple[torch.Tensor, int, int]:
            """Dummy encode - in reality, this would use your vision model."""
            # Simulate vision tokenization
            # Real implementation would use actual vision model
            w, h = image.size
            
            # Downsample to token resolution (e.g., 16x16 patches)
            token_h = h // 16
            token_w = w // 16
            
            # Generate dummy indices (in practice, these come from VQ-VAE codebook)
            indices = torch.randint(0, 1000, (token_h * token_w,))
            
            return indices, token_h, token_w
    
    return DummyVisionTokenizer()


def prepare_sample(image: Image.Image, vision_tokenizer, text_tokenizer) -> Dict:
    """Prepare a single sample with EMU3 tokenization."""
    # Step 1: Get vision token indices
    indices, height, width = vision_tokenizer.encode(image)
    
    # Step 2: Tokenize with EMU3 format
    tokens = text_tokenizer.tokenize_image_only(indices, height, width)
    
    return {
        'tokens': tokens.cpu().numpy(),
        'indices': indices.cpu().numpy(),
        'height': height,
        'width': width,
        'original_size': image.size
    }


def create_webdataset_shard(
    images: List[Tuple[str, Image.Image]],
    output_path: str,
    vision_tokenizer,
    text_tokenizer,
    shard_id: int = 0
):
    """Create a WebDataset shard with tokenized data."""
    shard_name = f"{output_path}/shard-{shard_id:06d}.tar"
    
    with wds.TarWriter(shard_name) as sink:
        for idx, (name, image) in enumerate(images):
            # Prepare tokenized data
            sample_data = prepare_sample(image, vision_tokenizer, text_tokenizer)
            
            # Create sample key
            key = f"{shard_id:06d}_{idx:06d}"
            
            # Write to shard
            sample = {
                "__key__": key,
                "jpg": image,
                "tokens.pyd": pickle.dumps(sample_data['tokens']),
                "indices.pyd": pickle.dumps({
                    'indices': sample_data['indices'],
                    'height': sample_data['height'],
                    'width': sample_data['width']
                }),
                "metadata.json": {
                    "original_name": name,
                    "original_size": sample_data['original_size'],
                    "token_shape": [sample_data['height'], sample_data['width']],
                    "num_tokens": len(sample_data['tokens'])
                }
            }
            sink.write(sample)
    
    print(f"Created shard: {shard_name}")
    return shard_name


def process_dataset_distributed(
    image_dir: str,
    output_dir: str,
    tokenizer_path: str,
    images_per_shard: int = 1000,
    num_shards: int = None
):
    """Process a dataset into WebDataset shards with EMU3 tokenization."""
    # Setup tokenizers
    vision_tokenizer = create_dummy_vision_tokenizer()
    text_tokenizer = EMU3ImageOnlyTokenizer(tokenizer_path)
    
    # Get list of images
    image_files = list(Path(image_dir).glob("**/*.jpg")) + \
                  list(Path(image_dir).glob("**/*.png"))
    
    print(f"Found {len(image_files)} images")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process in shards
    shard_id = 0
    current_batch = []
    
    for img_path in image_files:
        try:
            image = Image.open(img_path).convert('RGB')
            current_batch.append((str(img_path), image))
            
            if len(current_batch) >= images_per_shard:
                create_webdataset_shard(
                    current_batch, output_dir, 
                    vision_tokenizer, text_tokenizer, shard_id
                )
                shard_id += 1
                current_batch = []
                
                if num_shards and shard_id >= num_shards:
                    break
                    
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Process remaining images
    if current_batch:
        create_webdataset_shard(
            current_batch, output_dir,
            vision_tokenizer, text_tokenizer, shard_id
        )
    
    print(f"Created {shard_id + 1} shards in {output_dir}")


def verify_webdataset_shard(shard_path: str, tokenizer_path: str):
    """Verify a WebDataset shard can be read and processed."""
    text_tokenizer = EMU3ImageOnlyTokenizer(tokenizer_path)
    
    dataset = wds.WebDataset(shard_path).decode()
    
    for sample in dataset:
        print(f"\nSample: {sample['__key__']}")
        
        # Load tokenized data
        tokens = pickle.loads(sample['tokens.pyd'])
        indices_data = pickle.loads(sample['indices.pyd'])
        
        print(f"  Token shape: {tokens.shape}")
        print(f"  Image dimensions: {indices_data['height']}x{indices_data['width']}")
        print(f"  First 10 tokens: {tokens[:10]}")
        
        # Verify we can recreate the same tokens
        indices = torch.tensor(indices_data['indices'])
        recreated = text_tokenizer.tokenize_image_only(
            indices, indices_data['height'], indices_data['width']
        )
        
        assert np.array_equal(tokens, recreated.cpu().numpy()), "Token mismatch!"
        print("  ✓ Token verification passed")
        
        break  # Just verify first sample


def main():
    """Example usage of WebDataset with EMU3 tokenization."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "verify"], default="prepare")
    parser.add_argument("--image-dir", help="Directory containing images")
    parser.add_argument("--output-dir", default="./webdataset_shards")
    parser.add_argument("--tokenizer-path", help="Path to EMU3 tokenizer")
    parser.add_argument("--images-per-shard", type=int, default=1000)
    parser.add_argument("--num-shards", type=int, help="Limit number of shards")
    parser.add_argument("--shard-path", help="Path to shard for verification")
    
    args = parser.parse_args()
    
    # If no tokenizer path, create a temporary one
    if not args.tokenizer_path:
        print("Creating temporary EMU3 tokenizer...")
        with tempfile.TemporaryDirectory() as temp_dir:
            add_emu3_special_tokens(
                model_path="meta-llama/Meta-Llama-3-8B",
                output_path=temp_dir,
                visual_vocab_size=1000,
                num_reserved_tokens=20
            )
            args.tokenizer_path = temp_dir
            
            if args.mode == "prepare":
                # Create dummy images for demonstration
                if not args.image_dir:
                    print("Creating dummy images for demonstration...")
                    import tempfile
                    img_dir = tempfile.mkdtemp()
                    for i in range(10):
                        img = Image.new('RGB', (256, 256), color=(i*25, i*25, i*25))
                        img.save(f"{img_dir}/image_{i:03d}.jpg")
                    args.image_dir = img_dir
                
                process_dataset_distributed(
                    args.image_dir,
                    args.output_dir,
                    args.tokenizer_path,
                    args.images_per_shard,
                    args.num_shards
                )
                
                # Verify first shard
                first_shard = f"{args.output_dir}/shard-000000.tar"
                if os.path.exists(first_shard):
                    print("\nVerifying first shard...")
                    verify_webdataset_shard(first_shard, args.tokenizer_path)
                    
            elif args.mode == "verify":
                if not args.shard_path:
                    print("Please provide --shard-path for verification")
                    return
                verify_webdataset_shard(args.shard_path, args.tokenizer_path)
    else:
        if args.mode == "prepare":
            process_dataset_distributed(
                args.image_dir,
                args.output_dir,
                args.tokenizer_path,
                args.images_per_shard,
                args.num_shards
            )
        elif args.mode == "verify":
            verify_webdataset_shard(args.shard_path, args.tokenizer_path)


if __name__ == "__main__":
    main()