#!/usr/bin/env python3
"""
Batch Image Reconstruction Script

This script loads images from a folder, processes them through a tokenizer
(encode -> decode), and saves the reconstructed images to the appropriate
assets folder for metrics calculation.

Usage:
    python batch_reconstruct.py --tokenizer Emu3VisionTokenizer --input assets/original/ --output assets/Emu3VisionTokenizer/

    # With custom tokenizer path
    python batch_reconstruct.py --tokenizer Cosmos --tokenizer-path /path/to/model --input assets/original/ --output assets/Cosmos/
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from Tokenizer.base import Tokenizer
from utils_benchmark import load_all_images


# Import available tokenizers
def get_tokenizer_class(tokenizer_name: str):
    """Dynamically import and return the tokenizer class"""
    if tokenizer_name == "Emu3VisionTokenizer":
        from Tokenizer.Emu3VisionTokenizer import Emu3VisionTokenizer
        return Emu3VisionTokenizer
    elif tokenizer_name == "Emu3_5_IBQ":
        from Tokenizer.Emu3_5_IBQ import Emu3VisionTokenizerIBQ
        return Emu3VisionTokenizerIBQ
    elif tokenizer_name == "Cosmos":
        from Tokenizer.Cosmos import CosmosTokenizer
        return CosmosTokenizer
    elif tokenizer_name == "UniTok":
        from Tokenizer.UniTok import UniTok
        return UniTok
    elif tokenizer_name == "TokenFlow":
        from Tokenizer.TokenFlow_tiler import TokenFlowTokenizer
        return TokenFlowTokenizer
    elif tokenizer_name == "TokenFlow_molmo":
        from Tokenizer.TokenFlow_molmo import TokenFlowTokenizerMolmo
        return TokenFlowTokenizerMolmo
    elif tokenizer_name == "Selftok":
        from Tokenizer.Selftok import SelftokTokenizer
        return SelftokTokenizer
    elif tokenizer_name == "Selftok_molmo":
        from Tokenizer.Selftok_molmo import SelftokTokenizerMolmo
        return SelftokTokenizerMolmo
    elif tokenizer_name == "DetailFlow":
        from Tokenizer.DetailFlow import DetailFlowTokenizer
        return DetailFlowTokenizer
    elif tokenizer_name == "FlowMoTok":
        from Tokenizer.FlowMoTok import FlowMoTokTokenizer
        return FlowMoTokTokenizer
    elif tokenizer_name == "LlamaGenTok":
        from Tokenizer.LlamaGenTok import LlamaGenTokTokenizer
        return LlamaGenTokTokenizer
    elif tokenizer_name == "OpenMAGViT2":
        from Tokenizer.OpenMAGViT2 import OpenMAGViT2Tokenizer
        return OpenMAGViT2Tokenizer
    elif tokenizer_name == "FQGAN":
        from Tokenizer.fqgan import FQGANTokenizer
        return FQGANTokenizer
    elif tokenizer_name == "VILA":
        from Tokenizer.vila import VILATokenizer
        return VILATokenizer
    elif tokenizer_name == "VQGAN":
        from Tokenizer.vqgan import VQGANTokenizer
        return VQGANTokenizer
    elif tokenizer_name == "IBQ":
        from Tokenizer.IBQ import IBQTokenizer
        return IBQTokenizer
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer_name}")


def batch_reconstruct(
    input_folder: str,
    output_folder: str,
    tokenizer_name: str,
    tokenizer_path: Optional[str] = None,
    **tokenizer_kwargs
):
    """
    Batch process images through tokenizer reconstruction

    Args:
        input_folder: Path to folder containing original images
        output_folder: Path to folder where reconstructed images will be saved
        tokenizer_name: Name of the tokenizer to use
        tokenizer_path: Optional path to tokenizer model (if required by tokenizer)
        **tokenizer_kwargs: Additional keyword arguments to pass to tokenizer
    """
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load images
    print(f"Loading images from {input_folder}...")
    images, image_names, image_paths = load_all_images(input_folder)
    print(f"Loaded {len(images)} images\n")

    if len(images) == 0:
        print("No images found!")
        return

    # Initialize tokenizer
    print(f"Initializing {tokenizer_name}...")
    TokenizerClass = get_tokenizer_class(tokenizer_name)

    # Prepare tokenizer arguments
    tokenizer_args = {}
    if tokenizer_path:
        tokenizer_args["model_path"] = tokenizer_path
    tokenizer_args.update(tokenizer_kwargs)

    tokenizer = TokenizerClass(**tokenizer_args)
    print(f"Tokenizer loaded on device: {tokenizer.device}\n")

    # Process each image
    print("Processing images...")
    for idx, (image, name) in enumerate(tqdm(zip(images, image_names), total=len(images))):
        try:
            # Reconstruct image
            reconstructed_image, metrics = tokenizer.reconstruct(image)

            # Get number of tokens
            num_tokens = metrics["num_tokens"]

            # Save reconstructed image with token count in filename
            output_filename = f"{name}_{num_tokens}.png"
            output_filepath = output_path / output_filename

            reconstructed_image.save(output_filepath)

            # Print metrics for this image
            print(f"\n{name}:")
            print(f"  Tokens: {num_tokens}")
            print(f"  Compression ratio: {metrics['compression_ratio']:.2f}x")
            print(f"  Saved to: {output_filepath}")

        except Exception as e:
            print(f"\nError processing {name}: {e}")
            continue

    print(f"\n✓ Reconstruction complete!")
    print(f"  Output folder: {output_folder}")
    print(f"  You can now run: python calculate_metrics.py")


def main():
    parser = argparse.ArgumentParser(
        description="Batch reconstruct images using a tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process original images with Emu3 tokenizer
  python batch_reconstruct.py --tokenizer Emu3VisionTokenizer --input assets/original/ --output assets/Emu3VisionTokenizer/

  # Process with Cosmos tokenizer
  python batch_reconstruct.py --tokenizer Cosmos --input assets/original/ --output assets/Cosmos-custom/

  # Process with custom tokenizer path
  python batch_reconstruct.py --tokenizer Emu3_5_IBQ --tokenizer-path /path/to/model --input assets/original/ --output assets/Emu3_5_custom/

Available tokenizers:
  Emu3VisionTokenizer, Emu3_5_IBQ, Cosmos, UniTok, TokenFlow, TokenFlow_molmo,
  Selftok, Selftok_molmo, DetailFlow, FlowMoTok, LlamaGenTok, OpenMAGViT2,
  FQGAN, VILA, VQGAN, IBQ
        """
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Name of the tokenizer to use"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input folder containing original images"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output folder for reconstructed images"
    )

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Optional path to tokenizer model (if required)"
    )

    args = parser.parse_args()

    # Validate input folder exists
    if not os.path.exists(args.input):
        print(f"Error: Input folder does not exist: {args.input}")
        return

    # Run batch reconstruction
    batch_reconstruct(
        input_folder=args.input,
        output_folder=args.output,
        tokenizer_name=args.tokenizer,
        tokenizer_path=args.tokenizer_path
    )


if __name__ == "__main__":
    main()