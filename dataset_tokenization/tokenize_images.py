#!/usr/bin/env python3
"""
WebDataset-based Image Tokenization Pipeline
===========================================

This script tokenizes large-scale image datasets using:
1. WebDataset for efficient data loading and sharding
2. Emu3VisionTokenizer for image tokenization
3. Megatron IndexedDataset format for output

The pipeline is designed to:
- Handle datasets too large to fit in memory
- Support single-node multi-GPU processing (with future multi-node support)
- Create efficient indexed datasets for training

Usage:
    # Single GPU
    python tokenize_images.py --dataset llava --input-path /path/to/images
    
    # Multi-GPU on single node (coming soon)
    torchrun --nproc_per_node=8 tokenize_images.py --dataset llava --input-path /path/to/images
"""

import os
import sys
import argparse
import json
import glob
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

import torch
import numpy as np
import webdataset as wds
from PIL import Image
from tqdm import tqdm

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier')

# Import our modules
from utils.indexed_dataset_megatron import VisionTokenIndexedDatasetBuilder
from Tokenizer.Emu3VisionTokenizer import Emu3VisionTokenizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageDatasetTokenizer:
    """
    Main class for tokenizing image datasets.
    
    This class handles:
    1. Creating WebDataset shards from raw images
    2. Loading shards and tokenizing images
    3. Saving tokens in Megatron IndexedDataset format
    """
    
    def __init__(self, args):
        """Initialize tokenizer with command-line arguments."""
        self.args = args
        self.device = torch.device(args.device)
        
        # Initialize the vision tokenizer
        logger.info(f"Initializing {args.tokenizer} on {self.device}...")
        if args.tokenizer == "emu3":
            self.tokenizer = Emu3VisionTokenizer(device=self.device)
        else:
            raise ValueError(f"Unknown tokenizer: {args.tokenizer}")
        
        # Setup paths
        self.setup_paths()
        
    def setup_paths(self):
        """Setup input and output paths based on dataset."""
        # Output directory structure:
        # output_dir/
        #   ├── shards/          # WebDataset tar files
        #   ├── tokens/          # Tokenized IndexedDatasets
        #   └── logs/            # Processing logs
        
        self.output_base = Path(self.args.output_dir) / self.args.dataset
        self.shards_dir = self.output_base / "shards"
        self.tokens_dir = self.output_base / "tokens"
        self.logs_dir = self.output_base / "logs"
        
        # Create directories
        for dir_path in [self.shards_dir, self.tokens_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directories created under: {self.output_base}")
    
    def create_webdataset_shards(self) -> str:
        """
        Create WebDataset shards from image directory.
        
        This function:
        1. Scans the input directory for images
        2. Splits them into balanced shards
        3. Creates tar files with proper WebDataset format
        
        Returns:
            Pattern string for the created shards
        """
        logger.info(f"Scanning {self.args.input_path} for images...")
        
        # Support multiple image formats
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        image_paths = []
        
        for ext in image_extensions:
            # Handle both flat and nested directory structures
            # Pattern 1: /path/to/dataset/*.jpg (flat)
            # Pattern 2: /path/to/dataset/*/*.jpg (one level nested like LLaVA)
            # Pattern 3: /path/to/dataset/**/*.jpg (deeply nested)
            for pattern in [f"{ext}", f"*/{ext}", f"**/{ext}"]:
                full_pattern = os.path.join(self.args.input_path, pattern)
                found = glob.glob(full_pattern, recursive=True)
                image_paths.extend(found)
                if found:
                    logger.info(f"  Found {len(found)} images with pattern: {pattern}")
        
        # Remove duplicates and sort
        image_paths = sorted(list(set(image_paths)))
        logger.info(f"Total unique images found: {len(image_paths)}")
        
        if not image_paths:
            raise ValueError(f"No images found in {self.args.input_path}")
        
        # Calculate sharding
        images_per_shard = len(image_paths) // self.args.num_shards + 1
        logger.info(f"Creating {self.args.num_shards} shards with ~{images_per_shard} images each")
        
        # Create shards
        shard_pattern = str(self.shards_dir / "shard-%06d.tar")
        
        for shard_idx in range(self.args.num_shards):
            start_idx = shard_idx * images_per_shard
            end_idx = min((shard_idx + 1) * images_per_shard, len(image_paths))
            
            if start_idx >= len(image_paths):
                break
            
            shard_path = shard_pattern % shard_idx
            logger.info(f"Creating shard {shard_idx + 1}/{self.args.num_shards}: {os.path.basename(shard_path)}")
            
            # Create WebDataset tar file
            # Reference: webdataset documentation
            with wds.TarWriter(shard_path) as sink:
                for img_idx in range(start_idx, end_idx):
                    img_path = image_paths[img_idx]
                    
                    # Create unique key for each sample
                    # Format: shardXXXXXX_imageXXXXXXXX
                    key = f"{shard_idx:06d}_{img_idx:08d}"
                    
                    # Read image data
                    try:
                        with open(img_path, 'rb') as f:
                            img_data = f.read()
                    except Exception as e:
                        logger.warning(f"Failed to read {img_path}: {e}")
                        continue
                    
                    # Get image extension
                    ext = os.path.splitext(img_path)[1].lower().lstrip('.')
                    if ext == 'jpeg':
                        ext = 'jpg'  # WebDataset convention
                    
                    # Create sample with metadata
                    # WebDataset format: each sample is a dict with __key__ and data fields
                    sample = {
                        "__key__": key,
                        ext: img_data,  # Image data with proper extension
                        "json": json.dumps({
                            "original_path": img_path,
                            "index": img_idx,
                            "shard": shard_idx,
                            "relative_path": os.path.relpath(img_path, self.args.input_path)
                        }).encode()
                    }
                    
                    sink.write(sample)
        
        # Return pattern for loading
        # Format: shard-{000000..000009}.tar for 10 shards
        actual_shards = len(glob.glob(str(self.shards_dir / "shard-*.tar")))
        pattern = str(self.shards_dir / f"shard-{{000000..{actual_shards-1:06d}}}.tar")
        logger.info(f"Created {actual_shards} shards")
        logger.info(f"Shard pattern: {pattern}")
        
        return pattern
    
    def tokenize_shards(self, shard_pattern: str):
        """
        Load WebDataset shards and tokenize images.
        
        This function:
        1. Creates a WebDataset pipeline for efficient loading
        2. Tokenizes images in batches
        3. Saves tokens to IndexedDataset format
        
        Args:
            shard_pattern: Pattern for WebDataset shards
        """
        logger.info("Starting tokenization process...")
        
        # Create WebDataset pipeline
        # Reference: webdataset documentation and examples
        dataset = (
            wds.WebDataset(shard_pattern)
            .decode("pil")  # Automatically decode images to PIL format
            .to_tuple("jpg;png;jpeg;webp", "json")  # Extract image and metadata
            .batched(self.args.batch_size)  # Batch for efficient processing
        )
        
        # Initialize IndexedDataset builder with Megatron format
        # Configure vocabulary for multimodal tokenizer
        output_prefix = str(self.tokens_dir / f"{self.args.dataset}_{self.args.tokenizer}")
        
        if self.args.text_vocab_size > 0:
            # Multimodal tokenizer: text + image tokens
            total_vocab_size = self.args.text_vocab_size + self.args.image_vocab_size
            builder = VisionTokenIndexedDatasetBuilder(
                output_prefix, 
                total_vocab_size=total_vocab_size,
                text_vocab_size=self.args.text_vocab_size
            )
        else:
            # Pure vision tokenizer (Emu3 default)
            builder = VisionTokenIndexedDatasetBuilder(
                output_prefix, 
                total_vocab_size=self.args.image_vocab_size
            )
        
        # Statistics tracking
        stats = {
            'total_images': 0,
            'total_tokens': 0,
            'failed_images': 0,
            'processing_times': [],
            'token_counts': []
        }
        
        # Process batches
        with torch.no_grad():  # Disable gradients for inference
            for batch_idx, batch in enumerate(tqdm(dataset, desc="Tokenizing batches")):
                batch_start_time = time.time()
                images, metadata_list = batch
                
                # Process each image in the batch
                for img, meta_json in zip(images, metadata_list):
                    try:
                        # Parse metadata
                        meta = json.loads(meta_json)
                        
                        # Tokenize image
                        # Step 1: Preprocess - converts PIL to tensor, normalizes
                        # This uses the Emu3VisionTokenizer.preprocess method
                        img_tensor = self.tokenizer.preprocess(img)
                        
                        # Step 2: Encode - converts image tensor to discrete tokens
                        # Returns: (indices, additional_info)
                        # indices shape: typically [1, H, W] where H,W are spatial dimensions
                        indices, additional_info = self.tokenizer.encode(img_tensor)
                        
                        # Step 3: Prepare for storage
                        # Remove batch dimension and flatten spatial dimensions
                        # [1, H, W] -> [H, W] -> [H*W]
                        flat_indices = indices.squeeze(0).flatten()
                        
                        # Add to IndexedDataset
                        builder.add_image_tokens(flat_indices)
                        
                        # Update statistics
                        num_tokens = flat_indices.numel()
                        stats['total_images'] += 1
                        stats['total_tokens'] += num_tokens
                        stats['token_counts'].append(num_tokens)
                        
                        # Log progress periodically
                        if stats['total_images'] % 1000 == 0:
                            avg_tokens = stats['total_tokens'] / stats['total_images']
                            logger.info(
                                f"Processed {stats['total_images']} images, "
                                f"Total tokens: {stats['total_tokens']:,}, "
                                f"Avg tokens/image: {avg_tokens:.1f}"
                            )
                        
                    except Exception as e:
                        logger.error(f"Failed to process image {meta.get('original_path', 'unknown')}: {e}")
                        stats['failed_images'] += 1
                        continue
                
                # Track batch processing time
                batch_time = time.time() - batch_start_time
                stats['processing_times'].append(batch_time)
                
                # Clear GPU cache periodically
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Finalize dataset
        logger.info("Finalizing IndexedDataset...")
        builder.finalize()
        
        # Save detailed statistics
        stats['avg_processing_time'] = np.mean(stats['processing_times'])
        stats['avg_tokens_per_image'] = stats['total_tokens'] / max(1, stats['total_images'])
        stats['min_tokens'] = min(stats['token_counts']) if stats['token_counts'] else 0
        stats['max_tokens'] = max(stats['token_counts']) if stats['token_counts'] else 0
        
        stats_file = self.logs_dir / "tokenization_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("Tokenization Complete!")
        logger.info("=" * 80)
        logger.info(f"Total images processed: {stats['total_images']:,}")
        logger.info(f"Failed images: {stats['failed_images']:,}")
        logger.info(f"Total tokens generated: {stats['total_tokens']:,}")
        logger.info(f"Average tokens per image: {stats['avg_tokens_per_image']:.1f}")
        logger.info(f"Token range: [{stats['min_tokens']}, {stats['max_tokens']}]")
        logger.info(f"Average processing time per batch: {stats['avg_processing_time']:.2f}s")
        logger.info("-" * 80)
        logger.info(f"Output files:")
        logger.info(f"  Data: {output_prefix}.bin")
        logger.info(f"  Index: {output_prefix}.idx")
        logger.info(f"  Metadata: {output_prefix}.meta.json")
        logger.info(f"  Statistics: {stats_file}")
    
    def run(self):
        """Main execution function."""
        logger.info(f"Starting tokenization for dataset: {self.args.dataset}")
        logger.info(f"Input path: {self.args.input_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.args.batch_size}")
        
        # Step 1: Create or use existing shards
        if self.args.create_shards or not any(self.shards_dir.glob("*.tar")):
            logger.info("Creating WebDataset shards...")
            shard_pattern = self.create_webdataset_shards()
        else:
            # Use existing shards
            existing_shards = sorted(self.shards_dir.glob("shard-*.tar"))
            if not existing_shards:
                raise ValueError(f"No existing shards found in {self.shards_dir}")
            
            num_shards = len(existing_shards)
            shard_pattern = str(self.shards_dir / f"shard-{{000000..{num_shards-1:06d}}}.tar")
            logger.info(f"Using {num_shards} existing shards")
        
        # Step 2: Tokenize the shards
        self.tokenize_shards(shard_pattern)


def main():
    parser = argparse.ArgumentParser(
        description='Tokenize image datasets using WebDataset and vision tokenizers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tokenize LLaVA dataset
  python tokenize_images.py --dataset llava \\
    --input-path /capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain
  
  # Tokenize with custom settings
  python tokenize_images.py --dataset my_dataset \\
    --input-path /path/to/images \\
    --batch-size 32 \\
    --num-shards 20 \\
    --device cuda:1
        """
    )
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., llava, imagenet, custom)')
    parser.add_argument('--input-path', type=str, required=True,
                        help='Path to directory containing images')
    parser.add_argument('--output-dir', type=str, 
                        default='/iopsstor/scratch/cscs/xyixuan/tokenized_datasets',
                        help='Base output directory for all datasets')
    
    # Tokenizer arguments
    parser.add_argument('--tokenizer', type=str, default='emu3',
                        choices=['emu3'],  # Add more tokenizers here as needed
                        help='Vision tokenizer to use')
    
    # Vocabulary configuration for multimodal tokenizers
    parser.add_argument('--text-vocab-size', type=int, default=0,
                        help='Size of the text vocabulary (0 for pure vision tokenizer). '
                             'Image tokens will be offset by this amount.')
    parser.add_argument('--image-vocab-size', type=int, default=2**17,
                        help='Size of the image vocabulary (default: 131072 for Emu3)')
    parser.add_argument('--auto-detect-vocab', action='store_true',
                        help='Automatically detect vocabulary sizes from tokenizer (not implemented)')
    
    # Processing arguments
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num-shards', type=int, default=10,
                        help='Number of WebDataset shards to create')
    parser.add_argument('--create-shards', action='store_true',
                        help='Force recreation of WebDataset shards')
    
    # Future multi-node arguments (not implemented yet)
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed processing (not implemented)')
    parser.add_argument('--world-size', type=int, default=1,
                        help='Number of distributed processes')
    parser.add_argument('--rank', type=int, default=0,
                        help='Rank of current process')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_path):
        raise ValueError(f"Input path does not exist: {args.input_path}")
    
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Run tokenization
    tokenizer = ImageDatasetTokenizer(args)
    tokenizer.run()


if __name__ == "__main__":
    main()