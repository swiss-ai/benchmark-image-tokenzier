"""
Filter WebDataset while preserving original shard structure
Each sample stays in its original shard file
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
from typing import Set, Dict
import logging
from tqdm import tqdm
import webdataset as wds
import tarfile
import tempfile
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_kept_samples(filter_file: str = None, filter_parquet: str = None, epsilon: float = None) -> Set[str]:
    """Load the set of sample keys to keep"""
    
    if filter_file:
        # Load from text file
        logger.info(f"Loading kept samples from {filter_file}")
        with open(filter_file, 'r') as f:
            kept_samples = set(line.strip() for line in f)
    
    elif filter_parquet:
        # Load from parquet file
        logger.info(f"Loading filter data from {filter_parquet}")
        df = pd.read_parquet(filter_parquet)
        
        # If epsilon specified, filter by it
        if epsilon is not None:
            df = df[df['epsilon'] == epsilon]
        
        kept_samples = set(df[df['keep'] == True]['key'].values)
    
    else:
        raise ValueError("Either filter_file or filter_parquet must be specified")
    
    logger.info(f"Loaded {len(kept_samples)} samples to keep")
    return kept_samples


def filter_shard(input_shard: Path, output_shard: Path, kept_samples: Set[str], 
                 stats: Dict) -> Dict:
    """Filter a single shard file, keeping only specified samples"""
    
    shard_stats = {
        'total': 0,
        'kept': 0,
        'removed': 0,
        'removed_keys': []
    }
    
    # Create temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Open output tar file
        with tarfile.open(output_shard, 'w') as out_tar:
            # Open input tar file
            with tarfile.open(input_shard, 'r') as in_tar:
                
                # Get all members (files) in the tar
                members = in_tar.getmembers()
                
                # Group files by sample key
                sample_files = {}
                for member in members:
                    # Extract key from filename (e.g., "000000123.jpg" -> "000000123")
                    parts = member.name.split('.')
                    if len(parts) >= 2:
                        key = parts[0]
                        ext = '.'.join(parts[1:])
                        
                        if key not in sample_files:
                            sample_files[key] = []
                        sample_files[key].append((member, ext))
                
                # Process each sample
                for key, files in sample_files.items():
                    shard_stats['total'] += 1
                    
                    if key in kept_samples:
                        # Keep this sample - add all its files to output tar
                        shard_stats['kept'] += 1
                        
                        for member, ext in files:
                            # Extract to temp location
                            in_tar.extract(member, temp_path)
                            
                            # Add to output tar
                            extracted_file = temp_path / member.name
                            out_tar.add(extracted_file, arcname=member.name)
                            
                            # Clean up temp file
                            extracted_file.unlink()
                    else:
                        # Skip this sample
                        shard_stats['removed'] += 1
                        shard_stats['removed_keys'].append(key)
    
    return shard_stats


def filter_webdataset_preserved(
    input_path: str,
    output_path: str,
    kept_samples: Set[str],
    num_workers: int = 1,
    verbose: bool = False
):
    """Filter WebDataset while preserving shard structure"""
    
    input_dir = Path(input_path)
    output_dir = Path(output_path)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all tar shards
    input_shards = sorted(input_dir.glob("*.tar"))
    logger.info(f"Found {len(input_shards)} shards to process")
    
    # Statistics
    total_stats = {
        'total_samples': 0,
        'kept_samples': 0,
        'removed_samples': 0,
        'shards_processed': 0,
        'empty_shards': 0,
        'shard_details': {}
    }
    
    # Process each shard
    for shard_path in tqdm(input_shards, desc="Filtering shards"):
        shard_name = shard_path.name
        output_shard = output_dir / shard_name
        
        logger.info(f"Processing {shard_name}...")
        
        # Filter this shard
        shard_stats = filter_shard(
            shard_path, 
            output_shard, 
            kept_samples,
            total_stats
        )
        
        # Update statistics
        total_stats['total_samples'] += shard_stats['total']
        total_stats['kept_samples'] += shard_stats['kept']
        total_stats['removed_samples'] += shard_stats['removed']
        total_stats['shards_processed'] += 1
        total_stats['shard_details'][shard_name] = shard_stats
        
        # Check if shard is empty
        if shard_stats['kept'] == 0:
            total_stats['empty_shards'] += 1
            # Remove empty shard file
            output_shard.unlink()
            logger.warning(f"Shard {shard_name} is empty after filtering, removed")
        
        if verbose:
            logger.info(f"  Shard {shard_name}: {shard_stats['kept']}/{shard_stats['total']} kept")
            if shard_stats['removed'] > 0:
                logger.info(f"  Removed keys: {shard_stats['removed_keys'][:5]}...")
    
    # Save filtering statistics
    stats_file = output_dir / "filtering_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(total_stats, f, indent=2)
    
    # Print summary
    logger.info("="*60)
    logger.info("Filtering Complete!")
    logger.info(f"Total samples: {total_stats['total_samples']}")
    logger.info(f"Kept samples: {total_stats['kept_samples']} ({100*total_stats['kept_samples']/total_stats['total_samples']:.1f}%)")
    logger.info(f"Removed samples: {total_stats['removed_samples']} ({100*total_stats['removed_samples']/total_stats['total_samples']:.1f}%)")
    logger.info(f"Shards processed: {total_stats['shards_processed']}")
    logger.info(f"Empty shards removed: {total_stats['empty_shards']}")
    logger.info(f"Output directory: {output_path}")
    logger.info("="*60)
    
    return total_stats


def verify_filtered_dataset(output_path: str):
    """Verify the filtered dataset can be loaded properly"""
    
    logger.info("Verifying filtered dataset...")
    
    # Try to load the dataset
    tar_files = sorted(Path(output_path).glob("*.tar"))
    urls = [str(f) for f in tar_files]
    
    if not urls:
        logger.error("No tar files found in output directory!")
        return False
    
    dataset = wds.WebDataset(urls).decode("pil")
    
    # Count samples
    count = 0
    keys = []
    
    for sample in dataset:
        count += 1
        if '__key__' in sample:
            keys.append(sample['__key__'])
        
        # Just check first 100 samples
        if count >= 100:
            break
    
    logger.info(f"Successfully loaded {count} samples from filtered dataset")
    logger.info(f"Sample keys: {keys[:5]}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Filter WebDataset while preserving shard structure'
    )
    
    # Input/output paths
    parser.add_argument('--input-path', type=str, required=True,
                       help='Path to input WebDataset directory')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Path to output filtered WebDataset directory')
    
    # Filter specification (one of these required)
    filter_group = parser.add_mutually_exclusive_group(required=True)
    filter_group.add_argument('--filter-file', type=str,
                             help='Text file with kept sample keys')
    filter_group.add_argument('--filter-parquet', type=str,
                             help='Parquet file with filtering metadata')
    
    # Additional options
    parser.add_argument('--epsilon', type=float, default=0.05,
                       help='Epsilon value (used with parquet file)')
    parser.add_argument('--num-workers', type=int, default=1,
                       help='Number of parallel workers')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--verify', action='store_true',
                       help='Verify filtered dataset after creation')
    
    args = parser.parse_args()
    
    # Load kept samples
    kept_samples = load_kept_samples(
        filter_file=args.filter_file,
        filter_parquet=args.filter_parquet,
        epsilon=args.epsilon if args.filter_parquet else None
    )
    
    # Filter the dataset
    stats = filter_webdataset_preserved(
        input_path=args.input_path,
        output_path=args.output_path,
        kept_samples=kept_samples,
        num_workers=args.num_workers,
        verbose=args.verbose
    )
    
    # Verify if requested
    if args.verify:
        verify_filtered_dataset(args.output_path)


if __name__ == "__main__":
    main()