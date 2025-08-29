#!/usr/bin/env python3
"""
Filter webdataset tar files based on DeQA quality scores.
Preserves original webdataset format and metadata.
"""

import argparse
import json
import tarfile
import io
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Set, Tuple
from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np
import shutil


def setup_logging(log_dir: Path = None):
    """Setup logging configuration."""
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"filter_webdataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s'
        )
    return logging.getLogger(__name__)


def load_quality_scores(score_dir: Path, threshold: float) -> Tuple[Set[str], Dict]:
    """Load DeQA scores and filter by threshold."""
    logger = logging.getLogger(__name__)
    
    # Load all score files
    score_files = list(score_dir.glob("shard_idx=*/*.parquet"))
    if not score_files:
        score_files = list(score_dir.glob("*.parquet"))
    
    if not score_files:
        raise ValueError(f"No score files found in {score_dir}")
    
    logger.info(f"Loading scores from {len(score_files)} files...")
    
    # Combine all scores
    dfs = []
    for score_file in tqdm(score_files, desc="Loading score files"):
        df = pd.read_parquet(score_file)
        dfs.append(df)
    
    all_scores = pd.concat(dfs, ignore_index=True)
    
    # Filter by threshold
    high_quality = all_scores[all_scores['deqa_score'] >= threshold]
    
    # Create set of high quality sample IDs for fast lookup
    high_quality_ids = set(high_quality['sample_id'].astype(str))
    
    # Calculate statistics
    stats = {
        'total_samples': len(all_scores),
        'threshold': threshold,
        'kept_samples': len(high_quality),
        'kept_percent': len(high_quality) / len(all_scores) * 100,
        'removed_samples': len(all_scores) - len(high_quality),
        'removed_percent': (len(all_scores) - len(high_quality)) / len(all_scores) * 100,
        'kept_mean_score': float(high_quality['deqa_score'].mean()),
        'kept_std_score': float(high_quality['deqa_score'].std()),
        'kept_min_score': float(high_quality['deqa_score'].min()),
        'kept_max_score': float(high_quality['deqa_score'].max()),
        'removed_mean_score': float(all_scores[all_scores['deqa_score'] < threshold]['deqa_score'].mean()) if len(all_scores[all_scores['deqa_score'] < threshold]) > 0 else 0,
    }
    
    logger.info(f"Filtering with threshold >= {threshold:.2f}")
    logger.info(f"  Keeping: {stats['kept_samples']:,} samples ({stats['kept_percent']:.1f}%)")
    logger.info(f"  Removing: {stats['removed_samples']:,} samples ({stats['removed_percent']:.1f}%)")
    logger.info(f"  Kept samples mean score: {stats['kept_mean_score']:.3f}")
    
    return high_quality_ids, stats


def filter_tar_file(input_tar_path: Path, output_tar_path: Path, high_quality_ids: Set[str]) -> Dict:
    """Filter a single tar file based on quality scores."""
    logger = logging.getLogger(__name__)
    
    samples_processed = 0
    samples_kept = 0
    samples_removed = 0
    
    # Open input and output tar files
    with tarfile.open(input_tar_path, 'r') as input_tar:
        with tarfile.open(output_tar_path, 'w') as output_tar:
            
            # Get all members (files) in the tar
            members = input_tar.getmembers()
            
            # Group files by sample ID (assuming format: sample_id.extension)
            samples = {}
            for member in members:
                # Extract sample ID from filename (remove extension)
                sample_id = member.name.rsplit('.', 1)[0]
                if sample_id not in samples:
                    samples[sample_id] = []
                samples[sample_id].append(member)
            
            # Process each sample
            for sample_id, sample_files in tqdm(samples.items(), 
                                                desc=f"Processing {input_tar_path.name}",
                                                leave=False):
                samples_processed += 1
                
                # Check if this sample is high quality
                if sample_id in high_quality_ids:
                    samples_kept += 1
                    
                    # Add all files for this sample to output tar
                    for member in sample_files:
                        # Extract file content
                        file_obj = input_tar.extractfile(member)
                        if file_obj:
                            # Create new tarinfo for output
                            tarinfo = tarfile.TarInfo(name=member.name)
                            tarinfo.size = member.size
                            tarinfo.mtime = member.mtime
                            tarinfo.mode = member.mode
                            
                            # Add to output tar
                            output_tar.addfile(tarinfo, file_obj)
                else:
                    samples_removed += 1
    
    return {
        'samples_processed': samples_processed,
        'samples_kept': samples_kept,
        'samples_removed': samples_removed
    }


def filter_webdataset(input_dir: Path, output_dir: Path, score_dir: Path, 
                      threshold: float, pattern: str = "*.tar"):
    """Filter entire webdataset based on quality scores."""
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging in output directory
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Load quality scores
    logger.info("Loading quality scores...")
    high_quality_ids, score_stats = load_quality_scores(score_dir, threshold)
    
    # Find all tar files
    tar_files = sorted(input_dir.glob(pattern))
    if not tar_files:
        raise ValueError(f"No tar files found in {input_dir} with pattern {pattern}")
    
    logger.info(f"Found {len(tar_files)} tar files to process")
    
    # Process each tar file
    total_stats = {
        'total_samples_processed': 0,
        'total_samples_kept': 0,
        'total_samples_removed': 0,
        'tar_files_processed': 0,
        'tar_files_created': 0
    }
    
    for tar_file in tqdm(tar_files, desc="Filtering tar files"):
        output_tar_path = output_dir / tar_file.name
        
        # Skip if output already exists
        if output_tar_path.exists():
            logger.warning(f"Output file already exists, skipping: {output_tar_path}")
            continue
        
        # Filter the tar file
        tar_stats = filter_tar_file(tar_file, output_tar_path, high_quality_ids)
        
        # Update totals
        total_stats['total_samples_processed'] += tar_stats['samples_processed']
        total_stats['total_samples_kept'] += tar_stats['samples_kept']
        total_stats['total_samples_removed'] += tar_stats['samples_removed']
        total_stats['tar_files_processed'] += 1
        
        # Only count created tar if it has samples
        if tar_stats['samples_kept'] > 0:
            total_stats['tar_files_created'] += 1
        else:
            # Remove empty tar file
            output_tar_path.unlink()
            logger.info(f"Removed empty tar file: {output_tar_path.name}")
    
    # Save filtering statistics
    stats_output = {
        'filtering_config': {
            'input_dir': str(input_dir),
            'output_dir': str(output_dir),
            'score_dir': str(score_dir),
            'threshold': threshold,
            'pattern': pattern,
            'timestamp': datetime.now().isoformat()
        },
        'score_statistics': score_stats,
        'processing_statistics': total_stats
    }
    
    stats_file = output_dir / 'filtering_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats_output, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("FILTERING COMPLETE")
    logger.info("="*80)
    logger.info(f"Threshold: >= {threshold:.2f}")
    logger.info(f"Tar files processed: {total_stats['tar_files_processed']}")
    logger.info(f"Tar files created: {total_stats['tar_files_created']}")
    logger.info(f"Total samples processed: {total_stats['total_samples_processed']:,}")
    logger.info(f"Total samples kept: {total_stats['total_samples_kept']:,}")
    logger.info(f"Total samples removed: {total_stats['total_samples_removed']:,}")
    
    if total_stats['total_samples_processed'] > 0:
        keep_rate = total_stats['total_samples_kept'] / total_stats['total_samples_processed'] * 100
        logger.info(f"Overall keep rate: {keep_rate:.1f}%")
    
    logger.info(f"\nStatistics saved to: {stats_file}")
    logger.info(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Filter webdataset by DeQA quality scores')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing input webdataset tar files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save filtered webdataset tar files')
    parser.add_argument('--score-dir', type=str, required=True,
                        help='Directory containing DeQA score parquet files')
    parser.add_argument('--threshold', type=float, default=3.5,
                        help='Quality score threshold (keep >= threshold)')
    parser.add_argument('--pattern', type=str, default='*.tar',
                        help='Pattern to match tar files (default: *.tar)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only analyze scores without filtering')
    
    args = parser.parse_args()
    
    # Convert paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    score_dir = Path(args.score_dir)
    
    # Validate paths
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if not score_dir.exists():
        raise ValueError(f"Score directory does not exist: {score_dir}")
    
    # Setup logging
    logger = setup_logging(output_dir / "logs" if not args.dry_run else None)
    
    if args.dry_run:
        logger.info("DRY RUN MODE - Only analyzing scores")
        high_quality_ids, stats = load_quality_scores(score_dir, args.threshold)
        print(json.dumps(stats, indent=2))
    else:
        # Run filtering
        filter_webdataset(
            input_dir=input_dir,
            output_dir=output_dir,
            score_dir=score_dir,
            threshold=args.threshold,
            pattern=args.pattern
        )


if __name__ == "__main__":
    main()