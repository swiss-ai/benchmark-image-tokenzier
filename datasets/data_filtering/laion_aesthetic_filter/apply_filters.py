#!/usr/bin/env python3
"""
Apply filters based on computed scores and create filtered webdataset.
"""

import os
import json
import shutil
import tarfile
import pandas as pd
import webdataset as wds
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse
from typing import Dict, List, Set
import pyarrow.parquet as pq


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_filter_scores(scores_path: Path) -> pd.DataFrame:
    """
    Load all available filter scores from parquet files.
    
    Args:
        scores_path: Path to scores directory
    
    Returns:
        Combined DataFrame with all scores
    """
    dfs = []
    
    # Load aesthetic scores if available
    aesthetic_file = scores_path / "aesthetic_scores.parquet"
    aesthetic_dir = scores_path / "aesthetic_scores"
    
    if aesthetic_file.exists():
        # Single file format
        df_aesthetic = pd.read_parquet(aesthetic_file)
        dfs.append(df_aesthetic)
        print(f"Loaded aesthetic scores: {len(df_aesthetic)} samples")
    elif aesthetic_dir.exists():
        # Partitioned or multi-file format
        df_aesthetic = pd.read_parquet(aesthetic_dir)
        dfs.append(df_aesthetic)
        print(f"Loaded aesthetic scores from partitioned dataset: {len(df_aesthetic)} samples")
    
    # Add more score loading here as they are implemented
    # e.g., resolution_scores, clip_scores, etc.
    
    if not dfs:
        raise ValueError("No score files found in scores directory")
    
    # Merge all DataFrames on sample_id
    if len(dfs) == 1:
        return dfs[0]
    else:
        result = dfs[0]
        for df in dfs[1:]:
            result = pd.merge(result, df, on=['sample_id', 'shard_idx'], how='outer')
        return result


def apply_filters(df: pd.DataFrame, config: Dict) -> Set[str]:
    """
    Apply configured filters to determine which samples to keep.
    
    Args:
        df: DataFrame with all scores
        config: Configuration dictionary
    
    Returns:
        Set of sample_ids that pass all filters
    """
    # Start with all samples
    mask = pd.Series([True] * len(df), index=df.index)
    
    filters = config.get('filters', {})
    
    # Apply aesthetic score filter
    if filters.get('aesthetic_score', {}).get('enabled', False):
        min_score = filters['aesthetic_score'].get('min_score', 0)
        max_score = filters['aesthetic_score'].get('max_score', 10)
        
        if 'aesthetic_score' in df.columns:
            aesthetic_mask = (df['aesthetic_score'] >= min_score) & (df['aesthetic_score'] <= max_score)
            mask = mask & aesthetic_mask
            print(f"Aesthetic filter: {aesthetic_mask.sum()} / {len(df)} samples pass")
    
    # Apply resolution filter (if implemented)
    if filters.get('resolution', {}).get('enabled', False):
        if 'min_width' in df.columns and 'min_height' in df.columns:
            min_width = filters['resolution'].get('min_width', 0)
            min_height = filters['resolution'].get('min_height', 0)
            resolution_mask = (df['min_width'] >= min_width) & (df['min_height'] >= min_height)
            mask = mask & resolution_mask
            print(f"Resolution filter: {resolution_mask.sum()} / {len(df)} samples pass")
    
    # Apply CLIP score filter (if implemented)
    if filters.get('clip_score', {}).get('enabled', False):
        if 'clip_score' in df.columns:
            min_score = filters['clip_score'].get('min_score', 0)
            clip_mask = df['clip_score'] >= min_score
            mask = mask & clip_mask
            print(f"CLIP score filter: {clip_mask.sum()} / {len(df)} samples pass")
    
    # Get passing sample IDs
    passing_samples = set(df[mask]['sample_id'].values)
    print(f"\nTotal samples passing all filters: {len(passing_samples)} / {len(df)} ({100*len(passing_samples)/len(df):.1f}%)")
    
    return passing_samples


def create_filtered_webdataset(
    input_path: Path,
    output_path: Path,
    passing_samples: Set[str],
    samples_per_tar: int = 1000
):
    """
    Create a new webdataset with only the samples that pass filters.
    
    Args:
        input_path: Path to input webdataset
        output_path: Path to output directory
        passing_samples: Set of sample IDs to include
        samples_per_tar: Number of samples per output tar file
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of input tar files
    tar_files = sorted(input_path.glob("*.tar"))
    print(f"\nProcessing {len(tar_files)} input tar files...")
    
    # Track output
    current_tar_idx = 0
    current_tar_samples = []
    current_tar_path = None
    current_tar = None
    total_written = 0
    
    def save_current_tar():
        nonlocal current_tar, current_tar_path, current_tar_samples, total_written
        if current_tar:
            current_tar.close()
            print(f"Saved {current_tar_path.name} with {len(current_tar_samples)} samples")
            total_written += len(current_tar_samples)
            current_tar_samples = []
    
    # Process each input tar
    for tar_file in tqdm(tar_files, desc="Processing tar files"):
        with tarfile.open(tar_file, 'r') as input_tar:
            members = input_tar.getmembers()
            
            # Group files by sample ID
            samples = {}
            for member in members:
                # Extract sample ID from filename (e.g., "004539375.jpg" -> "004539375")
                sample_id = member.name.split('.')[0]
                if sample_id not in samples:
                    samples[sample_id] = []
                samples[sample_id].append(member)
            
            # Process each sample
            for sample_id, sample_members in samples.items():
                if sample_id in passing_samples:
                    # Start new tar if needed
                    if len(current_tar_samples) >= samples_per_tar:
                        save_current_tar()
                        current_tar = None
                    
                    if current_tar is None:
                        current_tar_path = output_path / f"filtered_{current_tar_idx:06d}.tar"
                        current_tar = tarfile.open(current_tar_path, 'w')
                        current_tar_idx += 1
                    
                    # Copy sample files to output tar
                    for member in sample_members:
                        fileobj = input_tar.extractfile(member)
                        if fileobj:
                            current_tar.addfile(member, fileobj)
                    
                    current_tar_samples.append(sample_id)
    
    # Save final tar if needed
    save_current_tar()
    
    print(f"\nFiltering complete!")
    print(f"Total samples written: {total_written}")
    print(f"Output tar files created: {current_tar_idx}")


def generate_filter_report(df: pd.DataFrame, passing_samples: Set[str], output_path: Path):
    """
    Generate a report about the filtering results.
    
    Args:
        df: DataFrame with all scores
        passing_samples: Set of sample IDs that passed filters
        output_path: Path to save report
    """
    report = {
        "total_samples": len(df),
        "passing_samples": len(passing_samples),
        "filter_rate": f"{100*len(passing_samples)/len(df):.2f}%",
        "scores_available": list(df.columns)
    }
    
    # Add statistics for available scores
    if 'aesthetic_score' in df.columns:
        passing_df = df[df['sample_id'].isin(passing_samples)]
        report['aesthetic_stats'] = {
            'before': {
                'mean': float(df['aesthetic_score'].mean()),
                'std': float(df['aesthetic_score'].std()),
                'min': float(df['aesthetic_score'].min()),
                'max': float(df['aesthetic_score'].max())
            },
            'after': {
                'mean': float(passing_df['aesthetic_score'].mean()),
                'std': float(passing_df['aesthetic_score'].std()),
                'min': float(passing_df['aesthetic_score'].min()),
                'max': float(passing_df['aesthetic_score'].max())
            } if len(passing_df) > 0 else {}
        }
    
    # Save report
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / "filter_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nFilter report saved to: {report_file}")
    
    # Save list of filtered samples
    filtered_samples_file = output_path / "filtered_samples.txt"
    with open(filtered_samples_file, 'w') as f:
        for sample_id in sorted(passing_samples):
            f.write(f"{sample_id}\n")
    
    print(f"Filtered sample list saved to: {filtered_samples_file}")


def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Setup paths
    input_path = Path(config['dataset']['input_path'])
    output_path = Path(config['dataset']['output_path'])
    scores_path = Path(config['metadata']['scores_path'])
    
    # Load all available scores
    print("Loading filter scores...")
    df_scores = load_filter_scores(scores_path)
    
    # Apply filters
    print("\nApplying filters...")
    passing_samples = apply_filters(df_scores, config)
    
    if len(passing_samples) == 0:
        print("Warning: No samples passed the filters! Check your filter thresholds.")
        return
    
    # Generate report
    generate_filter_report(df_scores, passing_samples, output_path)
    
    # Create filtered webdataset
    if not args.dry_run:
        print("\nCreating filtered webdataset...")
        create_filtered_webdataset(
            input_path,
            output_path,
            passing_samples,
            samples_per_tar=args.samples_per_tar
        )
    else:
        print("\nDry run mode - no webdataset created")
        print(f"Would create filtered dataset with {len(passing_samples)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply filters and create filtered webdataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without creating output dataset"
    )
    parser.add_argument(
        "--samples-per-tar",
        type=int,
        default=1000,
        help="Number of samples per output tar file"
    )
    
    args = parser.parse_args()
    main(args)