#!/usr/bin/env python3
"""
Extract image samples from each quality percentile based on DeQA scores.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import webdataset as wds
from PIL import Image
from tqdm import tqdm
import shutil
from collections import defaultdict


def load_scores(score_dir: Path) -> pd.DataFrame:
    """Load all score parquet files and combine them."""
    # Handle both flat and partitioned directory structures
    parquet_files = list(score_dir.glob("*.parquet"))
    parquet_files.extend(list(score_dir.glob("*/*.parquet")))  # For partitioned data
    parquet_files.extend(list(score_dir.glob("shard_idx=*/*.parquet")))  # Specific partition pattern
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {score_dir}")
    
    dfs = []
    for pq_file in tqdm(parquet_files, desc="Loading score files"):
        df = pd.read_parquet(pq_file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined_df)} scores from {len(parquet_files)} files")
    return combined_df


def calculate_percentiles(scores: np.ndarray, n_percentiles: int = 10) -> Dict[int, Tuple[float, float]]:
    """Calculate percentile boundaries."""
    percentiles = {}
    for i in range(n_percentiles):
        lower = i * (100 / n_percentiles)
        upper = (i + 1) * (100 / n_percentiles)
        
        lower_val = np.percentile(scores, lower)
        upper_val = np.percentile(scores, upper)
        
        percentiles[i] = (lower_val, upper_val)
        print(f"Percentile {i} ({lower:.0f}-{upper:.0f}%): {lower_val:.3f} - {upper_val:.3f}")
    
    return percentiles


def assign_percentile(score: float, percentiles: Dict[int, Tuple[float, float]]) -> int:
    """Assign a score to its percentile group."""
    for p_idx, (lower, upper) in percentiles.items():
        if lower <= score <= upper:
            return p_idx
    # Edge case: assign to highest percentile
    return len(percentiles) - 1


def extract_samples(
    df: pd.DataFrame,
    input_dir: Path,
    output_dir: Path,
    samples_per_percentile: int = 10,
    n_percentiles: int = 10,
    seed: int = 42
):
    """Extract sample images from each percentile."""
    np.random.seed(seed)
    
    # Calculate percentiles
    # Handle both 'score' and 'deqa_score' column names
    score_col = 'deqa_score' if 'deqa_score' in df.columns else 'score'
    scores = df[score_col].values
    percentiles = calculate_percentiles(scores, n_percentiles)
    
    # Assign each sample to a percentile
    df['percentile'] = df[score_col].apply(lambda x: assign_percentile(x, percentiles))
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample from each percentile
    samples_to_extract = defaultdict(list)
    
    for p_idx in range(n_percentiles):
        p_df = df[df['percentile'] == p_idx]
        
        if len(p_df) == 0:
            print(f"Warning: No samples in percentile {p_idx}")
            continue
        
        # Sample up to samples_per_percentile
        n_samples = min(samples_per_percentile, len(p_df))
        sampled = p_df.sample(n=n_samples, random_state=seed)
        
        samples_to_extract[p_idx] = sampled.to_dict('records')
        print(f"Percentile {p_idx}: Selected {n_samples} samples")
    
    # Extract images from webdataset
    extracted_counts = defaultdict(int)
    samples_by_shard = defaultdict(lambda: defaultdict(list))
    
    # Group samples by shard for efficient extraction
    for p_idx, samples in samples_to_extract.items():
        for sample in samples:
            shard_idx = sample['shard_idx']
            samples_by_shard[shard_idx][p_idx].append(sample)
    
    # Process each shard
    for shard_idx, percentile_samples in tqdm(samples_by_shard.items(), desc="Processing shards"):
        # Map shard_idx to the correct tar file name format
        # The format is: llava558k-XXXXXX-YYYYYYYYY.tar 
        # We need to find the specific tar file for this shard
        # Shard 0 -> llava558k-000000-000000000.tar
        # Shard 1 -> llava558k-000000-000000001.tar, etc.
        
        # Find the tar file that matches this specific shard index
        # Try different naming patterns
        tar_patterns = [
            f"llava558k-000000-{shard_idx:09d}.tar",  # LLaVA pattern
            f"coco_train2017-{shard_idx:06d}.tar",     # COCO pattern
            f"*-{shard_idx:06d}.tar",                  # Generic pattern
        ]
        
        tar_file = None
        for pattern in tar_patterns:
            potential_file = input_dir / pattern
            if potential_file.exists():
                tar_file = potential_file
                break
            else:
                # Try glob for wildcard patterns
                matches = list(input_dir.glob(pattern))
                if matches:
                    tar_file = matches[0]
                    break
        
        if not tar_file:
            print(f"Warning: No tar file found for shard {shard_idx}")
            continue
        
        print(f"Processing tar file: {tar_file.name} for shard {shard_idx}")
        
        # Create index of samples to extract
        samples_to_get = {}
        for p_idx, samples in percentile_samples.items():
            for sample in samples:
                # Handle both 'key' and 'sample_id' column names
                key = sample.get('key', sample.get('sample_id'))
                samples_to_get[key] = (p_idx, sample)
        
        # Read webdataset and extract samples
        dataset = wds.WebDataset(str(tar_file))
        
        sample_count = 0
        for sample in dataset:
            key = sample['__key__']
            sample_count += 1
            
            if key in samples_to_get:
                p_idx, sample_info = samples_to_get[key]
                
                # Create percentile directory
                p_dir = output_dir / f"percentile_{p_idx:02d}"
                p_dir.mkdir(exist_ok=True)
                
                # Save image
                if 'jpg' in sample:
                    img_data = sample['jpg']
                    img_ext = 'jpg'
                elif 'png' in sample:
                    img_data = sample['png']
                    img_ext = 'png'
                else:
                    print(f"Warning: No image found for {key}")
                    continue
                
                # Save with score in filename
                # Handle both 'score' and 'deqa_score' column names
                score = sample_info.get('deqa_score', sample_info.get('score'))
                filename = f"{key}_score_{score:.3f}.{img_ext}"
                img_path = p_dir / filename
                
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                
                extracted_counts[p_idx] += 1
                
                # Save metadata
                meta_path = p_dir / f"{key}_score_{score:.3f}.json"
                with open(meta_path, 'w') as f:
                    json.dump({
                        'key': key,
                        'score': float(score),
                        'percentile': p_idx,
                        'shard_idx': shard_idx,
                        'percentile_range': percentiles[p_idx]
                    }, f, indent=2)
        
        if sample_count > 0:
            print(f"  Checked {sample_count} samples in shard {shard_idx}, looking for: {list(samples_to_get.keys())[:3]}...")
    
    # Create summary
    summary = {
        'n_percentiles': n_percentiles,
        'samples_per_percentile_requested': samples_per_percentile,
        'percentiles': {
            f"p{k}": {
                'range': v,
                'extracted': extracted_counts[k],
                'score_range': f"{v[0]:.3f} - {v[1]:.3f}"
            }
            for k, v in percentiles.items()
        },
        'total_extracted': sum(extracted_counts.values()),
        'score_statistics': {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max())
        }
    }
    
    summary_path = output_dir / 'extraction_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExtraction complete!")
    print(f"Total samples extracted: {sum(extracted_counts.values())}")
    print(f"Summary saved to: {summary_path}")
    
    # Print extraction summary
    print("\nExtraction Summary by Percentile:")
    for p_idx in range(n_percentiles):
        count = extracted_counts[p_idx]
        lower, upper = percentiles[p_idx]
        print(f"  Percentile {p_idx:2d} (score {lower:.3f}-{upper:.3f}): {count:3d} samples")


def create_visualization_grid(output_dir: Path, n_percentiles: int = 10):
    """Create a grid visualization of samples from each percentile."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("PIL not available for visualization")
        return
    
    # Parameters
    img_size = (224, 224)
    images_per_row = 5
    margin = 10
    
    all_images = []
    
    for p_idx in range(n_percentiles):
        p_dir = output_dir / f"percentile_{p_idx:02d}"
        if not p_dir.exists():
            continue
        
        # Get up to images_per_row images
        img_files = list(p_dir.glob("*.jpg")) + list(p_dir.glob("*.png"))
        img_files = img_files[:images_per_row]
        
        row_images = []
        for img_file in img_files:
            img = Image.open(img_file)
            img = img.resize(img_size, Image.Resampling.LANCZOS)
            row_images.append(img)
        
        # Pad row if needed
        while len(row_images) < images_per_row:
            blank = Image.new('RGB', img_size, color='white')
            row_images.append(blank)
        
        all_images.append(row_images)
    
    if not all_images:
        print("No images found for visualization")
        return
    
    # Create grid
    grid_width = images_per_row * img_size[0] + (images_per_row + 1) * margin
    grid_height = len(all_images) * img_size[1] + (len(all_images) + 1) * margin
    
    grid = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(grid)
    
    # Place images
    for row_idx, row_images in enumerate(all_images):
        y = margin + row_idx * (img_size[1] + margin)
        
        # Add percentile label
        label = f"P{row_idx}"
        draw.text((5, y + img_size[1]//2), label, fill='black')
        
        for col_idx, img in enumerate(row_images):
            x = margin + col_idx * (img_size[0] + margin)
            grid.paste(img, (x, y))
    
    # Save grid
    grid_path = output_dir / 'percentile_grid.jpg'
    grid.save(grid_path, quality=95)
    print(f"Visualization grid saved to: {grid_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract image samples by quality percentile')
    parser.add_argument('--score-dir', type=str, required=True,
                        help='Directory containing score parquet files')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing webdataset tar files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for extracted samples')
    parser.add_argument('--samples-per-percentile', type=int, default=10,
                        help='Number of samples to extract per percentile (default: 10)')
    parser.add_argument('--n-percentiles', type=int, default=10,
                        help='Number of percentile groups (default: 10 for deciles)')
    parser.add_argument('--create-grid', action='store_true',
                        help='Create visualization grid of samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    
    args = parser.parse_args()
    
    score_dir = Path(args.score_dir)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not score_dir.exists():
        raise ValueError(f"Score directory does not exist: {score_dir}")
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Load scores
    df = load_scores(score_dir)
    
    # Extract samples
    extract_samples(
        df,
        input_dir,
        output_dir,
        args.samples_per_percentile,
        args.n_percentiles,
        args.seed
    )
    
    # Create visualization if requested
    if args.create_grid:
        create_visualization_grid(output_dir, args.n_percentiles)


if __name__ == "__main__":
    main()