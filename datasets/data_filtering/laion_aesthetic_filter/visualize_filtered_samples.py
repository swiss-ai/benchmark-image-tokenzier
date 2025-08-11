#!/usr/bin/env python3
"""
Extract and save sample images with their captions for different filtering levels.
This helps visualize what images pass or fail various filters.
"""

import os
import json
import tarfile
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import yaml
import argparse
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_scores(scores_path: Path, filter_type: str = "aesthetic") -> pd.DataFrame:
    """
    Load filter scores from parquet files.
    
    Args:
        scores_path: Path to scores directory
        filter_type: Type of filter scores to load
    
    Returns:
        DataFrame with scores
    """
    if filter_type == "aesthetic":
        aesthetic_file = scores_path / "aesthetic_scores.parquet"
        aesthetic_dir = scores_path / "aesthetic_scores"
        
        if aesthetic_file.exists():
            return pd.read_parquet(aesthetic_file)
        elif aesthetic_dir.exists():
            return pd.read_parquet(aesthetic_dir)
        else:
            raise ValueError(f"No aesthetic scores found in {scores_path}")
    
    # Add other filter types as they are implemented
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def get_sample_sets(df: pd.DataFrame, score_column: str, num_samples: int = 30) -> Dict[str, pd.DataFrame]:
    """
    Get different sample sets based on score distribution.
    
    Args:
        df: DataFrame with scores
        score_column: Name of the score column
        num_samples: Number of samples per category
    
    Returns:
        Dictionary with sample DataFrames for each category
    """
    # Sort by score
    df_sorted = df.sort_values(score_column)
    
    # Calculate percentiles
    p25 = df[score_column].quantile(0.25)
    p50 = df[score_column].quantile(0.50)
    p75 = df[score_column].quantile(0.75)
    
    samples = {
        'bottom_10_percent': df_sorted.head(int(len(df) * 0.1)).sample(min(num_samples, int(len(df) * 0.1))),
        'low_25_percentile': df_sorted[df_sorted[score_column] <= p25].sample(min(num_samples, len(df_sorted[df_sorted[score_column] <= p25]))),
        'median_range': df_sorted[(df_sorted[score_column] > p25) & (df_sorted[score_column] <= p75)].sample(min(num_samples, len(df_sorted[(df_sorted[score_column] > p25) & (df_sorted[score_column] <= p75)]))),
        'high_75_percentile': df_sorted[df_sorted[score_column] > p75].sample(min(num_samples, len(df_sorted[df_sorted[score_column] > p75]))),
        'top_10_percent': df_sorted.tail(int(len(df) * 0.1)).sample(min(num_samples, int(len(df) * 0.1)))
    }
    
    return samples


def extract_samples_from_tar(
    tar_path: Path,
    sample_ids: List[str],
    output_dir: Path
) -> Dict[str, Tuple[Image.Image, Dict]]:
    """
    Extract specific samples from a tar file.
    
    Args:
        tar_path: Path to tar file
        sample_ids: List of sample IDs to extract
        output_dir: Directory to save extracted samples
    
    Returns:
        Dictionary mapping sample_id to (image, json_data)
    """
    samples = {}
    
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        
        for sample_id in sample_ids:
            img_member = None
            json_member = None
            
            # Find image and json files for this sample
            for member in members:
                if member.name.startswith(sample_id):
                    if member.name.endswith('.jpg') or member.name.endswith('.png'):
                        img_member = member
                    elif member.name.endswith('.json'):
                        json_member = member
            
            if img_member and json_member:
                # Extract image
                img_file = tar.extractfile(img_member)
                if img_file:
                    img = Image.open(img_file)
                    
                    # Extract JSON
                    json_file = tar.extractfile(json_member)
                    if json_file:
                        json_data = json.loads(json_file.read().decode())
                        samples[sample_id] = (img.copy(), json_data)
    
    return samples


def save_sample_images(
    samples: Dict[str, Tuple[Image.Image, Dict]],
    scores_df: pd.DataFrame,
    output_dir: Path,
    category: str,
    score_column: str
):
    """
    Save sample images with their captions and scores.
    
    Args:
        samples: Dictionary of sample_id to (image, json_data)
        scores_df: DataFrame with scores
        output_dir: Directory to save images
        category: Category name (e.g., "top_10_percent")
        score_column: Name of the score column
    """
    category_dir = output_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual images and create metadata
    metadata = []
    
    for sample_id, (img, json_data) in samples.items():
        # Get score
        score_row = scores_df[scores_df['sample_id'] == sample_id]
        if len(score_row) > 0:
            score = float(score_row[score_column].iloc[0])
        else:
            score = None
        
        # Save image
        img_path = category_dir / f"{sample_id}.jpg"
        img.save(img_path, quality=95)
        
        # Extract caption from conversations
        caption = "No caption"
        if 'conversations' in json_data:
            for conv in json_data['conversations']:
                if conv.get('from') == 'gpt':
                    caption = conv.get('value', 'No caption')
                    break
        
        # Add to metadata
        metadata.append({
            'sample_id': sample_id,
            'score': score,
            'caption': caption,
            'image_file': f"{sample_id}.jpg"
        })
    
    # Save metadata
    metadata_file = category_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved {len(samples)} samples to {category_dir}")


def create_visualization_grid(
    samples: Dict[str, Tuple[Image.Image, Dict]],
    scores_df: pd.DataFrame,
    output_path: Path,
    category: str,
    score_column: str,
    grid_size: Tuple[int, int] = (5, 6)
):
    """
    Create a grid visualization of samples.
    
    Args:
        samples: Dictionary of sample_id to (image, json_data)
        scores_df: DataFrame with scores
        output_path: Path to save the grid image
        category: Category name
        score_column: Name of the score column
        grid_size: (rows, cols) for the grid
    """
    rows, cols = grid_size
    fig = plt.figure(figsize=(cols * 3, rows * 3.5))
    fig.suptitle(f'{category.replace("_", " ").title()} - {score_column.replace("_", " ").title()}', 
                 fontsize=16, fontweight='bold')
    
    # Create grid
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3)
    
    sample_items = list(samples.items())[:rows * cols]
    
    for idx, (sample_id, (img, json_data)) in enumerate(sample_items):
        if idx >= rows * cols:
            break
        
        row = idx // cols
        col = idx % cols
        ax = fig.add_subplot(gs[row, col])
        
        # Display image
        ax.imshow(img)
        ax.axis('off')
        
        # Get score
        score_row = scores_df[scores_df['sample_id'] == sample_id]
        if len(score_row) > 0:
            score = float(score_row[score_column].iloc[0])
            ax.set_title(f'Score: {score:.2f}', fontsize=10, pad=5)
        
    plt.tight_layout()
    
    # Save grid
    grid_file = output_path / f"{category}_grid.png"
    plt.savefig(grid_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization grid to {grid_file}")


def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Setup paths
    input_path = Path(config['dataset']['input_path'])
    scores_path = Path(config['metadata']['scores_path'])
    base_output_dir = Path(args.output_dir)
    
    # Load scores
    print(f"Loading {args.filter_type} scores...")
    df_scores = load_scores(scores_path, args.filter_type)
    
    # Determine score column name
    score_column = f"{args.filter_type}_score"
    
    # Get sample sets
    print(f"Selecting sample sets based on {score_column} distribution...")
    sample_sets = get_sample_sets(df_scores, score_column, args.num_samples)
    
    # Create output directory for this filter type
    filter_output_dir = base_output_dir / args.filter_type
    filter_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of tar files
    tar_files = sorted(input_path.glob("*.tar"))
    
    # Process each sample set
    for category, category_df in sample_sets.items():
        print(f"\nProcessing {category}: {len(category_df)} samples")
        
        # Group samples by shard
        samples_by_shard = category_df.groupby('shard_idx')
        
        all_samples = {}
        
        for shard_idx, shard_df in samples_by_shard:
            # Find corresponding tar file
            tar_file = tar_files[shard_idx] if shard_idx < len(tar_files) else None
            
            if tar_file and tar_file.exists():
                print(f"  Extracting from shard {shard_idx}: {tar_file.name}")
                sample_ids = shard_df['sample_id'].tolist()
                
                # Extract samples
                shard_samples = extract_samples_from_tar(tar_file, sample_ids, filter_output_dir)
                all_samples.update(shard_samples)
        
        if all_samples:
            # Save individual images
            save_sample_images(all_samples, df_scores, filter_output_dir, category, score_column)
            
            # Create visualization grid
            if args.create_grid:
                create_visualization_grid(all_samples, df_scores, filter_output_dir, category, score_column)
    
    # Generate summary statistics
    stats = {
        'filter_type': args.filter_type,
        'total_samples': len(df_scores),
        'score_distribution': {
            'mean': float(df_scores[score_column].mean()),
            'std': float(df_scores[score_column].std()),
            'min': float(df_scores[score_column].min()),
            'max': float(df_scores[score_column].max()),
            'percentiles': {
                '10': float(df_scores[score_column].quantile(0.1)),
                '25': float(df_scores[score_column].quantile(0.25)),
                '50': float(df_scores[score_column].quantile(0.5)),
                '75': float(df_scores[score_column].quantile(0.75)),
                '90': float(df_scores[score_column].quantile(0.9))
            }
        },
        'categories_processed': list(sample_sets.keys())
    }
    
    # Save summary
    summary_file = filter_output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nVisualization complete! Results saved to {filter_output_dir}")
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize filtered samples")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--filter-type",
        type=str,
        default="aesthetic",
        choices=["aesthetic"],  # Add more as implemented
        help="Type of filter to visualize"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="Number of samples per category"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/iopsstor/scratch/cscs/rqu/benchmark-image-tokenzier/datasets/data_filtering/filtered_samples",
        help="Output directory for samples"
    )
    parser.add_argument(
        "--create-grid",
        action="store_true",
        help="Create visualization grids"
    )
    
    args = parser.parse_args()
    main(args)