#!/usr/bin/env python3
"""
Create a grid visualization of sample images from different percentiles.
"""

import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json


def create_percentile_grid(analysis_dir: str, output_path: str = "percentile_grid.png"):
    """Create a grid showing samples from each percentile."""
    
    percentile_dir = Path(analysis_dir) / "percentile_samples"
    
    # Get all percentile folders sorted
    folders = sorted([f for f in percentile_dir.iterdir() if f.is_dir()])
    
    if not folders:
        print("No percentile folders found!")
        return
    
    # Create figure with subplots
    n_rows = 2
    n_cols = 5
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.3, wspace=0.2)
    
    for idx, folder in enumerate(folders):
        if idx >= n_rows * n_cols:
            break
        
        # Get first image from folder
        images = list(folder.glob("*.jpg"))
        if not images:
            continue
        
        img_path = images[0]
        img = Image.open(img_path)
        
        # Create subplot
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Display image
        ax.imshow(img)
        ax.axis('off')
        
        # Extract percentile info from folder name
        folder_name = folder.name
        parts = folder_name.split('_')
        percentile = parts[0]
        score_range = folder_name.split('score_')[1] if 'score_' in folder_name else ""
        
        ax.set_title(f"{percentile}\nScore: {score_range}", fontsize=10, fontweight='bold')
    
    plt.suptitle("Image Quality Distribution Across Percentiles\n(Sample from each decile)", 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Grid visualization saved to {output_path}")
    plt.show()


def create_metrics_comparison(analysis_dir: str, output_path: str = "metrics_comparison.png"):
    """Create a comparison chart of metrics across percentiles."""
    
    # Load analysis report
    report_path = Path(analysis_dir) / "analysis_report.json"
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = ['luminance_entropy', 'spatial_information', 'edge_density', 'variance_of_laplacian']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        if metric not in report['percentiles']:
            continue
        
        percentiles = report['percentiles'][metric]
        
        # Extract percentile values
        p_values = []
        p_labels = []
        for key in ['p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90']:
            if key in percentiles:
                p_values.append(percentiles[key])
                p_labels.append(key[1:])
        
        # Plot
        ax.plot(p_labels, p_values, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Percentile')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add mean line
        mean_val = report['metrics'][metric]['mean']
        ax.axhline(mean_val, color='red', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.3f}')
        ax.legend()
    
    plt.suptitle('Metrics Distribution Across Percentiles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Metrics comparison saved to {output_path}")
    plt.show()


def print_threshold_recommendations(analysis_dir: str):
    """Print recommended thresholds from analysis."""
    
    report_path = Path(analysis_dir) / "analysis_report.json"
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    print("\n" + "="*60)
    print("RECOMMENDED FILTERING THRESHOLDS")
    print("="*60)
    
    if 'threshold_recommendations' not in report:
        print("No threshold recommendations found in report")
        return
    
    recommendations = report['threshold_recommendations']
    
    # Create a table of recommendations
    print("\nFor keeping different proportions of the dataset:\n")
    
    levels = ['permissive', 'moderate', 'strict', 'very_strict']
    level_descriptions = {
        'permissive': 'Keep 95% (remove bottom 5%)',
        'moderate': 'Keep 85% (remove bottom 15%)',
        'strict': 'Keep 70% (remove bottom 30%)',
        'very_strict': 'Keep 50% (remove bottom 50%)'
    }
    
    for level in levels:
        print(f"\n{level.upper()} - {level_descriptions[level]}:")
        print("-" * 50)
        
        for metric in ['luminance_entropy', 'spatial_information', 'edge_density', 'variance_of_laplacian']:
            if metric in recommendations and level in recommendations[metric]:
                min_val = recommendations[metric][level]['min']
                print(f"  {metric:25s}: >= {min_val:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize analysis results")
    parser.add_argument('--analysis-dir', default='./analysis_output',
                       help='Directory containing analysis results')
    parser.add_argument('--output-dir', default='.',
                       help='Directory for output visualizations')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Create visualizations
    print("Creating percentile grid visualization...")
    create_percentile_grid(args.analysis_dir, output_dir / "percentile_grid.png")
    
    print("\nCreating metrics comparison chart...")
    create_metrics_comparison(args.analysis_dir, output_dir / "metrics_comparison.png")
    
    # Print recommendations
    print_threshold_recommendations(args.analysis_dir)