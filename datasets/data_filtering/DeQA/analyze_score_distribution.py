#!/usr/bin/env python3
"""
Analyze DeQA score distribution to determine optimal filtering thresholds.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def load_scores(score_dir: Path) -> pd.DataFrame:
    """Load all score parquet files and combine them."""
    # Check for partitioned structure first (most common)
    parquet_files = list(score_dir.glob("shard_idx=*/*.parquet"))
    
    # If no partitioned files, check for flat structure
    if not parquet_files:
        parquet_files = list(score_dir.glob("*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {score_dir}")
    
    dfs = []
    for pq_file in tqdm(parquet_files, desc="Loading score files"):
        df = pd.read_parquet(pq_file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined_df)} scores from {len(parquet_files)} files")
    return combined_df


def analyze_score_intervals(df: pd.DataFrame, intervals: List[Tuple[float, float]] = None) -> Dict:
    """Analyze score distribution across different intervals."""
    
    # Get score column name
    score_col = 'deqa_score' if 'deqa_score' in df.columns else 'score'
    scores = df[score_col].values
    
    if intervals is None:
        # Default intervals for quality assessment
        intervals = [
            (0.0, 1.5),   # Bad
            (1.5, 2.0),   # Very Poor
            (2.0, 2.5),   # Poor
            (2.5, 3.0),   # Below Average
            (3.0, 3.5),   # Average
            (3.5, 4.0),   # Good
            (4.0, 4.5),   # Very Good
            (4.5, 5.0),   # Excellent
        ]
    
    results = {
        'total_samples': len(scores),
        'overall_stats': {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
        },
        'intervals': []
    }
    
    print(f"\n{'='*80}")
    print(f"Score Distribution Analysis")
    print(f"{'='*80}")
    print(f"Total samples: {len(scores):,}")
    print(f"Overall mean: {results['overall_stats']['mean']:.3f} ± {results['overall_stats']['std']:.3f}")
    print(f"Range: [{results['overall_stats']['min']:.3f}, {results['overall_stats']['max']:.3f}]")
    print(f"Median: {results['overall_stats']['median']:.3f}")
    
    print(f"\n{'='*80}")
    print(f"{'Interval':<20} {'Count':>10} {'Percent':>10} {'Cumulative':>12} {'Mean':>10} {'Std':>10}")
    print(f"{'-'*80}")
    
    cumulative_count = 0
    cumulative_percent = 0.0
    
    for lower, upper in intervals:
        mask = (scores >= lower) & (scores < upper)
        count = np.sum(mask)
        percent = count / len(scores) * 100
        cumulative_count += count
        cumulative_percent += percent
        
        interval_scores = scores[mask]
        interval_mean = float(np.mean(interval_scores)) if len(interval_scores) > 0 else 0
        interval_std = float(np.std(interval_scores)) if len(interval_scores) > 0 else 0
        
        interval_data = {
            'range': f"[{lower:.1f}, {upper:.1f})",
            'lower': lower,
            'upper': upper,
            'count': int(count),
            'percent': float(percent),
            'cumulative_count': int(cumulative_count),
            'cumulative_percent': float(cumulative_percent),
            'mean': interval_mean,
            'std': interval_std
        }
        
        results['intervals'].append(interval_data)
        
        print(f"{interval_data['range']:<20} {count:>10,} {percent:>9.2f}% {cumulative_percent:>11.2f}% {interval_mean:>10.3f} {interval_std:>10.3f}")
    
    # Add percentile-based thresholds
    percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
    results['percentiles'] = {}
    
    print(f"\n{'='*80}")
    print(f"Percentile-based Thresholds")
    print(f"{'-'*80}")
    print(f"{'Percentile':<15} {'Score':>10} {'Keep Above':>15} {'Remove Below':>15}")
    print(f"{'-'*80}")
    
    for p in percentiles:
        threshold = float(np.percentile(scores, p))
        results['percentiles'][str(p)] = threshold
        keep_percent = 100 - p
        print(f"{p:>3}th percentile {threshold:>10.3f} {f'{keep_percent}%':>15} {f'{p}%':>15}")
    
    # Suggest filtering thresholds
    print(f"\n{'='*80}")
    print(f"Suggested Filtering Thresholds")
    print(f"{'-'*80}")
    
    thresholds = [2.0, 2.5, 3.0, 3.5, 4.0]
    results['threshold_analysis'] = []
    
    for threshold in thresholds:
        above = np.sum(scores >= threshold)
        below = np.sum(scores < threshold)
        percent_kept = above / len(scores) * 100
        percent_removed = below / len(scores) * 100
        
        above_mean = float(np.mean(scores[scores >= threshold])) if above > 0 else 0
        below_mean = float(np.mean(scores[scores < threshold])) if below > 0 else 0
        
        threshold_data = {
            'threshold': threshold,
            'kept_count': int(above),
            'kept_percent': float(percent_kept),
            'removed_count': int(below),
            'removed_percent': float(percent_removed),
            'kept_mean': above_mean,
            'removed_mean': below_mean
        }
        
        results['threshold_analysis'].append(threshold_data)
        
        print(f"\nThreshold >= {threshold:.1f}:")
        print(f"  Keep: {above:,} samples ({percent_kept:.1f}%) with mean score {above_mean:.3f}")
        print(f"  Remove: {below:,} samples ({percent_removed:.1f}%) with mean score {below_mean:.3f}")
    
    return results


def create_visualization(df: pd.DataFrame, output_path: Path = None):
    """Create visualization of score distribution."""
    if not PLOTTING_AVAILABLE:
        print("\nWarning: matplotlib not available for visualization")
        return
    
    try:
        # Get score column name
        score_col = 'deqa_score' if 'deqa_score' in df.columns else 'score'
        scores = df[score_col].values
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Histogram
        ax1 = axes[0, 0]
        ax1.hist(scores, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
        ax1.axvline(np.median(scores), color='green', linestyle='--', label=f'Median: {np.median(scores):.3f}')
        ax1.set_xlabel('DeQA Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Score Distribution Histogram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CDF
        ax2 = axes[0, 1]
        sorted_scores = np.sort(scores)
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
        ax2.plot(sorted_scores, cumulative, linewidth=2)
        ax2.set_xlabel('DeQA Score')
        ax2.set_ylabel('Cumulative Percentage (%)')
        ax2.set_title('Cumulative Distribution Function')
        ax2.grid(True, alpha=0.3)
        
        # Add threshold lines
        thresholds = [2.0, 2.5, 3.0, 3.5, 4.0]
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        for threshold, color in zip(thresholds, colors):
            percent_above = np.sum(scores >= threshold) / len(scores) * 100
            ax2.axvline(threshold, color=color, linestyle=':', alpha=0.5, 
                       label=f'{threshold:.1f} ({percent_above:.1f}% kept)')
        ax2.legend(loc='lower right')
        
        # Box plot with quality categories
        ax3 = axes[1, 0]
        quality_labels = []
        quality_scores = []
        
        quality_ranges = [
            (0.0, 1.5, 'Bad'),
            (1.5, 2.5, 'Poor'),
            (2.5, 3.5, 'Fair'),
            (3.5, 4.5, 'Good'),
            (4.5, 5.0, 'Excellent')
        ]
        
        for lower, upper, label in quality_ranges:
            mask = (scores >= lower) & (scores < upper)
            if np.sum(mask) > 0:
                quality_scores.append(scores[mask])
                quality_labels.append(f'{label}\n({np.sum(mask):,})')
        
        ax3.boxplot(quality_scores, labels=quality_labels)
        ax3.set_ylabel('DeQA Score')
        ax3.set_title('Score Distribution by Quality Category')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Density plot
        ax4 = axes[1, 1]
        ax4.hist(scores, bins=50, density=True, alpha=0.5, color='blue', edgecolor='black')
        
        # Add KDE
        from scipy import stats
        kde = stats.gaussian_kde(scores)
        x_range = np.linspace(scores.min(), scores.max(), 200)
        ax4.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        ax4.set_xlabel('DeQA Score')
        ax4.set_ylabel('Density')
        ax4.set_title('Probability Density Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'DeQA Score Analysis (n={len(scores):,})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {output_path}")
        else:
            plt.show()
        
    except ImportError:
        print("\nWarning: matplotlib/seaborn not available for visualization")
    except Exception as e:
        print(f"\nWarning: Could not create visualization: {e}")


def main():
    parser = argparse.ArgumentParser(description='Analyze DeQA score distribution')
    parser.add_argument('--score-dir', type=str, required=True,
                        help='Directory containing score parquet files')
    parser.add_argument('--output-json', type=str, default='score_analysis.json',
                        help='Output JSON file for analysis results')
    parser.add_argument('--output-plot', type=str, default='score_distribution.png',
                        help='Output plot file')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip creating visualization')
    
    args = parser.parse_args()
    
    score_dir = Path(args.score_dir)
    if not score_dir.exists():
        raise ValueError(f"Score directory does not exist: {score_dir}")
    
    # Load scores
    df = load_scores(score_dir)
    
    # Analyze distribution
    results = analyze_score_intervals(df)
    
    # Save results
    output_json = Path(args.output_json)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAnalysis results saved to: {output_json}")
    
    # Create visualization
    if not args.no_plot:
        output_plot = Path(args.output_plot)
        create_visualization(df, output_plot)


if __name__ == "__main__":
    main()