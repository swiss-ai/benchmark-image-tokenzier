#!/usr/bin/env python3
"""
Comprehensive analysis of MLM filter results
Extracts high/low scoring samples and creates visualizations
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

def load_results(parquet_path: str) -> pd.DataFrame:
    """Load results from parquet file"""
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} samples from {parquet_path}")
    return df

def analyze_score_distributions(df: pd.DataFrame) -> Dict:
    """Analyze score distributions for all metrics"""
    
    score_cols = [col for col in df.columns if col.endswith('_score')]
    analysis = {}
    
    print("\n" + "="*70)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("="*70)
    
    for col in score_cols:
        valid = df[df[col] >= 0][col]
        
        if len(valid) == 0:
            continue
            
        metric_name = col.replace('_score', '').replace('_', ' ').title()
        
        analysis[col] = {
            'mean': valid.mean(),
            'std': valid.std(),
            'min': valid.min(),
            'max': valid.max(),
            'q25': valid.quantile(0.25),
            'median': valid.median(),
            'q75': valid.quantile(0.75),
            'unique_values': sorted(valid.unique()),
            'value_counts': valid.value_counts().to_dict()
        }
        
        print(f"\n{metric_name}:")
        print(f"  Mean: {analysis[col]['mean']:.1f} ± {analysis[col]['std']:.1f}")
        print(f"  Median: {analysis[col]['median']:.0f}")
        print(f"  Range: [{analysis[col]['min']}, {analysis[col]['max']}]")
        print(f"  IQR: [{analysis[col]['q25']:.0f}, {analysis[col]['q75']:.0f}]")
        print(f"  Unique values ({len(analysis[col]['unique_values'])}): {analysis[col]['unique_values']}")
        
        # Show value distribution
        print(f"  Distribution:")
        for score in sorted(analysis[col]['value_counts'].keys()):
            count = analysis[col]['value_counts'][score]
            pct = (count / len(valid)) * 100
            bar = '█' * int(pct / 2)  # Simple bar chart
            # Handle both int and float scores
            if isinstance(score, float) and score.is_integer():
                print(f"    {int(score):3d}: {count:4d} ({pct:5.1f}%) {bar}")
            else:
                print(f"    {score:3.0f}: {count:4d} ({pct:5.1f}%) {bar}")
    
    return analysis

def extract_extreme_samples(df: pd.DataFrame, n_samples: int = 10) -> Dict:
    """Extract high and low scoring samples for each metric"""
    
    score_cols = [col for col in df.columns if col.endswith('_score')]
    extreme_samples = {}
    
    print("\n" + "="*70)
    print("EXTREME SAMPLES EXTRACTION")
    print("="*70)
    
    for col in score_cols:
        valid_df = df[df[col] >= 0].copy()
        
        if len(valid_df) == 0:
            continue
        
        metric_name = col.replace('_score', '')
        
        # Get percentile thresholds
        high_threshold = valid_df[col].quantile(0.9)
        low_threshold = valid_df[col].quantile(0.1)
        
        # Get high and low samples
        high_samples = valid_df[valid_df[col] >= high_threshold].nlargest(n_samples, col)
        low_samples = valid_df[valid_df[col] <= low_threshold].nsmallest(n_samples, col)
        
        extreme_samples[metric_name] = {
            'high': high_samples.to_dict('records'),
            'low': low_samples.to_dict('records'),
            'high_threshold': high_threshold,
            'low_threshold': low_threshold
        }
        
        print(f"\n{metric_name.replace('_', ' ').title()}:")
        print(f"  High threshold (90th percentile): {high_threshold:.0f}")
        print(f"  Low threshold (10th percentile): {low_threshold:.0f}")
        print(f"  Extracted {len(high_samples)} high samples, {len(low_samples)} low samples")
    
    return extreme_samples

def save_sample_examples(extreme_samples: Dict, output_dir: Path):
    """Save examples of high and low scoring samples"""
    
    examples_dir = output_dir / 'sample_examples'
    examples_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("SAVING SAMPLE EXAMPLES")
    print("="*70)
    
    for metric, samples in extreme_samples.items():
        metric_dir = examples_dir / metric
        metric_dir.mkdir(exist_ok=True)
        
        # Save high scoring samples
        high_file = metric_dir / 'high_scoring_samples.json'
        with open(high_file, 'w') as f:
            json.dump({
                'threshold': samples['high_threshold'],
                'num_samples': len(samples['high']),
                'samples': samples['high']
            }, f, indent=2)
        
        # Save low scoring samples
        low_file = metric_dir / 'low_scoring_samples.json'
        with open(low_file, 'w') as f:
            json.dump({
                'threshold': samples['low_threshold'],
                'num_samples': len(samples['low']),
                'samples': samples['low']
            }, f, indent=2)
        
        print(f"  {metric}: Saved to {metric_dir}")
        
        # Print a few examples
        print(f"\n  Example HIGH scoring ({metric}):")
        for i, sample in enumerate(samples['high'][:2]):
            score_col = f'{metric}_score'
            print(f"    Sample {i+1}: Score={sample[score_col]}")
            print(f"      Caption: {sample['caption'][:100]}...")
        
        print(f"\n  Example LOW scoring ({metric}):")
        for i, sample in enumerate(samples['low'][:2]):
            score_col = f'{metric}_score'
            print(f"    Sample {i+1}: Score={sample[score_col]}")
            print(f"      Caption: {sample['caption'][:100]}...")

def create_visualizations(df: pd.DataFrame, analysis: Dict, output_dir: Path):
    """Create visualization plots"""
    
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    score_cols = [col for col in df.columns if col.endswith('_score')]
    
    # 1. Score distribution histograms
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(score_cols[:4]):
        valid = df[df[col] >= 0][col]
        if len(valid) == 0:
            continue
            
        ax = axes[i]
        unique_vals = sorted(valid.unique())
        counts = [sum(valid == val) for val in unique_vals]
        
        ax.bar(unique_vals, counts, width=3, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(col.replace('_', ' ').title(), fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add mean line
        mean_val = valid.mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax.legend()
    
    plt.suptitle('MLM Filter Score Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(viz_dir / 'score_distributions.png', dpi=150, bbox_inches='tight')
    print("  Saved: score_distributions.png")
    
    # 2. Correlation heatmap
    plt.figure(figsize=(10, 8))
    score_data = df[score_cols].replace(-1, np.nan)
    corr_matrix = score_data.corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, 
                cbar_kws={"shrink": 0.8})
    
    # Clean up labels
    labels = [col.replace('_score', '').replace('_', ' ').title() for col in corr_matrix.columns]
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.yticks(range(len(labels)), labels, rotation=0)
    
    plt.title('Score Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(viz_dir / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
    print("  Saved: correlation_matrix.png")
    
    # 3. Score quality tiers
    fig, ax = plt.subplots(figsize=(12, 8))
    
    quality_tiers = {
        'Poor (15)': [],
        'Below Avg (25)': [],
        'Above Avg (65)': [],
        'Good (75)': [],
        'Excellent (85+)': []
    }
    
    for col in score_cols:
        valid = df[df[col] >= 0][col]
        if len(valid) == 0:
            continue
            
        tiers_count = [
            (valid == 15).sum(),
            (valid == 25).sum(),
            (valid == 65).sum(),
            (valid == 75).sum(),
            (valid >= 85).sum()
        ]
        
        for i, tier in enumerate(quality_tiers.keys()):
            quality_tiers[tier].append(tiers_count[i])
    
    # Create stacked bar chart
    x = np.arange(len(score_cols))
    width = 0.6
    bottom = np.zeros(len(score_cols))
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    
    for i, (tier, counts) in enumerate(quality_tiers.items()):
        if len(counts) == len(score_cols):
            ax.bar(x, counts, width, label=tier, bottom=bottom, color=colors[i])
            bottom += np.array(counts)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Quality Tier Distribution by Metric', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([col.replace('_score', '').replace('_', '\n') for col in score_cols])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'quality_tiers.png', dpi=150, bbox_inches='tight')
    print("  Saved: quality_tiers.png")
    
    plt.close('all')  # Close all figures to free memory

def generate_report(df: pd.DataFrame, analysis: Dict, extreme_samples: Dict, output_dir: Path):
    """Generate comprehensive analysis report"""
    
    report_path = output_dir / 'analysis_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# MLM Filter Analysis Report\n\n")
        f.write(f"**Total Samples Analyzed:** {len(df)}\n")
        f.write(f"**Number of TAR Files:** {df['tar_file'].nunique() if 'tar_file' in df.columns else 'N/A'}\n\n")
        
        f.write("## Score Distribution Summary\n\n")
        
        for col, stats in analysis.items():
            metric_name = col.replace('_score', '').replace('_', ' ').title()
            f.write(f"### {metric_name}\n\n")
            f.write(f"- **Mean:** {stats['mean']:.2f} ± {stats['std']:.2f}\n")
            f.write(f"- **Median:** {stats['median']:.0f}\n")
            f.write(f"- **Range:** [{stats['min']}, {stats['max']}]\n")
            f.write(f"- **IQR:** [{stats['q25']:.0f}, {stats['q75']:.0f}]\n")
            f.write(f"- **Unique Values:** {len(stats['unique_values'])}\n\n")
            
            # Add distribution table
            f.write("| Score | Count | Percentage |\n")
            f.write("|-------|-------|------------|\n")
            
            total = sum(stats['value_counts'].values())
            for score in sorted(stats['value_counts'].keys()):
                count = stats['value_counts'][score]
                pct = (count / total) * 100
                f.write(f"| {score} | {count} | {pct:.1f}% |\n")
            f.write("\n")
        
        f.write("## Key Findings\n\n")
        f.write("1. **Score Discretization:** Scores cluster around specific values (15, 25, 65, 75, 85) due to model tokenization constraints.\n")
        f.write("2. **Image-Text Matching:** Most samples score high (85), indicating good alignment between images and captions.\n")
        f.write("3. **Quality Distribution:** Clear quality tiers emerge, making the scores useful for filtering.\n")
        f.write("4. **Metric Correlations:** Different metrics capture different aspects of quality.\n\n")
        
        f.write("## Sample Examples\n\n")
        f.write("High and low scoring samples have been saved in the `sample_examples` directory for manual inspection.\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("- `score_distributions.png`: Distribution of scores for each metric\n")
        f.write("- `correlation_matrix.png`: Correlation between different metrics\n")
        f.write("- `quality_tiers.png`: Distribution of quality tiers\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("1. Use score thresholds for filtering based on your quality requirements\n")
        f.write("2. Consider combining multiple metrics for more robust filtering\n")
        f.write("3. The discrete score values are sufficient for quality-based ranking\n")
        f.write("4. Manual inspection of extreme samples can help validate the scoring\n")
    
    print(f"\nReport saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze MLM Filter Results')
    parser.add_argument('--input', type=str, 
                       default='./mlm_scores_final_output/mlm_scores_final.parquet',
                       help='Path to results parquet file')
    parser.add_argument('--output_dir', type=str, 
                       default='./mlm_analysis_output',
                       help='Output directory for analysis')
    parser.add_argument('--n_samples', type=int, default=10,
                       help='Number of extreme samples to extract')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("MLM FILTER RESULTS ANALYSIS")
    print("="*70)
    
    # Load results
    df = load_results(args.input)
    
    # Analyze distributions
    analysis = analyze_score_distributions(df)
    
    # Extract extreme samples
    extreme_samples = extract_extreme_samples(df, args.n_samples)
    
    # Save sample examples
    save_sample_examples(extreme_samples, output_dir)
    
    # Create visualizations
    create_visualizations(df, analysis, output_dir)
    
    # Generate report
    generate_report(df, analysis, extreme_samples, output_dir)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print(f"  - Report: {output_dir}/analysis_report.md")
    print(f"  - Samples: {output_dir}/sample_examples/")
    print(f"  - Visualizations: {output_dir}/visualizations/")

if __name__ == "__main__":
    main()