#!/usr/bin/env python3
"""
Run comprehensive analysis on WebDataset metrics and extract sample images.
"""

import argparse
import json
from pathlib import Path
from metrics_analyzer import MetricsAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description='Analyze WebDataset metrics and extract sample images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python run_analysis.py --metrics-path ./densefusion_3shards_metrics --dataset-path /path/to/dataset
  
  # Extract more samples per percentile
  python run_analysis.py --metrics-path ./densefusion_3shards_metrics --dataset-path /path/to/dataset --samples-per-percentile 10
  
  # Custom output directory
  python run_analysis.py --metrics-path ./densefusion_3shards_metrics --dataset-path /path/to/dataset --output ./my_analysis
        """
    )
    
    parser.add_argument('--metrics-path', required=True,
                       help='Path to directory containing metrics parquet files')
    parser.add_argument('--dataset-path', required=True,
                       help='Path to WebDataset directory with tar files')
    parser.add_argument('--output', default='./analysis_output',
                       help='Output directory for analysis results')
    
    # Sample extraction options
    parser.add_argument('--n-percentiles', type=int, default=10,
                       help='Number of percentile bins (default: 10 for deciles)')
    parser.add_argument('--samples-per-percentile', type=int, default=5,
                       help='Number of samples to extract per percentile (default: 5)')
    parser.add_argument('--percentile-ranges', nargs='+', type=float,
                       help='Custom percentile boundaries (e.g., 0 10 25 50 75 90 100)')
    parser.add_argument('--skip-missing-images', action='store_true',
                       help='Continue even if images cannot be loaded')
    
    # Analysis options
    parser.add_argument('--no-images', action='store_true',
                       help='Skip image extraction, only do statistical analysis')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--use-metrics', nargs='+', 
                       choices=['luminance_entropy', 'spatial_information', 'edge_density', 'variance_of_laplacian', 'brenner_focus'],
                       help='Select which metrics to use for average score calculation')
    parser.add_argument('--show-all-metrics', action='store_true',
                       help='Show all metrics in saved images (default: only show selected metrics)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    print("Initializing metrics analyzer...")
    analyzer = MetricsAnalyzer(
        metrics_path=args.metrics_path,
        dataset_path=args.dataset_path,
        output_dir=args.output,
        use_metrics=args.use_metrics
    )
    
    # Generate analysis report
    print("\nGenerating analysis report...")
    analysis = analyzer.generate_analysis_report()
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total samples: {analysis['total_samples']}")
    
    print("\nMetric Statistics:")
    for metric, stats in analysis['metrics'].items():
        print(f"\n{metric}:")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Skewness: {stats['skewness']:.4f}")
        print(f"  Kurtosis: {stats['kurtosis']:.4f}")
    
    print("\nPercentiles for average score:")
    percentiles = analysis['percentiles'].get('avg_score', {})
    for p_key, value in percentiles.items():
        print(f"  {p_key}: {value:.4f}")
    
    print("\nRecommended Thresholds (for each metric):")
    for metric, recommendations in analysis['threshold_recommendations'].items():
        print(f"\n{metric}:")
        for level, config in recommendations.items():
            print(f"  {level}: min={config['min']:.4f} ({config['description']})")
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating distribution plots...")
        analyzer.plot_distributions(
            save_path=Path(args.output) / "distributions.png"
        )
        
        print("Generating correlation matrix...")
        analyzer.plot_correlation_matrix(
            save_path=Path(args.output) / "correlation_matrix.png"
        )
    
    # Extract percentile samples
    if not args.no_images:
        print(f"\nExtracting {args.samples_per_percentile} samples per percentile...")
        percentile_samples = analyzer.extract_percentile_samples(
            n_percentiles=args.n_percentiles,
            samples_per_percentile=args.samples_per_percentile,
            percentile_ranges=args.percentile_ranges
        )
        
        print("\nSaving sample images with metrics overlay...")
        analyzer.save_percentile_samples_with_images(
            percentile_samples,
            skip_missing=args.skip_missing_images,
            show_all_metrics=args.show_all_metrics
        )
        
        # Save percentile sample metadata
        percentile_metadata = {}
        for key, df in percentile_samples.items():
            percentile_metadata[key] = {
                'n_samples': len(df),
                'avg_score_range': [float(df['avg_score'].min()), float(df['avg_score'].max())],
                'sample_keys': df['key'].tolist()
            }
        
        metadata_path = Path(args.output) / "percentile_samples_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(percentile_metadata, f, indent=2)
        print(f"Sample metadata saved to {metadata_path}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()