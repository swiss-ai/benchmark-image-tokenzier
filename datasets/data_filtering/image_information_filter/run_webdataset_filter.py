#!/usr/bin/env python3
"""
Filter WebDataset based on computed image quality metrics.
Supports both threshold-based and percentile-based filtering.
"""

import argparse
import json
import yaml
from pathlib import Path
from webdataset_filter import WebDatasetFilter, FilterConfig


def main():
    parser = argparse.ArgumentParser(
        description='Filter WebDataset based on image quality metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter using percentiles (keep middle 80%)
  python run_webdataset_filter.py --mode percentile --percentile-range 10 90
  
  # Filter using thresholds
  python run_webdataset_filter.py --mode threshold --min-avg-score 0.4 --max-avg-score 0.8
  
  # Filter using specific metrics
  python run_webdataset_filter.py --mode threshold \\
    --metrics spatial_information edge_density \\
    --min-avg-score 0.3
  
  # Use configuration file
  python run_webdataset_filter.py --config filter_config.yaml
        """
    )
    
    # Paths
    parser.add_argument('--input-path', 
                       default='/capstor/store/cscs/swissai/infra01/vision-datasets/DenseFusion/DenseFusion-1M',
                       help='Path to original WebDataset tar files')
    parser.add_argument('--metrics-path',
                       default='/capstor/store/cscs/swissai/infra01/vision-datasets/DenseFusion/DenseFusion-1M/image_info_scores',
                       help='Path to computed metrics parquet files')
    parser.add_argument('--output-path', required=True,
                       help='Output path for filtered dataset')
    
    # Filter mode
    parser.add_argument('--mode', choices=['threshold', 'percentile'], default='threshold',
                       help='Filtering mode')
    
    # Percentile-based filtering
    parser.add_argument('--percentile-range', nargs=2, type=float,
                       help='Percentile range to keep (e.g., 10 90 keeps middle 80%%)')
    parser.add_argument('--percentile-min', type=float,
                       help='Minimum percentile threshold (e.g., 30 keeps top 70%%)')
    parser.add_argument('--percentile-score', default='avg_score',
                       help='Score to use for percentile filtering (default: avg_score)')
    
    # Threshold-based filtering
    parser.add_argument('--min-avg-score', type=float,
                       help='Minimum average score threshold')
    parser.add_argument('--max-avg-score', type=float,
                       help='Maximum average score threshold')
    
    # Individual metric thresholds
    parser.add_argument('--min-luminance', type=float,
                       help='Minimum luminance entropy')
    parser.add_argument('--max-luminance', type=float,
                       help='Maximum luminance entropy')
    parser.add_argument('--min-spatial', type=float,
                       help='Minimum spatial information')
    parser.add_argument('--max-spatial', type=float,
                       help='Maximum spatial information')
    parser.add_argument('--min-edge', type=float,
                       help='Minimum edge density')
    parser.add_argument('--max-edge', type=float,
                       help='Maximum edge density')
    parser.add_argument('--min-laplacian', type=float,
                       help='Minimum variance of Laplacian')
    parser.add_argument('--max-laplacian', type=float,
                       help='Maximum variance of Laplacian')
    
    # Score computation
    parser.add_argument('--metrics', nargs='+',
                       choices=['luminance_entropy', 'spatial_information', 'edge_density', 'variance_of_laplacian', 'brenner_focus'],
                       help='Metrics to use for average score computation')
    
    # Processing options
    parser.add_argument('--shards', nargs='+', type=int,
                       help='Specific shard indices to process')
    parser.add_argument('--num-workers', type=int, default=40,
                       help='Number of parallel workers')
    parser.add_argument('--compression', choices=['gz', 'bz2', 'xz'],
                       help='Output compression format')
    
    # Configuration file
    parser.add_argument('--config', help='Configuration file (JSON or YAML)')
    
    args = parser.parse_args()
    
    # Build filter configuration
    if args.config:
        # Load from config file
        with open(args.config) as f:
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        filter_config = FilterConfig(**config_dict)
    else:
        # Build from command-line arguments
        filter_config = FilterConfig(
            filter_mode=args.mode,
            compression=args.compression
        )
        
        if args.mode == 'percentile':
            if not args.percentile_range and args.percentile_min is None:
                parser.error("Either --percentile-range or --percentile-min required for percentile mode")
            if args.percentile_range:
                filter_config.percentile_range = tuple(args.percentile_range)
            if args.percentile_min is not None:
                filter_config.percentile_min = args.percentile_min
            filter_config.percentile_score_type = args.percentile_score
        else:
            # Build thresholds from arguments
            thresholds = {}
            
            # Individual metric thresholds
            if args.min_luminance is not None or args.max_luminance is not None:
                thresholds['luminance_entropy'] = {}
                if args.min_luminance is not None:
                    thresholds['luminance_entropy']['min'] = args.min_luminance
                if args.max_luminance is not None:
                    thresholds['luminance_entropy']['max'] = args.max_luminance
            
            if args.min_spatial is not None or args.max_spatial is not None:
                thresholds['spatial_information'] = {}
                if args.min_spatial is not None:
                    thresholds['spatial_information']['min'] = args.min_spatial
                if args.max_spatial is not None:
                    thresholds['spatial_information']['max'] = args.max_spatial
            
            if args.min_edge is not None or args.max_edge is not None:
                thresholds['edge_density'] = {}
                if args.min_edge is not None:
                    thresholds['edge_density']['min'] = args.min_edge
                if args.max_edge is not None:
                    thresholds['edge_density']['max'] = args.max_edge
            
            if args.min_laplacian is not None or args.max_laplacian is not None:
                thresholds['variance_of_laplacian'] = {}
                if args.min_laplacian is not None:
                    thresholds['variance_of_laplacian']['min'] = args.min_laplacian
                if args.max_laplacian is not None:
                    thresholds['variance_of_laplacian']['max'] = args.max_laplacian
            
            filter_config.thresholds = thresholds if thresholds else None
            
            # Average score threshold
            if args.min_avg_score is not None or args.max_avg_score is not None:
                filter_config.average_score_threshold = {}
                if args.min_avg_score is not None:
                    filter_config.average_score_threshold['min'] = args.min_avg_score
                if args.max_avg_score is not None:
                    filter_config.average_score_threshold['max'] = args.max_avg_score
        
        # Metrics for average score
        filter_config.average_score_metrics = args.metrics
    
    # Print configuration
    print("Filter Configuration:")
    print(f"  Mode: {filter_config.filter_mode}")
    if filter_config.filter_mode == 'percentile':
        if filter_config.percentile_range:
            print(f"  Percentile range: {filter_config.percentile_range}")
        if filter_config.percentile_min is not None:
            print(f"  Percentile minimum: {filter_config.percentile_min} (keep top {100-filter_config.percentile_min:.0f}%)")
        print(f"  Score type: {filter_config.percentile_score_type}")
    else:
        if filter_config.thresholds:
            print(f"  Metric thresholds: {filter_config.thresholds}")
        if filter_config.average_score_threshold:
            print(f"  Average score threshold: {filter_config.average_score_threshold}")
    if filter_config.average_score_metrics:
        print(f"  Metrics for average: {filter_config.average_score_metrics}")
    print()
    
    # Initialize filter
    filter_system = WebDatasetFilter(
        input_path=args.input_path,
        metrics_path=args.metrics_path,
        output_path=args.output_path,
        filter_config=filter_config,
        num_workers=args.num_workers
    )
    
    # Run filtering
    report = filter_system.filter_dataset(shard_indices=args.shards)
    
    # Print summary
    print("\nFiltering Summary:")
    summary = report['summary']
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Kept samples: {summary['kept_samples']} ({summary['keep_percentage']:.1f}%)")
    print(f"  Filtered samples: {summary['filtered_samples']}")
    print(f"  Input size: {summary['total_input_size_gb']:.2f} GB")
    print(f"  Output size: {summary['total_output_size_gb']:.2f} GB")
    print(f"  Size reduction: {(1 - summary['compression_ratio']) * 100:.1f}%")


if __name__ == "__main__":
    main()