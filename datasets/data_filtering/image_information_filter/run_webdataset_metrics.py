#!/usr/bin/env python3
"""
Script to run image metrics computation on WebDataset format data.
"""

import argparse
import yaml
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys

import pandas as pd
import numpy as np

from webdataset_processor import WebDatasetProcessor


def setup_logging(log_path: Optional[str] = None, verbose: bool = False) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger('webdataset_metrics')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def apply_filters(df: pd.DataFrame, filter_config: Dict[str, Any]) -> pd.DataFrame:
    """Apply filtering based on metric thresholds"""
    if not filter_config.get('enabled', False):
        df['passed_filter'] = True  # If filtering disabled, all pass
        return df
    
    # Start with all samples that have computed metrics
    mask = df['metrics_computed'] == True
    
    thresholds = filter_config.get('thresholds', {})
    
    for metric, bounds in thresholds.items():
        if metric not in df.columns:
            continue
        
        if bounds is None:
            continue
            
        min_val = bounds.get('min')
        max_val = bounds.get('max')
        
        if min_val is not None:
            mask = mask & (df[metric] >= min_val)
        if max_val is not None:
            mask = mask & (df[metric] <= max_val)
    
    # Set passed_filter for all rows
    df['passed_filter'] = mask
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Compute image metrics on WebDataset format data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with config file
  python run_webdataset_metrics.py --config webdataset_config.yaml
  
  # Process specific shards
  python run_webdataset_metrics.py --config webdataset_config.yaml --shards 0 1 2
  
  # Test mode with limited samples
  python run_webdataset_metrics.py --config webdataset_config.yaml --test --max-samples 100
  
  # Custom input/output paths
  python run_webdataset_metrics.py --input /path/to/dataset --output ./metrics_output
        """
    )
    
    # Configuration
    parser.add_argument('--config', '-c', default='webdataset_config.yaml',
                       help='Configuration YAML file')
    
    # Override paths
    parser.add_argument('--input', '-i',
                       help='Override input dataset path')
    parser.add_argument('--output', '-o',
                       help='Override output directory path')
    
    # Shard selection
    parser.add_argument('--shards', nargs='+', type=int,
                       help='Specific shard indices to process')
    parser.add_argument('--num-shards', type=int,
                       help='Process first N shards')
    
    # Testing options
    parser.add_argument('--test', action='store_true',
                       help='Test mode (process limited samples)')
    parser.add_argument('--max-samples', type=int,
                       help='Maximum samples per shard (for testing)')
    
    # Processing options
    parser.add_argument('--num-workers', type=int,
                       help='Override number of parallel workers')
    parser.add_argument('--metrics', nargs='+',
                       choices=['luminance_entropy', 'spatial_information', 
                               'edge_density', 'variance_of_laplacian', 'brenner_focus'],
                       help='Override metrics to compute')
    
    # Output options
    parser.add_argument('--no-preserve-sharding', action='store_true',
                       help='Save all results in single parquet file')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only compute and save summary statistics')
    
    # Verbosity
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Override config with command-line arguments
    if args.input:
        config['dataset']['input_path'] = args.input
    if args.output:
        config['dataset']['output_path'] = args.output
    if args.num_workers:
        config['processing']['num_workers'] = args.num_workers
    if args.metrics:
        config['metrics']['compute_metrics'] = args.metrics
    if args.no_preserve_sharding:
        config['output']['preserve_sharding'] = False
    
    # Setup logging
    log_path = config['monitoring'].get('log_path') if config['monitoring'].get('save_logs') else None
    verbose = args.verbose or config['monitoring'].get('verbose', False)
    logger = setup_logging(log_path, verbose and not args.quiet)
    
    # Test mode settings
    if args.test:
        args.max_samples = args.max_samples or 100
        if not args.shards and not args.num_shards:
            args.num_shards = 2
        logger.info(f"Test mode: processing {args.max_samples} samples per shard")
    
    # Determine shards to process
    shard_indices = None
    if args.shards:
        shard_indices = args.shards
        logger.info(f"Processing specific shards: {shard_indices}")
    elif args.num_shards:
        shard_indices = list(range(args.num_shards))
        logger.info(f"Processing first {args.num_shards} shards")
    elif config['dataset'].get('shard_indices'):
        shard_indices = config['dataset']['shard_indices']
        logger.info(f"Processing shards from config: {shard_indices}")
    
    # Override max samples if specified
    max_samples_per_shard = args.max_samples or config['dataset'].get('max_samples_per_shard')
    
    # Initialize processor
    try:
        logger.info(f"Initializing WebDataset processor...")
        # Handle reference_size being None for resolution-independent processing
        ref_size = config['metrics'].get('reference_size')
        if ref_size is not None:
            ref_size = tuple(ref_size)
        
        # Check for resolution_independent flag
        resolution_independent = config['metrics'].get('resolution_independent', False)
        
        processor = WebDatasetProcessor(
            dataset_path=config['dataset']['input_path'],
            output_path=config['dataset']['output_path'],
            reference_size=ref_size,
            color_space=config['metrics'].get('color_space', 'bt709'),
            num_workers=config['processing']['num_workers'],
            batch_size=config['processing']['batch_size'],
            metrics_to_compute=config['metrics']['compute_metrics'],
            resolution_independent=resolution_independent
        )
        
        logger.info(f"Dataset path: {config['dataset']['input_path']}")
        logger.info(f"Output path: {config['dataset']['output_path']}")
        logger.info(f"Found {processor.num_shards} total shards")
        
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        sys.exit(1)
    
    # Process shards
    logger.info("Starting processing...")
    start_time = time.time()
    
    try:
        df = processor.process_shards_parallel(
            shard_indices=shard_indices,
            max_samples_per_shard=max_samples_per_shard
        )
        
        if len(df) == 0:
            logger.warning("No samples were processed successfully")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    processing_time = time.time() - start_time
    logger.info(f"Processing completed in {processing_time:.2f} seconds")
    
    # Apply filters if configured
    if config['filtering'].get('enabled'):
        logger.info("Applying filters...")
        df = apply_filters(df, config['filtering'])
        passed = df[df['passed_filter'] == True] if 'passed_filter' in df.columns else df
        logger.info(f"Filter results: {len(passed)}/{len(df)} passed ({len(passed)/len(df)*100:.1f}%)")
    
    # Generate summary statistics
    if not args.summary_only:
        # Save results to parquet
        logger.info("Saving results to parquet...")
        processor.save_results_to_parquet(
            df,
            preserve_sharding=config['output']['preserve_sharding']
        )
    
    # Save summary
    if config['output'].get('save_summary', True):
        logger.info("Generating summary statistics...")
        summary = processor.generate_summary_statistics(df)
        summary['processing_time_total'] = processing_time
        summary['shards_processed'] = len(shard_indices) if shard_indices else processor.num_shards
        
        # Save summary
        summary_path = Path(config['dataset']['output_path']) / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to {summary_path}")
        
        # Print summary
        if not args.quiet:
            print("\n" + "="*60)
            print("PROCESSING SUMMARY")
            print("="*60)
            print(f"Total samples: {summary['total_samples']}")
            print(f"Successfully processed: {summary['successfully_processed']}")
            print(f"Failed: {summary['failed_processing']}")
            print(f"Processing rate: {summary['processing_rate']*100:.1f}%")
            print(f"Total time: {processing_time:.2f}s")
            print(f"Avg time per sample: {summary['avg_processing_time']*1000:.1f}ms")
            
            print("\nMetric Statistics:")
            for metric in config['metrics']['compute_metrics']:
                if f'{metric}_mean' in summary:
                    print(f"\n{metric}:")
                    print(f"  Mean: {summary[f'{metric}_mean']:.4f}")
                    print(f"  Std: {summary[f'{metric}_std']:.4f}")
                    print(f"  Min: {summary[f'{metric}_min']:.4f}")
                    print(f"  Max: {summary[f'{metric}_max']:.4f}")
                    print(f"  Median: {summary[f'{metric}_median']:.4f}")
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()