#!/usr/bin/env python3
"""
Utility to check Q-Align scoring progress and view statistics
"""

import json
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import sys
from datetime import datetime

def check_checkpoint(checkpoint_file):
    """Check checkpoint status"""
    if not checkpoint_file.exists():
        print("No checkpoint file found")
        return None
    
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
    
    print("=" * 60)
    print("Checkpoint Status")
    print("=" * 60)
    print(f"Last updated: {checkpoint.get('timestamp', 'Unknown')}")
    print(f"Shards completed: {checkpoint.get('total_processed', 0)}")
    print(f"Total shards: {checkpoint.get('total_shards', 'Unknown')}")
    
    if checkpoint.get('total_shards'):
        progress = 100 * checkpoint['total_processed'] / checkpoint['total_shards']
        print(f"Progress: {progress:.1f}%")
    
    return checkpoint

def check_statistics(stats_file):
    """Check statistics file"""
    if not stats_file.exists():
        print("\nNo statistics file found")
        return None
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    print("\n" + "=" * 60)
    print("Q-Align Score Statistics")
    print("=" * 60)
    
    metadata = stats.get('metadata', {})
    print(f"Total samples: {metadata.get('total_samples', 'Unknown'):,}")
    print(f"Unique shards: {metadata.get('unique_shards', 'Unknown')}")
    print(f"Last updated: {metadata.get('timestamp', 'Unknown')}")
    
    scores = stats.get('scores', {})
    for score_name, score_stats in scores.items():
        print(f"\n{score_name}:")
        print(f"  Mean: {score_stats['mean']:.4f} ± {score_stats['std']:.4f}")
        print(f"  Min: {score_stats['min']:.4f}")
        print(f"  Max: {score_stats['max']:.4f}")
        print(f"  Percentiles:")
        for p in [25, 50, 75, 90, 95]:
            if str(p) in score_stats.get('percentiles', {}):
                print(f"    P{p}: {score_stats['percentiles'][str(p)]:.4f}")
    
    # Show thresholds
    if 'thresholds' in stats:
        print("\n" + "=" * 60)
        print("Configured Thresholds")
        print("=" * 60)
        for score_type, thresholds in stats['thresholds'].items():
            print(f"\n{score_type}:")
            for level, value in thresholds.items():
                print(f"  {level}: {value}")
    
    return stats

def check_output_files(output_dir):
    """Check output files"""
    output_dir = Path(output_dir)
    
    print("\n" + "=" * 60)
    print("Output Files")
    print("=" * 60)
    
    # Check for parquet files
    parquet_files = list(output_dir.rglob("*.parquet"))
    print(f"Parquet files found: {len(parquet_files)}")
    
    if parquet_files:
        # Read a sample to get row count
        total_rows = 0
        for pf in parquet_files[:5]:  # Check first 5 files
            table = pq.read_table(pf)
            total_rows += len(table)
        
        if len(parquet_files) > 5:
            print(f"  (Sampled {total_rows} rows from first 5 files)")
        else:
            print(f"  Total rows: {total_rows:,}")
    
    # Check for CSV files
    csv_files = list(output_dir.glob("*.csv"))
    if csv_files:
        print(f"CSV files found: {len(csv_files)}")
        for cf in csv_files:
            size_mb = cf.stat().st_size / 1024 / 1024
            print(f"  {cf.name}: {size_mb:.1f} MB")
    
    # Check for logs
    log_dir = output_dir.parent / "logs"
    if log_dir.exists():
        log_files = list(log_dir.glob("*.log"))
        print(f"\nLog files: {len(log_files)}")
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            print(f"  Latest: {latest_log.name}")
            
            # Check if process is running
            pid_file = log_dir / "production_run.pid"
            if pid_file.exists():
                with open(pid_file, 'r') as f:
                    pid_line = f.read().strip()
                    if pid_line.startswith("PID:"):
                        pid = pid_line.split(":")[1].strip()
                        print(f"  Process PID: {pid}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Check Q-Align scoring progress')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain/metadata/scores/q_align_scores',
        help='Output directory to check'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Check test output instead'
    )
    
    args = parser.parse_args()
    
    if args.test:
        output_dir = Path('/iopsstor/scratch/cscs/rqu/benchmark-image-tokenzier/datasets/data_filtering/Q-Align/test_output_config')
    else:
        output_dir = Path(args.output_dir)
    
    print(f"Checking: {output_dir}")
    print("=" * 60)
    
    # Check checkpoint
    checkpoint_file = output_dir / 'checkpoint.json'
    check_checkpoint(checkpoint_file)
    
    # Check statistics
    stats_file = output_dir / 'q_align_statistics.json'
    check_statistics(stats_file)
    
    # Check output files
    check_output_files(output_dir)
    
    print("\n" + "=" * 60)
    print("Check complete")
    print("=" * 60)

if __name__ == '__main__':
    main()