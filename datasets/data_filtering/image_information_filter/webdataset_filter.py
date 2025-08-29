"""
WebDataset filtering system that preserves original format while applying quality filters.
Supports both threshold-based and percentile-based filtering.
"""

import os
import json
import time
import tarfile
import io
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


@dataclass
class FilterConfig:
    """Configuration for filtering WebDataset."""
    
    # Filter mode: 'threshold' or 'percentile'
    filter_mode: str = 'threshold'
    
    # Threshold-based filtering (when mode='threshold')
    thresholds: Dict[str, Dict[str, float]] = None  # e.g., {'luminance_entropy': {'min': 0.3}}
    
    # Percentile-based filtering (when mode='percentile')
    percentile_range: Tuple[float, float] = None  # e.g., (10, 90) to keep middle 80%
    percentile_min: Optional[float] = None  # e.g., 30 to keep top 70%
    
    # Score-based filtering
    use_average_score: bool = True
    average_score_metrics: Optional[List[str]] = None  # Metrics to use for average
    average_score_threshold: Optional[Dict[str, float]] = None  # {'min': 0.3, 'max': 0.8}
    
    # For percentile mode: which score to use
    percentile_score_type: str = 'avg_score'  # 'avg_score' or specific metric name
    
    # Processing options
    preserve_all_files: bool = True  # Keep json/txt even if image filtered
    output_format: str = 'tar'  # 'tar' or 'directory'
    compression: Optional[str] = None  # None, 'gz', 'bz2', 'xz'
    

class WebDatasetFilter:
    """Filter WebDataset based on computed metrics while preserving format."""
    
    def __init__(self,
                 input_path: str,
                 metrics_path: str,
                 output_path: str,
                 filter_config: FilterConfig,
                 num_workers: int = 40):
        """
        Initialize WebDataset filter.
        
        Args:
            input_path: Path to original WebDataset tar files
            metrics_path: Path to computed metrics parquet files
            output_path: Output path for filtered dataset
            filter_config: Filtering configuration
            num_workers: Number of parallel workers
        """
        self.input_path = Path(input_path)
        self.metrics_path = Path(metrics_path)
        self.output_path = Path(output_path)
        self.filter_config = filter_config
        self.num_workers = num_workers
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Get list of tar files
        self.tar_files = sorted(self.input_path.glob("*.tar"))
        self.num_shards = len(self.tar_files)
        
        print(f"Found {self.num_shards} tar shards")
        
        # Load all metrics for statistics
        self.metrics_df = self._load_all_metrics()
        
        # Compute filtering criteria
        self._prepare_filters()
        
    def _load_all_metrics(self) -> pd.DataFrame:
        """Load all metrics from parquet files."""
        parquet_files = sorted(self.metrics_path.glob("*_metrics.parquet"))
        
        if not parquet_files:
            raise ValueError(f"No metrics files found in {self.metrics_path}")
        
        dfs = []
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded metrics for {len(combined_df)} samples")
        
        return combined_df
    
    def _compute_average_scores(self):
        """Compute average normalized scores for filtering."""
        metrics = self.filter_config.average_score_metrics
        if metrics is None:
            metrics = ['luminance_entropy', 'spatial_information', 
                      'edge_density', 'variance_of_laplacian', 'brenner_focus']
        
        # Filter to only requested metrics
        metrics = [m for m in metrics if m in self.metrics_df.columns]
        print(f"Computing average score from: {metrics}")
        
        # Normalize each metric using percentile scaling
        for metric in metrics:
            if metric in self.metrics_df.columns:
                values = self.metrics_df[metric]
                p5 = values.quantile(0.05)
                p95 = values.quantile(0.95)
                self.metrics_df[f'{metric}_normalized'] = (values - p5) / (p95 - p5)
                self.metrics_df[f'{metric}_normalized'] = self.metrics_df[f'{metric}_normalized'].clip(0, 1)
        
        # Compute average
        norm_cols = [f'{m}_normalized' for m in metrics]
        self.metrics_df['avg_score'] = self.metrics_df[norm_cols].mean(axis=1)
        
    def _prepare_filters(self):
        """Prepare filtering criteria based on config."""
        
        # Compute average scores if needed
        if self.filter_config.use_average_score:
            self._compute_average_scores()
        
        if self.filter_config.filter_mode == 'percentile':
            # Compute percentile thresholds
            self._compute_percentile_thresholds()
        elif self.filter_config.filter_mode == 'threshold':
            # Use provided thresholds directly
            self.filter_thresholds = self.filter_config.thresholds or {}
            if self.filter_config.average_score_threshold:
                self.filter_thresholds['avg_score'] = self.filter_config.average_score_threshold
        else:
            raise ValueError(f"Unknown filter mode: {self.filter_config.filter_mode}")
        
        # Apply filters to metrics dataframe
        self._apply_filters_to_metrics()
        
    def _compute_percentile_thresholds(self):
        """Compute thresholds based on percentiles."""
        # Determine which percentile mode to use
        if self.filter_config.percentile_range:
            # Range mode: keep samples within percentile range
            p_low, p_high = self.filter_config.percentile_range
            use_range = True
        elif self.filter_config.percentile_min is not None:
            # Minimum mode: keep samples above percentile threshold
            p_min = self.filter_config.percentile_min
            use_range = False
        else:
            raise ValueError("Either percentile_range or percentile_min required for percentile mode")
        
        self.filter_thresholds = {}
        
        # Determine which score to use for percentile filtering
        score_col = self.filter_config.percentile_score_type
        
        # If using avg_score, ensure it's computed
        if score_col == 'avg_score' and 'avg_score' not in self.metrics_df.columns:
            self._compute_average_scores()
        
        # Apply percentile filtering to the specified score
        if score_col in self.metrics_df.columns:
            values = self.metrics_df[score_col].dropna()
            if len(values) > 0:
                if use_range:
                    threshold_min = values.quantile(p_low / 100)
                    threshold_max = values.quantile(p_high / 100)
                    self.filter_thresholds[score_col] = {
                        'min': float(threshold_min),
                        'max': float(threshold_max)
                    }
                    print(f"{score_col}: keeping range [{threshold_min:.3f}, {threshold_max:.3f}] (percentiles {p_low}-{p_high})")
                else:
                    threshold_min = values.quantile(p_min / 100)
                    self.filter_thresholds[score_col] = {
                        'min': float(threshold_min)
                    }
                    print(f"{score_col}: keeping above {threshold_min:.3f} (top {100-p_min:.0f}%)")
        else:
            # Fallback to filtering all metrics if specified score not found
            print(f"Warning: {score_col} not found, filtering all available metrics")
            for col in ['luminance_entropy', 'spatial_information', 
                       'edge_density', 'variance_of_laplacian', 'brenner_focus', 'avg_score']:
                if col in self.metrics_df.columns:
                    values = self.metrics_df[col].dropna()
                    if len(values) > 0:
                        if use_range:
                            threshold_min = values.quantile(p_low / 100)
                            threshold_max = values.quantile(p_high / 100)
                            self.filter_thresholds[col] = {
                                'min': float(threshold_min),
                                'max': float(threshold_max)
                            }
                            print(f"{col}: keeping range [{threshold_min:.3f}, {threshold_max:.3f}]")
                        else:
                            threshold_min = values.quantile(p_min / 100)
                            self.filter_thresholds[col] = {
                                'min': float(threshold_min)
                            }
                            print(f"{col}: keeping above {threshold_min:.3f}")
    
    def _apply_filters_to_metrics(self):
        """Apply filters to metrics dataframe to mark samples to keep."""
        self.metrics_df['keep'] = True
        
        for metric, thresholds in self.filter_thresholds.items():
            if metric in self.metrics_df.columns:
                if 'min' in thresholds:
                    self.metrics_df['keep'] &= (self.metrics_df[metric] >= thresholds['min'])
                if 'max' in thresholds:
                    self.metrics_df['keep'] &= (self.metrics_df[metric] <= thresholds['max'])
        
        kept = self.metrics_df['keep'].sum()
        total = len(self.metrics_df)
        print(f"Filtering: keeping {kept}/{total} samples ({kept/total*100:.1f}%)")
        
        # Create lookup dict for fast access
        self.keep_lookup = {}
        for _, row in self.metrics_df.iterrows():
            key = (int(row['shard_idx']), row['key'])
            self.keep_lookup[key] = bool(row['keep'])
    
    def filter_shard(self, shard_idx: int) -> Dict[str, Any]:
        """
        Filter a single shard and save filtered version.
        
        Args:
            shard_idx: Index of shard to process
            
        Returns:
            Statistics about filtering
        """
        tar_path = self.tar_files[shard_idx]
        output_tar_path = self.output_path / tar_path.name
        
        if self.filter_config.compression:
            output_tar_path = Path(str(output_tar_path) + f'.{self.filter_config.compression}')
        
        stats = {
            'shard_idx': shard_idx,
            'input_file': str(tar_path),
            'output_file': str(output_tar_path),
            'total_samples': 0,
            'kept_samples': 0,
            'filtered_samples': 0,
            'file_size_input': tar_path.stat().st_size,
            'file_size_output': 0
        }
        
        # Open input and output tar files
        mode = 'w'
        if self.filter_config.compression == 'gz':
            mode = 'w:gz'
        elif self.filter_config.compression == 'bz2':
            mode = 'w:bz2'
        elif self.filter_config.compression == 'xz':
            mode = 'w:xz'
        
        with tarfile.open(tar_path, 'r') as input_tar, \
             tarfile.open(output_tar_path, mode) as output_tar:
            
            # Group files by sample key
            members = input_tar.getmembers()
            samples = {}
            
            for member in members:
                # Extract key from filename (remove extension)
                key = member.name.rsplit('.', 1)[0]
                if key not in samples:
                    samples[key] = []
                samples[key].append(member)
            
            stats['total_samples'] = len(samples)
            
            # Process each sample
            for key, sample_members in samples.items():
                # Check if we should keep this sample
                lookup_key = (shard_idx, key)
                keep = self.keep_lookup.get(lookup_key, True)  # Default to keep if not found
                
                if keep:
                    stats['kept_samples'] += 1
                    # Add all files for this sample to output
                    for member in sample_members:
                        # Extract file content
                        file_obj = input_tar.extractfile(member)
                        if file_obj:
                            # Create new tarinfo
                            new_info = tarfile.TarInfo(name=member.name)
                            new_info.size = member.size
                            new_info.mtime = member.mtime
                            new_info.mode = member.mode
                            
                            # Add to output tar
                            output_tar.addfile(new_info, file_obj)
                else:
                    stats['filtered_samples'] += 1
        
        # Get output file size
        stats['file_size_output'] = output_tar_path.stat().st_size
        stats['compression_ratio'] = stats['file_size_output'] / stats['file_size_input']
        stats['keep_ratio'] = stats['kept_samples'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
        
        return stats
    
    def filter_dataset(self, shard_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Filter entire dataset or specific shards.
        
        Args:
            shard_indices: List of shard indices to process (None for all)
            
        Returns:
            Filtering report
        """
        if shard_indices is None:
            shard_indices = list(range(self.num_shards))
        
        print(f"Filtering {len(shard_indices)} shards with {self.num_workers} workers")
        
        start_time = time.time()
        all_stats = []
        
        # Process shards in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_shard = {
                executor.submit(self.filter_shard, idx): idx
                for idx in shard_indices
            }
            
            # Collect results with progress bar
            with tqdm(total=len(shard_indices), desc="Filtering shards") as pbar:
                for future in as_completed(future_to_shard):
                    shard_idx = future_to_shard[future]
                    try:
                        shard_stats = future.result()
                        all_stats.append(shard_stats)
                        pbar.update(1)
                        
                        # Update progress info
                        pbar.set_postfix({
                            'kept': f"{shard_stats['keep_ratio']*100:.1f}%",
                            'size': f"{shard_stats['compression_ratio']:.2f}x"
                        })
                    except Exception as e:
                        print(f"\nError filtering shard {shard_idx}: {e}")
                        pbar.update(1)
        
        # Compile final report
        elapsed_time = time.time() - start_time
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': asdict(self.filter_config),
            'filter_thresholds': self.filter_thresholds,
            'input_path': str(self.input_path),
            'output_path': str(self.output_path),
            'metrics_path': str(self.metrics_path),
            'num_shards_processed': len(all_stats),
            'processing_time_seconds': elapsed_time,
            'shards': all_stats,
            'summary': self._compute_summary(all_stats)
        }
        
        # Save report
        report_path = self.output_path / 'filtering_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nFiltering complete! Report saved to {report_path}")
        print(f"Summary: {report['summary']}")
        
        return report
    
    def _compute_summary(self, all_stats: List[Dict]) -> Dict:
        """Compute summary statistics from shard stats."""
        if not all_stats:
            return {}
        
        total_input_size = sum(s['file_size_input'] for s in all_stats)
        total_output_size = sum(s['file_size_output'] for s in all_stats)
        total_samples = sum(s['total_samples'] for s in all_stats)
        kept_samples = sum(s['kept_samples'] for s in all_stats)
        filtered_samples = sum(s['filtered_samples'] for s in all_stats)
        
        return {
            'total_shards': len(all_stats),
            'total_samples': total_samples,
            'kept_samples': kept_samples,
            'filtered_samples': filtered_samples,
            'keep_percentage': kept_samples / total_samples * 100 if total_samples > 0 else 0,
            'total_input_size_gb': total_input_size / 1e9,
            'total_output_size_gb': total_output_size / 1e9,
            'compression_ratio': total_output_size / total_input_size if total_input_size > 0 else 0,
            'samples_per_second': total_samples / len(all_stats) if all_stats else 0
        }


def load_filter_config(config_path: str) -> FilterConfig:
    """Load filter configuration from JSON or YAML file."""
    from pathlib import Path
    import yaml
    
    config_path = Path(config_path)
    
    if config_path.suffix == '.json':
        with open(config_path) as f:
            config_dict = json.load(f)
    elif config_path.suffix in ['.yaml', '.yml']:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
    else:
        raise ValueError(f"Unknown config format: {config_path.suffix}")
    
    return FilterConfig(**config_dict)