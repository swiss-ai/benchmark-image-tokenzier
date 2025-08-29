#!/usr/bin/env python3
"""
Filter and sample images based on Q-Align scores
Supports percentile-based and threshold-based filtering
"""

import os
import json
import yaml
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
from tqdm import tqdm
import webdataset as wds
from io import BytesIO
from PIL import Image
import argparse

class QAlignFilter:
    def __init__(self, config_path: str):
        """Initialize filter with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scores_dir = Path(self.config['scores_dir'])
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load scores
        self.df_scores = self.load_scores()
        print(f"Loaded {len(self.df_scores)} samples with scores")
        
        # Print statistics
        self.print_statistics()
    
    def load_scores(self) -> pd.DataFrame:
        """Load all score parquet files"""
        print(f"Loading scores from {self.scores_dir}")
        
        # First check for CSV file (primary format)
        csv_file = self.scores_dir / "q_align_scores.csv"
        if csv_file.exists():
            print(f"  Loading from CSV: {csv_file}")
            df = pd.read_csv(csv_file)
            # Ensure sample_id is string for matching with WebDataset keys
            df['sample_id'] = df['sample_id'].astype(str)
            # Ensure shard_idx is integer
            if 'shard_idx' in df.columns:
                df['shard_idx'] = df['shard_idx'].astype(int)
            return df
        
        # Check for partitioned parquet as fallback
        parquet_files = list(self.scores_dir.rglob("*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"No CSV or parquet files found in {self.scores_dir}")
        
        # Load all parquet files
        dfs = []
        for pf in tqdm(parquet_files, desc="Loading parquet files"):
            df = pd.read_parquet(pf)
            # Extract shard_idx from parent directory name if not in columns
            if 'shard_idx' not in df.columns and 'shard_idx=' in str(pf.parent):
                shard_idx = int(str(pf.parent).split('shard_idx=')[1])
                df['shard_idx'] = shard_idx
            dfs.append(df)
        
        result_df = pd.concat(dfs, ignore_index=True)
        # Ensure sample_id is string and shard_idx is integer
        if 'sample_id' in result_df.columns:
            result_df['sample_id'] = result_df['sample_id'].astype(str)
        if 'shard_idx' in result_df.columns:
            result_df['shard_idx'] = result_df['shard_idx'].astype(int)
        
        return result_df
    
    def print_statistics(self):
        """Print score statistics"""
        print("\n" + "="*60)
        print("Score Statistics")
        print("="*60)
        
        score_cols = [col for col in self.df_scores.columns if 'score' in col]
        
        for col in score_cols:
            if col in self.df_scores.columns:
                scores = self.df_scores[col].values
                print(f"\n{col}:")
                print(f"  Count: {len(scores):,}")
                print(f"  Mean:  {np.mean(scores):.4f} ± {np.std(scores):.4f}")
                print(f"  Range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
                
                percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                print("  Percentiles:")
                for p in percentiles:
                    val = np.percentile(scores, p)
                    print(f"    P{p:2d}: {val:.4f}")
    
    def filter_by_percentile(
        self,
        score_column: str,
        lower_percentile: Optional[float] = None,
        upper_percentile: Optional[float] = None
    ) -> pd.DataFrame:
        """Filter samples by percentile range"""
        df = self.df_scores.copy()
        
        if score_column not in df.columns:
            raise ValueError(f"Score column '{score_column}' not found")
        
        scores = df[score_column].values
        
        if lower_percentile is not None:
            threshold_low = np.percentile(scores, lower_percentile)
            df = df[df[score_column] >= threshold_low]
            print(f"  Lower bound (P{lower_percentile}): {threshold_low:.4f}")
        
        if upper_percentile is not None:
            threshold_high = np.percentile(scores, upper_percentile)
            df = df[df[score_column] <= threshold_high]
            print(f"  Upper bound (P{upper_percentile}): {threshold_high:.4f}")
        
        return df
    
    def filter_by_threshold(
        self,
        score_column: str,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None
    ) -> pd.DataFrame:
        """Filter samples by absolute threshold"""
        df = self.df_scores.copy()
        
        if score_column not in df.columns:
            raise ValueError(f"Score column '{score_column}' not found")
        
        if min_score is not None:
            df = df[df[score_column] >= min_score]
            print(f"  Min threshold: {min_score:.4f}")
        
        if max_score is not None:
            df = df[df[score_column] <= max_score]
            print(f"  Max threshold: {max_score:.4f}")
        
        return df
    
    def get_samples_by_criteria(self, criteria: Dict) -> pd.DataFrame:
        """Get samples based on filtering criteria"""
        print(f"\nApplying filter: {criteria['name']}")
        
        df = self.df_scores.copy()
        initial_count = len(df)
        
        # Apply filters based on type
        if criteria['type'] == 'percentile':
            df = self.filter_by_percentile(
                score_column=criteria['score_column'],
                lower_percentile=criteria.get('lower_percentile'),
                upper_percentile=criteria.get('upper_percentile')
            )
        elif criteria['type'] == 'threshold':
            df = self.filter_by_threshold(
                score_column=criteria['score_column'],
                min_score=criteria.get('min_score'),
                max_score=criteria.get('max_score')
            )
        elif criteria['type'] == 'top_k':
            k = criteria['k']
            if criteria.get('bottom', False):
                df = df.nsmallest(k, criteria['score_column'])
                print(f"  Bottom {k} samples by {criteria['score_column']}")
            else:
                df = df.nlargest(k, criteria['score_column'])
                print(f"  Top {k} samples by {criteria['score_column']}")
        
        print(f"  Filtered: {initial_count:,} -> {len(df):,} samples")
        
        # Apply sampling if specified
        if 'sample_size' in criteria and criteria['sample_size'] < len(df):
            df = df.sample(n=criteria['sample_size'], random_state=42)
            print(f"  Sampled: {criteria['sample_size']} samples")
        
        return df
    
    def save_filtered_results(self, df: pd.DataFrame, name: str):
        """Save filtered results"""
        output_path = self.output_dir / f"filtered_{name}"
        output_path.mkdir(exist_ok=True)
        
        # Save as CSV
        csv_path = output_path / "samples.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved to: {csv_path}")
        
        # Save statistics
        stats = {
            'name': name,
            'total_samples': len(df),
            'statistics': {}
        }
        
        score_cols = [col for col in df.columns if 'score' in col]
        for col in score_cols:
            scores = df[col].values
            stats['statistics'][col] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores))
            }
        
        stats_path = output_path / "statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return output_path
    
    def extract_sample_images(self, df: pd.DataFrame, name: str, max_images: int = 100):
        """Extract actual images for filtered samples"""
        if 'extract_images' not in self.config or not self.config['extract_images']:
            return
        
        print(f"  Extracting sample images (max {max_images})...")
        
        output_path = self.output_dir / f"filtered_{name}" / "sample_images"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get WebDataset path
        webdataset_dir = Path(self.config.get('webdataset_dir', ''))
        if not webdataset_dir.exists():
            print("  WebDataset directory not found, skipping image extraction")
            return
        
        # Sample uniformly if we have more samples than max_images
        if len(df) > max_images:
            # Sample uniformly across score distribution
            df_sample = df.sample(n=max_images, random_state=42).sort_values('q_align_quality_score')
        else:
            df_sample = df
        
        # For broader search, we'll search multiple shards if needed
        # Get all tar files
        all_tar_files = sorted(webdataset_dir.glob("*.tar"))
        print(f"  Found {len(all_tar_files)} tar files to search")
        
        # Prepare samples to find
        samples_to_find = set(df_sample['sample_id'].values)
        total_to_extract = len(samples_to_find)
        
        extracted = 0
        samples_found = set()
        pbar = tqdm(total=total_to_extract, desc="    Extracting images")
        
        # Create a lookup for sample info
        sample_info_dict = {row['sample_id']: row for _, row in df_sample.iterrows()}
        
        # Search through tar files until we find all samples or exhaust our search
        max_shards_to_search = min(10, len(all_tar_files))  # Limit search to first 10 shards for efficiency
        
        for tar_idx, tar_file in enumerate(all_tar_files[:max_shards_to_search]):
            if len(samples_found) >= total_to_extract:
                break
                
            remaining = samples_to_find - samples_found
            if not remaining:
                break
            
            try:
                # Load tar and extract samples
                dataset = wds.WebDataset(str(tar_file), handler=wds.ignore_and_continue)
                
                found_in_this_shard = 0
                for sample in dataset:
                    if sample['__key__'] in remaining:
                        try:
                            # Save image
                            if 'jpg' in sample:
                                img_bytes = sample['jpg']
                                ext = 'jpg'
                            elif 'png' in sample:
                                img_bytes = sample['png']
                                ext = 'png'
                            elif 'jpeg' in sample:
                                img_bytes = sample['jpeg']
                                ext = 'jpeg'
                            else:
                                continue
                            
                            img = Image.open(BytesIO(img_bytes))
                            
                            # Get score info
                            sample_info = sample_info_dict[sample['__key__']]
                            
                            # Save with score in filename
                            if 'q_align_quality_score' in sample_info:
                                q_score = sample_info['q_align_quality_score']
                                a_score = sample_info.get('q_align_aesthetic_score', 0)
                                c_score = sample_info.get('q_align_combined_score', 0)
                                filename = f"{sample['__key__']}_q{q_score:.3f}_a{a_score:.3f}_c{c_score:.3f}.{ext}"
                            else:
                                filename = f"{sample['__key__']}.{ext}"
                            
                            img.save(output_path / filename)
                            extracted += 1
                            samples_found.add(sample['__key__'])
                            found_in_this_shard += 1
                            pbar.update(1)
                            
                            if len(samples_found) >= total_to_extract:
                                break
                                
                        except Exception as e:
                            print(f"\n    Error processing sample {sample['__key__']}: {e}")
                
                if found_in_this_shard > 0:
                    print(f"\n    Found {found_in_this_shard} samples in {tar_file.name}")
                    
            except Exception as e:
                print(f"\n    Error processing tar file {tar_file.name}: {e}")
        
        # Update progress for any remaining unfound samples
        not_found = total_to_extract - len(samples_found)
        if not_found > 0:
            pbar.update(not_found)
        
        pbar.close()
        print(f"  Extracted {extracted} images to {output_path}")
        if not_found > 0:
            print(f"  Could not find {not_found} images (searched {min(max_shards_to_search, len(all_tar_files))} tar files)")
    
    def run_all_filters(self):
        """Run all configured filters"""
        print("\n" + "="*60)
        print("Running Filters")
        print("="*60)
        
        for criteria in self.config['filters']:
            df_filtered = self.get_samples_by_criteria(criteria)
            
            if len(df_filtered) > 0:
                output_path = self.save_filtered_results(df_filtered, criteria['name'])
                
                # Extract sample images if configured
                if self.config.get('extract_sample_images', False):
                    self.extract_sample_images(
                        df_filtered, 
                        criteria['name'],
                        max_images=self.config.get('max_sample_images', 100)
                    )
            else:
                print(f"  No samples matched criteria")
        
        print("\n" + "="*60)
        print("Filtering Complete")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Filter samples by Q-Align scores')
    parser.add_argument(
        '--config',
        type=str,
        default='config_filter.yaml',
        help='Path to filter configuration file'
    )
    
    args = parser.parse_args()
    
    filter = QAlignFilter(args.config)
    filter.run_all_filters()

if __name__ == '__main__':
    main()