#!/usr/bin/env python3
"""
Configurable Q-Align scoring for WebDataset
Uses YAML configuration for all parameters
"""

import os
import sys
import yaml
import json
import time
import torch
import torch.multiprocessing as mp
import webdataset as wds
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add Q-Align to path
sys.path.insert(0, '/iopsstor/scratch/cscs/rqu/benchmark-image-tokenzier/datasets/data_filtering/Q-Align')

class ConfigurableQAlignScorer:
    def __init__(self, config_path: str):
        """Initialize scorer with configuration file"""
        self.config = self.load_config(config_path)
        self.setup_environment()
        self.setup_paths()
        self.setup_checkpoint()
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from: {config_path}")
        return config
    
    def setup_environment(self):
        """Setup environment variables"""
        os.environ['HF_HOME'] = self.config['paths']['cache_dir']
        if self.config['runtime']['debug']:
            print("DEBUG MODE ENABLED - Processing limited shards")
    
    def setup_paths(self):
        """Setup input/output paths"""
        self.input_dir = Path(self.config['paths']['input_dir'])
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.log_dir = Path(self.config['paths']['log_dir'])
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'q_align_scoring_{timestamp}.log'
        
        # Get tar files
        self.tar_files = sorted(self.input_dir.glob(self.config['dataset']['file_pattern']))
        
        # Apply debug limit
        if self.config['runtime']['debug']:
            self.tar_files = self.tar_files[:5]
        elif self.config['processing']['max_shards']:
            self.tar_files = self.tar_files[:self.config['processing']['max_shards']]
        
        if self.config['processing']['shuffle_shards']:
            import random
            random.shuffle(self.tar_files)
        
        print(f"Found {len(self.tar_files)} tar files to process")
    
    def setup_checkpoint(self):
        """Setup checkpointing system"""
        self.checkpoint_file = self.output_dir / 'checkpoint.json'
        self.completed_shards = set()
        
        if self.config['runtime']['checkpoint_enabled'] and self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                self.completed_shards = set(checkpoint.get('completed_shards', []))
                print(f"Resuming from checkpoint: {len(self.completed_shards)} shards already completed")
    
    def save_checkpoint(self, new_completed: List[str]):
        """Save checkpoint with completed shards"""
        if not self.config['runtime']['checkpoint_enabled']:
            return
        
        self.completed_shards.update(new_completed)
        checkpoint = {
            'completed_shards': list(self.completed_shards),
            'timestamp': datetime.now().isoformat(),
            'total_processed': len(self.completed_shards),
            'total_shards': len(self.tar_files)
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def extract_shard_info(self, tar_path: Path) -> Tuple[int, int]:
        """Extract shard and subshard indices from filename"""
        name = tar_path.stem
        parts = name.split('-')
        if len(parts) >= 3:
            shard_idx = int(parts[1])
            subshard_idx = int(parts[2])
            return shard_idx, subshard_idx
        return 0, 0
    
    def process_shard_batch(self, args):
        """Process a batch of shards on a single GPU"""
        shard_paths, gpu_id, config = args
        
        # Import inside process
        from q_align import QAlignScorer, QAlignAestheticScorer
        
        # Set device
        if gpu_id >= 0:
            torch.cuda.set_device(gpu_id)
            device = f"cuda:{gpu_id}"
        else:
            device = "cpu"
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Worker on {device} starting with {len(shard_paths)} shards")
        
        # Initialize models based on config
        models = {}
        if 'quality' in config['model']['score_types']:
            models['quality'] = QAlignScorer(
                pretrained=config['model']['pretrained'],
                device=device
            )
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Quality scorer loaded on {device}")
        
        if 'aesthetic' in config['model']['score_types']:
            models['aesthetic'] = QAlignAestheticScorer(
                pretrained=config['model']['pretrained'],
                device=device
            )
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Aesthetic scorer loaded on {device}")
        
        all_results = []
        completed_shards = []
        
        for i, shard_path in enumerate(shard_paths, 1):
            shard_path = Path(shard_path)
            shard_idx, subshard_idx = self.extract_shard_info(shard_path)
            
            try:
                start_time = time.time()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {device}: Processing {shard_path.name} ({i}/{len(shard_paths)})")
                
                results = self.process_single_shard(
                    shard_path,
                    models,
                    shard_idx,
                    subshard_idx,
                    config
                )
                
                all_results.extend(results)
                completed_shards.append(str(shard_path))
                
                elapsed = time.time() - start_time
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {device}: Completed {shard_path.name} - {len(results)} samples in {elapsed:.1f}s")
                    
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {device}: ERROR processing {shard_path.name}: {e}")
                if config['runtime']['retry_failed_shards']:
                    # Could implement retry logic here
                    pass
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {device}: Worker finished - processed {len(completed_shards)} shards, {len(all_results)} samples total")
        return all_results, completed_shards
    
    def process_single_shard(
        self,
        shard_path: Path,
        models: Dict,
        shard_idx: int,
        subshard_idx: int,
        config: Dict
    ) -> List[Dict]:
        """Process a single shard"""
        results = []
        dataset = wds.WebDataset(str(shard_path))
        
        batch_images = []
        batch_ids = []
        batch_size = config['processing']['batch_size']
        
        for sample in dataset:
            try:
                sample_id = sample['__key__']
                
                if 'jpg' in sample:
                    img_bytes = sample['jpg']
                elif 'png' in sample:
                    img_bytes = sample['png']
                else:
                    continue
                
                img = Image.open(BytesIO(img_bytes)).convert('RGB')
                batch_images.append(img)
                batch_ids.append(sample_id)
                
                if len(batch_images) >= batch_size:
                    batch_results = self.score_batch(
                        batch_images,
                        batch_ids,
                        models,
                        shard_idx,
                        subshard_idx,
                        config
                    )
                    results.extend(batch_results)
                    batch_images = []
                    batch_ids = []
                    
            except Exception as e:
                if config['runtime']['verbose']:
                    print(f"Error processing sample: {e}")
                continue
        
        # Process remaining
        if batch_images:
            batch_results = self.score_batch(
                batch_images,
                batch_ids,
                models,
                shard_idx,
                subshard_idx,
                config
            )
            results.extend(batch_results)
        
        return results
    
    def score_batch(
        self,
        images: List,
        sample_ids: List,
        models: Dict,
        shard_idx: int,
        subshard_idx: int,
        config: Dict
    ) -> List[Dict]:
        """Score a batch of images"""
        results = []
        
        # Get scores
        scores = {}
        with torch.no_grad():
            if 'quality' in models:
                scores['quality'] = models['quality'](images).cpu().numpy()
            if 'aesthetic' in models:
                scores['aesthetic'] = models['aesthetic'](images).cpu().numpy()
        
        # Create records
        for i, sample_id in enumerate(sample_ids):
            record = {
                'sample_id': sample_id,
                'shard_idx': shard_idx,
                'subshard_idx': subshard_idx
            }
            
            if 'quality' in scores:
                record['q_align_quality_score'] = float(scores['quality'][i])
            
            if 'aesthetic' in scores:
                record['q_align_aesthetic_score'] = float(scores['aesthetic'][i])
            
            if config['model']['compute_combined'] and 'quality' in scores and 'aesthetic' in scores:
                record['q_align_combined_score'] = float(
                    (scores['quality'][i] + scores['aesthetic'][i]) / 2
                )
            
            results.append(record)
        
        return results
    
    def run(self):
        """Run the scoring pipeline"""
        start_time = time.time()
        
        # Filter out already completed shards
        shards_to_process = [
            tar for tar in self.tar_files
            if str(tar) not in self.completed_shards
        ]
        
        if not shards_to_process:
            print("All shards already processed!")
            return
        
        # Print detailed progress information
        print(f"\n{'='*60}")
        print(f"Starting Q-Align Scoring")
        print(f"{'='*60}")
        print(f"Total shards to process: {len(shards_to_process)}")
        print(f"Already completed: {len(self.completed_shards)}")
        print(f"Estimated samples: {len(shards_to_process) * self.config['dataset']['samples_per_shard']:,}")
        print(f"{'='*60}\n")
        
        # Setup GPU assignments
        gpu_ids = self.config['processing']['gpu_ids']
        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))
        
        num_workers = min(self.config['processing']['num_workers'], len(gpu_ids))
        
        # Distribute shards
        gpu_assignments = [[] for _ in range(num_workers)]
        for i, tar_path in enumerate(shards_to_process):
            worker_idx = i % num_workers
            gpu_assignments[worker_idx].append(str(tar_path))
        
        # Prepare work items
        work_items = []
        for worker_idx in range(num_workers):
            if gpu_assignments[worker_idx]:
                gpu_id = gpu_ids[worker_idx % len(gpu_ids)]
                work_items.append((
                    gpu_assignments[worker_idx],
                    gpu_id,
                    self.config
                ))
        
        print(f"Starting {len(work_items)} workers on GPUs: {gpu_ids}")
        for i, (shards, gpu_id, _) in enumerate(work_items):
            print(f"  Worker {i} (GPU {gpu_id}): {len(shards)} shards")
        print()
        
        # Process in parallel
        all_results = []
        all_completed = []
        total_samples_processed = 0
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.process_shard_batch, item) for item in work_items]
            
            # Enhanced progress bar with more info
            with tqdm(total=len(shards_to_process), desc="Shards", position=0) as pbar:
                pbar.set_postfix({
                    'Samples': 0,
                    'Speed': '0/s',
                    'Remaining': len(shards_to_process)
                })
                
                batch_start_time = time.time()
                
                for future in as_completed(futures):
                    try:
                        results, completed = future.result(
                            timeout=self.config['runtime']['timeout_per_shard'] * 100
                        )
                        all_results.extend(results)
                        all_completed.extend(completed)
                        total_samples_processed += len(results)
                        
                        # Update progress bar
                        pbar.update(len(completed))
                        
                        # Calculate speed
                        elapsed = time.time() - batch_start_time
                        speed = total_samples_processed / elapsed if elapsed > 0 else 0
                        remaining_shards = len(shards_to_process) - len(all_completed)
                        
                        # Update detailed progress
                        pbar.set_postfix({
                            'Samples': f'{total_samples_processed:,}',
                            'Speed': f'{speed:.1f}/s',
                            'Remaining': remaining_shards
                        })
                        
                        # Print milestone messages
                        if len(all_completed) % 50 == 0:
                            eta_seconds = remaining_shards * (elapsed / len(all_completed)) if len(all_completed) > 0 else 0
                            eta_minutes = eta_seconds / 60
                            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Milestone: {len(all_completed)}/{len(shards_to_process)} shards completed")
                            print(f"  Total samples processed: {total_samples_processed:,}")
                            print(f"  Average speed: {speed:.1f} samples/s")
                            print(f"  Estimated time remaining: {eta_minutes:.1f} minutes\n")
                        
                        # Save checkpoint periodically
                        if len(all_completed) % self.config['runtime']['checkpoint_interval'] == 0:
                            self.save_checkpoint(all_completed)
                            
                    except Exception as e:
                        print(f"Worker error: {e}")
        
        # Save final results
        if all_results:
            self.save_results(all_results)
            self.save_checkpoint(all_completed)
            
            elapsed = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"Processing Complete!")
            print(f"{'='*60}")
            print(f"Total time: {elapsed/60:.1f} minutes")
            print(f"Total samples: {len(all_results):,}")
            print(f"Samples per second: {len(all_results)/elapsed:.1f}")
    
    def save_results(self, results: List[Dict]):
        """Save results to disk"""
        print(f"Saving {len(results)} results...")
        
        df = pd.DataFrame(results)
        df = df.sort_values(['shard_idx', 'subshard_idx', 'sample_id'])
        
        # Save based on format config
        save_format = self.config['output']['save_format']
        
        if save_format in ['parquet', 'both']:
            table = pa.Table.from_pandas(df)
            pq.write_to_dataset(
                table,
                root_path=str(self.output_dir),
                partition_cols=self.config['output']['partition_by'],
                compression=self.config['output']['compression'],
                existing_data_behavior='overwrite_or_ignore'
            )
            print(f"Parquet saved to: {self.output_dir}")
        
        if save_format in ['csv', 'both']:
            csv_path = self.output_dir / 'q_align_scores.csv'
            df.to_csv(csv_path, index=False)
            print(f"CSV saved to: {csv_path}")
        
        # Compute and save statistics
        if self.config['statistics']['compute_stats']:
            stats = self.compute_statistics(df)
            stats_path = self.output_dir / 'q_align_statistics.json'
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Statistics saved to: {stats_path}")
            
            # Print summary
            self.print_summary(stats)
    
    def compute_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute comprehensive statistics"""
        stats = {
            'metadata': {
                'total_samples': len(df),
                'unique_shards': df['shard_idx'].nunique() if 'shard_idx' in df else 0,
                'config_file': str(self.config),
                'timestamp': datetime.now().isoformat()
            },
            'scores': {},
            'thresholds': self.config.get('thresholds', {})
        }
        
        score_cols = [col for col in df.columns if 'score' in col]
        
        for col in score_cols:
            if col in df.columns:
                scores = df[col].values
                stats['scores'][col] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'percentiles': {
                        str(p): float(np.percentile(scores, p))
                        for p in self.config['statistics']['percentiles']
                    }
                }
                
                if self.config['statistics']['histogram_bins']:
                    bins = np.linspace(0, 1, self.config['statistics']['histogram_bins'] + 1)
                    hist, _ = np.histogram(scores, bins=bins)
                    stats['scores'][col]['histogram'] = {
                        'bins': bins.tolist(),
                        'counts': hist.tolist()
                    }
        
        return stats
    
    def print_summary(self, stats: Dict):
        """Print summary statistics"""
        print(f"\n{'='*60}")
        print("Score Statistics:")
        print(f"{'='*60}")
        
        for score_name, score_stats in stats['scores'].items():
            print(f"\n{score_name}:")
            print(f"  Mean: {score_stats['mean']:.4f} ± {score_stats['std']:.4f}")
            print(f"  Range: [{score_stats['min']:.4f}, {score_stats['max']:.4f}]")
            print(f"  Median (P50): {score_stats['percentiles']['50']:.4f}")
            print(f"  P90: {score_stats['percentiles']['90']:.4f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Q-Align scoring with configuration')
    parser.add_argument(
        '--config',
        type=str,
        default='config_q_align_scoring.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Set spawn method for multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Run scorer
    scorer = ConfigurableQAlignScorer(args.config)
    scorer.run()


if __name__ == '__main__':
    main()