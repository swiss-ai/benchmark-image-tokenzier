#!/usr/bin/env python3
"""
Process webdataset format data with DFN filtering scores using multi-GPU
"""

import os
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import webdataset as wds
from PIL import Image
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import logging
from datetime import datetime
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
from dfn_filter import DFNFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WebDatasetProcessor:
    """Process webdataset with DFN scores using multi-GPU"""
    
    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        model_name: str = 'hf-hub:apple/DFN-public'
    ):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name
        
        # Create output directory
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
    def get_shard_files(self) -> List[str]:
        """Get all shard files from the dataset directory"""
        dataset_dir = Path(self.dataset_path)
        
        # Common webdataset patterns
        patterns = ['*.tar', '*.tar.gz', '*.tgz']
        shard_files = []
        
        for pattern in patterns:
            shard_files.extend(sorted(dataset_dir.glob(pattern)))
        
        if not shard_files:
            raise ValueError(f"No webdataset shards found in {self.dataset_path}")
        
        logger.info(f"Found {len(shard_files)} shard files")
        return [str(f) for f in shard_files]
    
    def process_shard_batch(
        self,
        batch_data: List[Dict],
        model: DFNFilter,
        device: str
    ) -> List[Dict]:
        """Process a batch of samples"""
        
        results = []
        
        # Prepare batch
        images = []
        texts = []
        sample_ids = []
        
        for sample in batch_data:
            try:
                # Extract image and text
                if 'jpg' in sample:
                    img_data = sample['jpg']
                elif 'png' in sample:
                    img_data = sample['png']
                elif 'image' in sample:
                    img_data = sample['image']
                else:
                    logger.warning(f"No image found in sample {sample.get('__key__', 'unknown')}")
                    continue
                
                # Get text/caption
                text = ""
                if 'txt' in sample:
                    text = sample['txt']
                elif 'text' in sample:
                    text = sample['text']
                elif 'caption' in sample:
                    text = sample['caption']
                elif 'json' in sample:
                    # Try to extract caption from JSON
                    json_data = sample['json']
                    if isinstance(json_data, bytes):
                        json_data = json_data.decode('utf-8')
                    if isinstance(json_data, str):
                        json_data = json.loads(json_data)
                    
                    # For LLaVA dataset, caption is in the GPT response
                    if 'conversations' in json_data and len(json_data['conversations']) > 1:
                        # Find the gpt/assistant response
                        for conv in json_data['conversations']:
                            if conv.get('from') in ['gpt', 'assistant']:
                                text = conv.get('value', '')
                                break
                    
                    if not text:
                        text = json_data.get('caption', '') or json_data.get('text', '')
                else:
                    text = ""
                
                # Convert to PIL Image if needed
                if isinstance(img_data, bytes):
                    import io
                    img = Image.open(io.BytesIO(img_data)).convert('RGB')
                elif isinstance(img_data, Image.Image):
                    img = img_data.convert('RGB')
                else:
                    img = Image.fromarray(img_data).convert('RGB')
                
                images.append(img)
                texts.append(str(text))
                sample_ids.append(sample.get('__key__', f"sample_{len(sample_ids)}"))
                
            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
                continue
        
        if not images:
            return results
        
        # Compute DFN scores
        try:
            pairs = list(zip(images, texts))
            scores = model.batch_compute_scores(pairs, show_progress=False)
            
            # Create results
            for sid, score, text in zip(sample_ids, scores, texts):
                results.append({
                    'sample_id': sid,
                    'dfn_score': float(score),
                    'caption_length': len(text),
                    'caption_preview': text[:100] if text else ""
                })
                
        except Exception as e:
            logger.error(f"Error computing scores: {e}")
        
        return results
    
    def process_shard_on_gpu(
        self,
        rank: int,
        world_size: int,
        shard_files: List[str],
        output_queue: Optional[mp.Queue] = None
    ):
        """Process shards on a specific GPU"""
        
        # Set device
        device = f'cuda:{rank}'
        torch.cuda.set_device(device)
        
        logger.info(f"GPU {rank}: Initializing DFN model")
        model = DFNFilter(model_name=self.model_name, device=device, batch_size=self.batch_size)
        
        # Split shards among GPUs
        shards_per_gpu = len(shard_files) // world_size
        extra_shards = len(shard_files) % world_size
        
        start_idx = rank * shards_per_gpu + min(rank, extra_shards)
        end_idx = start_idx + shards_per_gpu + (1 if rank < extra_shards else 0)
        
        gpu_shards = shard_files[start_idx:end_idx]
        logger.info(f"GPU {rank}: Processing {len(gpu_shards)} shards")
        
        all_results = []
        
        for shard_idx, shard_path in enumerate(gpu_shards):
            logger.info(f"GPU {rank}: Processing shard {shard_idx + 1}/{len(gpu_shards)}: {Path(shard_path).name}")
            
            # Create webdataset
            dataset = wds.WebDataset(shard_path).decode()
            
            batch_data = []
            shard_results = []
            
            for sample in dataset:
                batch_data.append(sample)
                
                # Process batch when full
                if len(batch_data) >= self.batch_size:
                    batch_results = self.process_shard_batch(batch_data, model, device)
                    shard_results.extend(batch_results)
                    batch_data = []
            
            # Process remaining samples
            if batch_data:
                batch_results = self.process_shard_batch(batch_data, model, device)
                shard_results.extend(batch_results)
            
            all_results.extend(shard_results)
            logger.info(f"GPU {rank}: Shard complete. Processed {len(shard_results)} samples")
        
        # Save results for this GPU
        output_file = Path(self.output_path) / f"dfn_scores_gpu{rank}.parquet"
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_parquet(output_file, engine='pyarrow')
            logger.info(f"GPU {rank}: Saved {len(all_results)} results to {output_file}")
        
        if output_queue:
            output_queue.put((rank, len(all_results)))
    
    def process_dataset_multiGPU(self, max_shards=None, num_gpus=None):
        """Process entire dataset using multiple GPUs"""
        
        logger.info("Starting multi-GPU processing")
        
        # Get shard files
        shard_files = self.get_shard_files()
        
        # Limit shards if specified
        if max_shards is not None and max_shards > 0:
            shard_files = shard_files[:max_shards]
            logger.info(f"Processing {len(shard_files)} shards (limited from total)")
        
        # Get number of GPUs
        available_gpus = torch.cuda.device_count()
        world_size = min(num_gpus, available_gpus) if num_gpus else available_gpus
        logger.info(f"Using {world_size} GPUs (available: {available_gpus})")
        
        # Create process pool
        mp.set_start_method('spawn', force=True)
        processes = []
        output_queue = mp.Queue()
        
        for rank in range(world_size):
            p = mp.Process(
                target=self.process_shard_on_gpu,
                args=(rank, world_size, shard_files, output_queue)
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes
        for p in processes:
            p.join()
        
        # Collect results
        total_samples = 0
        gpu_results = []
        while not output_queue.empty():
            rank, count = output_queue.get()
            gpu_results.append((rank, count))
            total_samples += count
        
        logger.info(f"Processing complete. Total samples: {total_samples}")
        
        # Merge parquet files
        self.merge_results()
        
        return total_samples
    
    def merge_results(self):
        """Merge results from all GPUs into a single parquet file"""
        
        logger.info("Merging results from all GPUs")
        
        output_dir = Path(self.output_path)
        gpu_files = sorted(output_dir.glob("dfn_scores_gpu*.parquet"))
        
        if not gpu_files:
            logger.warning("No GPU result files found")
            return
        
        # Read all parquet files
        dfs = []
        for f in gpu_files:
            df = pd.read_parquet(f)
            dfs.append(df)
            logger.info(f"Loaded {len(df)} results from {f.name}")
        
        # Merge dataframes
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Save merged results
        output_file = output_dir / "dfn_scores_all.parquet"
        merged_df.to_parquet(output_file, engine='pyarrow')
        logger.info(f"Saved merged results ({len(merged_df)} samples) to {output_file}")
        
        # Compute statistics
        stats = {
            'total_samples': len(merged_df),
            'mean_score': float(merged_df['dfn_score'].mean()),
            'std_score': float(merged_df['dfn_score'].std()),
            'min_score': float(merged_df['dfn_score'].min()),
            'max_score': float(merged_df['dfn_score'].max()),
            'percentiles': {
                '10': float(merged_df['dfn_score'].quantile(0.10)),
                '25': float(merged_df['dfn_score'].quantile(0.25)),
                '50': float(merged_df['dfn_score'].quantile(0.50)),
                '75': float(merged_df['dfn_score'].quantile(0.75)),
                '90': float(merged_df['dfn_score'].quantile(0.90)),
            }
        }
        
        # Save statistics
        stats_file = output_dir / "dfn_scores_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_file}")
        logger.info(f"Score statistics: mean={stats['mean_score']:.4f}, std={stats['std_score']:.4f}")
        
        # Clean up GPU files (optional)
        # for f in gpu_files:
        #     f.unlink()
        
        return merged_df


def main():
    parser = argparse.ArgumentParser(description='Process webdataset with DFN scores')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to webdataset directory')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output directory for scores')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--model', type=str, default='hf-hub:apple/DFN-public',
                        help='DFN model to use')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')
    parser.add_argument('--max_shards', type=int, default=None,
                        help='Maximum number of shards to process (default: all)')
    
    args = parser.parse_args()
    
    # Create processor
    processor = WebDatasetProcessor(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_name=args.model
    )
    
    # Process dataset
    start_time = datetime.now()
    total_samples = processor.process_dataset_multiGPU(
        max_shards=args.max_shards,
        num_gpus=args.num_gpus
    )
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Processing complete in {duration:.2f} seconds")
    logger.info(f"Processed {total_samples} samples")
    logger.info(f"Average: {total_samples / duration:.2f} samples/second")


if __name__ == "__main__":
    # Ensure we don't get library conflicts
    if 'LD_LIBRARY_PATH' in os.environ:
        del os.environ['LD_LIBRARY_PATH']
    if 'PYTHONPATH' in os.environ:
        del os.environ['PYTHONPATH']
    
    main()