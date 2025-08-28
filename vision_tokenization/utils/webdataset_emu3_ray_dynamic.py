#!/usr/bin/env python3
"""
Ray-based EMU3 tokenization with dynamic work scheduling.
GPUs pull work as they become available (work stealing pattern).
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import ray
import torch
import webdataset as wds
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from indexed_dataset_megatron import DType, IndexedDatasetBuilder
from tokenization_emu3_image_only import EMU3ImageOnlyTokenizer


# ============================================================================
# Work Queue for Dynamic Scheduling
# ============================================================================

@ray.remote
class ShardQueue:
    """Centralized queue of shards to process."""
    
    def __init__(self, shard_paths: List[str]):
        self.pending = list(shard_paths)
        self.in_progress = {}  # shard -> (worker_id, start_time)
        self.completed = []
        self.failed = []
    
    def get_next_shard(self, worker_id: int) -> Optional[str]:
        """Get next available shard for processing."""
        if self.pending:
            shard = self.pending.pop(0)
            self.in_progress[shard] = (worker_id, time.time())
            return shard
        return None
    
    def mark_completed(self, shard: str):
        """Mark shard as successfully completed."""
        if shard in self.in_progress:
            del self.in_progress[shard]
        self.completed.append(shard)
    
    def mark_failed(self, shard: str, error: str):
        """Mark shard as failed."""
        if shard in self.in_progress:
            del self.in_progress[shard]
        self.failed.append((shard, error))
        # Could re-add to pending for retry
    
    def get_status(self) -> Dict:
        """Get current processing status."""
        return {
            'pending': len(self.pending),
            'in_progress': len(self.in_progress),
            'completed': len(self.completed),
            'failed': len(self.failed),
            'total': len(self.pending) + len(self.in_progress) + len(self.completed) + len(self.failed)
        }


# ============================================================================
# GPU Worker with Dynamic Work Pulling
# ============================================================================

@ray.remote(num_gpus=1)
class EMU3DynamicWorker:
    """Worker that pulls shards dynamically from queue."""
    
    def __init__(self, config: Dict, worker_id: int):
        self.config = config
        self.worker_id = worker_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Setup logging
        self.logger = logging.getLogger(f"Worker{worker_id:02d}")
        self.logger.setLevel(logging.INFO)
        
        # Initialize tokenizer
        self.tokenizer = EMU3ImageOnlyTokenizer(
            text_tokenizer_path=config['tokenizer_path'],
            device=self.device
        )
        
        # Initialize builder
        self.output_path = f"{config['output_prefix']}_worker{worker_id}"
        self.builder = IndexedDatasetBuilder(
            f"{self.output_path}.bin",
            dtype=DType.optimal_dtype(config.get('vocab_size', 161129))
        )
        
        # Statistics
        self.stats = {
            'shards_processed': 0,
            'samples_processed': 0,
            'tokens_generated': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        self.logger.info(f"Worker {worker_id} initialized on {self.device}")
    
    def process_shard(self, shard_path: str) -> Dict[str, Any]:
        """Process a single shard."""
        shard_name = Path(shard_path).name
        self.logger.info(f"Starting {shard_name}")
        
        start_time = time.time()
        samples = 0
        tokens = 0
        
        try:
            # Create dataset with small batches for I/O efficiency
            dataset = (
                wds.WebDataset(shard_path, shardshuffle=False)
                .decode("pil")
                .to_tuple("jpg;png;jpeg;webp", "json", "__key__")
                .batched(8)  # Load 8 at a time for better I/O
            )
            
            for batch in dataset:
                images, _, keys = batch
                
                # Process each image
                for img, key in zip(images, keys):
                    try:
                        # Tokenize
                        img_tokens = self.tokenizer.tokenize_image(img)
                        tokens_np = img_tokens.cpu().numpy() if torch.is_tensor(img_tokens) else img_tokens
                        
                        # Add to dataset
                        self.builder.add_document(tokens_np, [len(tokens_np)])
                        
                        samples += 1
                        tokens += len(tokens_np)
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing {key}: {e}")
                        self.stats['errors'] += 1
            
            # Update stats
            self.stats['shards_processed'] += 1
            self.stats['samples_processed'] += samples
            self.stats['tokens_generated'] += tokens
            
            elapsed = time.time() - start_time
            self.logger.info(
                f"Completed {shard_name}: {samples} samples, {tokens} tokens in {elapsed:.1f}s"
            )
            
            return {
                'success': True,
                'shard': shard_name,
                'samples': samples,
                'tokens': tokens,
                'time': elapsed
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process {shard_name}: {e}")
            return {
                'success': False,
                'shard': shard_name,
                'error': str(e)
            }
    
    def run(self, shard_queue) -> Dict[str, Any]:
        """Main loop: pull and process shards until done."""
        self.logger.info("Starting work loop")
        
        while True:
            # Get next shard from queue
            shard_path = ray.get(shard_queue.get_next_shard.remote(self.worker_id))
            
            if shard_path is None:
                self.logger.info("No more shards, finishing")
                break
            
            # Process the shard
            result = self.process_shard(shard_path)
            
            # Report result back to queue
            if result['success']:
                ray.get(shard_queue.mark_completed.remote(shard_path))
            else:
                ray.get(shard_queue.mark_failed.remote(shard_path, result.get('error', 'Unknown')))
        
        # Finalize dataset
        self.builder.finalize(f"{self.output_path}.idx")
        
        # Return final statistics
        elapsed = time.time() - self.stats['start_time']
        self.stats['elapsed_time'] = elapsed
        self.stats['throughput'] = self.stats['tokens_generated'] / elapsed if elapsed > 0 else 0
        
        return self.stats


# ============================================================================
# Main Pipeline with Dynamic Scheduling
# ============================================================================

def process_with_dynamic_scheduling(config: Dict):
    """Main processing with dynamic work distribution."""
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    resources = ray.available_resources()
    num_gpus = config.get('num_gpus') or int(resources.get('GPU', 1))
    
    logging.info(f"Starting with {num_gpus} GPU workers")
    
    # Get list of shards
    input_files = sorted(glob.glob(config['input_pattern']))
    if not input_files:
        logging.error(f"No files found: {config['input_pattern']}")
        return
    
    logging.info(f"Found {len(input_files)} shards to process")
    
    # Create shard queue
    shard_queue = ShardQueue.remote(input_files)
    
    # Create workers
    workers = []
    for i in range(num_gpus):
        worker = EMU3DynamicWorker.remote(config, i)
        workers.append(worker)
    
    # Start all workers (they'll pull work as needed)
    futures = [worker.run.remote(shard_queue) for worker in workers]
    
    # Monitor progress
    pbar = tqdm(total=len(input_files), desc="Shards processed")
    last_completed = 0
    
    while True:
        # Check queue status
        status = ray.get(shard_queue.get_status.remote())
        
        # Update progress bar
        completed = status['completed']
        if completed > last_completed:
            pbar.update(completed - last_completed)
            last_completed = completed
        
        # Check if done
        if status['pending'] == 0 and status['in_progress'] == 0:
            break
        
        time.sleep(1)  # Check every second
    
    pbar.close()
    
    # Get final results from all workers
    results = ray.get(futures)
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    
    total_shards = sum(r['shards_processed'] for r in results)
    total_samples = sum(r['samples_processed'] for r in results)
    total_tokens = sum(r['tokens_generated'] for r in results)
    total_errors = sum(r['errors'] for r in results)
    max_time = max(r['elapsed_time'] for r in results)
    
    for i, r in enumerate(results):
        print(f"Worker {i}: {r['shards_processed']} shards, "
              f"{r['samples_processed']} samples, "
              f"{r['throughput']:.0f} tok/s")
    
    print("-"*60)
    print(f"Total: {total_shards} shards, {total_samples} samples, {total_tokens} tokens")
    print(f"Errors: {total_errors}")
    print(f"Time: {max_time:.1f}s")
    print(f"Overall throughput: {total_tokens/max_time:.0f} tokens/sec")
    print("="*60)
    
    # Check for failed shards
    final_status = ray.get(shard_queue.get_status.remote())
    if final_status['failed'] > 0:
        print(f"\nWarning: {final_status['failed']} shards failed processing")
    
    ray.shutdown()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EMU3 tokenization with dynamic Ray scheduling"
    )
    parser.add_argument("--input-pattern", required=True, help="Input shard pattern")
    parser.add_argument("--output-prefix", required=True, help="Output prefix")
    parser.add_argument("--tokenizer-path", required=True, help="EMU3 tokenizer path")
    parser.add_argument("--num-gpus", type=int, help="Number of GPUs")
    parser.add_argument("--vocab-size", type=int, default=161129)
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    config = vars(args)
    process_with_dynamic_scheduling(config)


if __name__ == "__main__":
    main()