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
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import ray
import torch
import webdataset as wds
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.indexed_dataset_megatron import DType, IndexedDatasetBuilder
from utils.tokenization_emu3_image_only import EMU3ImageOnlyTokenizer
from utils.parse_utils import add_emu3_tokenization_args


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
        # Setup imports for Ray worker
        self._setup_imports()

        self.config = config
        self.worker_id = worker_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Setup logging
        self.logger = logging.getLogger(f"Worker{worker_id:02d}")
        self.logger.setLevel(logging.INFO)

        # Set number of decode workers
        self.num_decode_workers = min(os.cpu_count(), 32)

        # Initialize tokenizer
        self._initialize_tokenizer()

        # Initialize builder
        self._initialize_builder()
        
        # Statistics
        self.stats = {
            'shards_processed': 0,
            'samples_processed': 0,
            'tokens_generated': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        self.logger.info(f"Worker {worker_id} initialized on {self.device}")

    def _setup_imports(self):
        """Setup Python paths and imports for Ray worker."""
        import sys
        from pathlib import Path

        # Add paths for local imports
        sys.path.append(str(Path(__file__).parent.parent))
        sys.path.append(str(Path(__file__).parent.parent / "utils"))

    def _initialize_tokenizer(self):
        """Initialize the EMU3 tokenizer."""
        # Import here after paths are set
        from tokenization_emu3_image_only import EMU3ImageOnlyTokenizer

        self.tokenizer = EMU3ImageOnlyTokenizer(
            text_tokenizer_path=self.config['tokenizer_path'],
            device=self.device,
            min_pixels=self.config.get('min_resolution')
        )

    def _initialize_builder(self):
        """Initialize the indexed dataset builder."""
        import os

        # Import here after paths are set
        from indexed_dataset_megatron import DType, IndexedDatasetBuilder

        # Store imports for later use
        self.DType = DType
        self.IndexedDatasetBuilder = IndexedDatasetBuilder

        # Create output directory if it doesn't exist
        output_dir = self.config['output_dir']

        # Add range info to output directory name if specified
        if self.config.get('range'):
            output_dir = f"{output_dir}_range_{self.config['range'].replace(':', '-')}"

        # Add resolution info to output directory name if filtering is enabled
        if self.config.get('min_resolution') or self.config.get('max_resolution'):
            min_res = self.config.get('min_resolution', 0)
            max_res = self.config.get('max_resolution', 'inf')
            output_dir = f"{output_dir}_res_{min_res}_{max_res}"

        os.makedirs(output_dir, exist_ok=True)

        # Use rank-based naming inside the directory
        self.output_path = os.path.join(output_dir, f"rank_{self.worker_id:03d}")
        self.builder = IndexedDatasetBuilder(
            f"{self.output_path}.bin",
            dtype=DType.optimal_dtype(self.tokenizer.text_tokenizer.vocab_size)
        )

    def process_shard(self, shard_path: str) -> Dict[str, Any]:
        """Process a single shard."""
        from pathlib import Path
        import time
        import torch
        import webdataset as wds
        from PIL import Image
        import io

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
                .batched(64)  # Load 64 at a time for better I/O
                .unbatched()  # Back to individual samples for simpler iteration
            )

            for img, json_data, key in dataset:
                # Apply filtering only if metadata has width/height
                if 'width' in json_data and 'height' in json_data:
                    w, h = json_data['width'], json_data['height']
                    resolution = w * h

                    # Skip images that don't meet resolution criteria
                    if self.config.get('min_resolution') and resolution < self.config['min_resolution']:
                        continue
                    if self.config.get('max_resolution') and resolution > self.config['max_resolution']:
                        continue
                # If no width/height metadata, process without filtering

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

    # Apply range filter if specified
    if config.get('range'):
        try:
            start, end = map(int, config['range'].split(':'))
            input_files = input_files[start:end]
            logging.info(f"Processing range {start}:{end} of shards")
        except (ValueError, IndexError) as e:
            logging.error(f"Invalid range format: {config['range']}. Use format 'start:end'")
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
    parser = add_emu3_tokenization_args(
        description="EMU3 tokenization with dynamic Ray scheduling"
    )
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