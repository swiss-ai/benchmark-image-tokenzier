#!/usr/bin/env python3
"""
HuggingFace dataset tokenization with Ray dynamic work-stealing.
Workers pull work as they become available for optimal GPU utilization.
"""

import argparse
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import ray
import torch
from datasets import load_dataset
from tqdm import tqdm

# Add paths for imports
import sys
base_dir = Path(__file__).parent.parent
sys.path.append(str(base_dir))
sys.path.append(str(base_dir / 'Tokenizer'))

from utils.indexed_dataset_megatron import DType, IndexedDatasetBuilder
from utils.tokenization_emu3_image_only import EMU3ImageOnlyTokenizer


@ray.remote
class ProgressActor:
    """Lightweight actor for collecting progress updates without polling."""
    
    def __init__(self, total_samples: int):
        self.total_samples = total_samples
        self.processed = 0
        self.pbar = tqdm(total=total_samples, desc="Samples processed")
    
    def update(self, samples: int):
        """Update progress bar with completed samples."""
        self.processed += samples
        self.pbar.update(samples)
    
    def close(self):
        """Close the progress bar."""
        self.pbar.close()
        return self.processed


@ray.remote
class WorkQueue:
    """Dynamic work queue for distributing batches to workers."""
    
    def __init__(self, total_samples: int, batch_size: int):
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.next_idx = 0
        self.in_progress = {}  # batch_id -> (worker_id, start_time)
        self.completed = []
        self.failed = []
        self.progress_events = []  # Store progress updates for event-driven monitoring
    
    def get_next_batch(self, worker_id: int) -> Optional[Dict]:
        """Get next batch for a worker (work-stealing)."""
        if self.next_idx >= self.total_samples:
            return None
        
        start_idx = self.next_idx
        end_idx = min(start_idx + self.batch_size, self.total_samples)
        batch_id = f"batch_{start_idx:08d}_{end_idx:08d}"
        
        self.next_idx = end_idx
        self.in_progress[batch_id] = (worker_id, time.time())
        
        return {
            'batch_id': batch_id,
            'indices': list(range(start_idx, end_idx))
        }
    
    def mark_completed(self, batch_id: str, stats: Dict):
        """Mark batch as completed."""
        if batch_id in self.in_progress:
            del self.in_progress[batch_id]
        self.completed.append((batch_id, stats))
    
    def mark_failed(self, batch_id: str, error: str):
        """Mark batch as failed."""
        if batch_id in self.in_progress:
            del self.in_progress[batch_id]
        self.failed.append((batch_id, error))
    
    def get_status(self) -> Dict:
        """Get current processing status."""
        return {
            'processed': self.next_idx,
            'total': self.total_samples,
            'in_progress': len(self.in_progress),
            'completed': len(self.completed),
            'failed': len(self.failed)
        }


@ray.remote(num_gpus=1)
class DynamicTokenizerWorker:
    """GPU worker that pulls work dynamically from queue."""
    
    def __init__(self, tokenizer_path: str, output_dir: str, worker_id: int, 
                 min_pixels: int = None, max_pixels: int = None):
        self.worker_id = worker_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Setup logging
        self.logger = logging.getLogger(f"Worker{worker_id:02d}")
        self.logger.setLevel(logging.INFO)
        
        # Initialize tokenizer with custom pixel limits if provided
        self.tokenizer = EMU3ImageOnlyTokenizer(
            text_tokenizer_path=tokenizer_path,
            device=self.device,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        
        # Setup output
        os.makedirs(output_dir, exist_ok=True)
        self.output_path = os.path.join(output_dir, f"rank_{worker_id:03d}")
        self.builder = IndexedDatasetBuilder(
            f"{self.output_path}.bin",
            dtype=DType.optimal_dtype(self.tokenizer.text_tokenizer.vocab_size)
        )
        
        # Initialize prefetch queue for async loading
        self.image_queue = queue.Queue(maxsize=64)
        
        self.stats = {
            'batches_processed': 0,
            'samples_processed': 0,
            'tokens_generated': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        self.logger.info(f"Worker {worker_id} initialized on {self.device}")
    
    def _extract_image(self, sample: Dict):
        """Extract image from sample, handling various formats."""
        for key in ['image', 'img', 'images']:
            if key not in sample:
                continue
            value = sample[key]
            # Handle single image in list
            return value[0] if isinstance(value, list) and len(value) == 1 else value
        return None
    
    def _load_images_async(self, indices, dataset_info):
        """Background thread to load images into queue."""
        try:
            # Load dataset slice
            samples = load_dataset(
                dataset_info['name'],
                name=dataset_info['config'],
                split=f"{dataset_info['split']}[{indices[0]}:{indices[-1]+1}]",
                cache_dir=dataset_info.get('cache_dir')
            )
            
            # Queue images for processing
            for sample in samples:
                if image := self._extract_image(sample):
                    self.image_queue.put(image)
        except Exception as e:
            self.logger.error(f"Loader thread error: {e}")
        finally:
            self.image_queue.put(None)  # Sentinel
    
    def _process_image_stream(self):
        """Process images from queue as GPU becomes available."""
        stats = {'samples': 0, 'tokens': 0, 'errors': 0}
        
        while True:
            image = self.image_queue.get()
            if image is None:  # Done
                break
            
            try:
                # GPU processes immediately - image already loaded
                tokens = self.tokenizer.tokenize_image(image)
                tokens_np = tokens.cpu().numpy() if torch.is_tensor(tokens) else tokens
                
                # Save results
                self.builder.add_document(tokens_np, [len(tokens_np)])
                
                stats['samples'] += 1
                stats['tokens'] += len(tokens_np)
            except Exception as e:
                self.logger.warning(f"Processing error: {e}")
                stats['errors'] += 1
        
        return stats
    
    def process_batch(self, batch_info: Dict, dataset_info: Dict) -> Dict:
        """Process batch with async prefetching for optimal GPU utilization."""
        batch_id = batch_info['batch_id']
        indices = batch_info['indices']
        
        self.logger.info(f"Processing {batch_id} ({len(indices)} samples)")
        start_time = time.time()
        
        # Start async loading
        loader = threading.Thread(
            target=self._load_images_async,
            args=(indices, dataset_info),
            daemon=True
        )
        loader.start()
        
        # Process images as they arrive
        stats = self._process_image_stream()
        
        # Ensure loader completes
        loader.join()
        
        # Update global stats
        self.stats['batches_processed'] += 1
        self.stats['samples_processed'] += stats['samples']
        self.stats['tokens_generated'] += stats['tokens']
        self.stats['errors'] += stats['errors']
        
        # Log completion with average tokens per image for this work batch
        elapsed = time.time() - start_time
        avg_tokens = stats['tokens'] / stats['samples'] if stats['samples'] > 0 else 0
        throughput = stats['samples'] / elapsed if elapsed > 0 else 0
        self.logger.info(
            f"Completed {batch_id}: {stats['samples']} samples, "
            f"{stats['tokens']} tokens ({avg_tokens:.1f} avg/image), "
            f"{throughput:.1f} img/s"
        )
        
        return {
            'batch_id': batch_id,
            'time': elapsed,
            **stats
        }
    
    def run(self, work_queue, dataset_info, progress_actor=None) -> Dict:
        """Main loop: pull and process batches until done."""
        self.logger.info("Starting work loop")
        
        while True:
            # Pull next batch from queue
            batch_info = ray.get(work_queue.get_next_batch.remote(self.worker_id))
            
            if batch_info is None:
                self.logger.info("No more batches, finishing")
                break
            
            # Process the batch
            try:
                result = self.process_batch(batch_info, dataset_info)
                ray.get(work_queue.mark_completed.remote(batch_info['batch_id'], result))
                
                # Report progress if actor provided
                if progress_actor:
                    progress_actor.update.remote(result['samples'])
                    
            except Exception as e:
                self.logger.error(f"Failed to process batch: {e}")
                ray.get(work_queue.mark_failed.remote(batch_info['batch_id'], str(e)))
        
        # Finalize dataset
        self.builder.finalize(f"{self.output_path}.idx")
        
        # Return final stats
        elapsed = time.time() - self.stats['start_time']
        self.stats['elapsed_time'] = elapsed
        self.stats['throughput'] = self.stats['tokens_generated'] / elapsed if elapsed > 0 else 0
        
        return self.stats
    
    def finalize(self) -> Dict:
        """Finalize the indexed dataset and return stats."""
        self.builder.finalize(f"{self.output_path}.idx")
        
        elapsed = time.time() - self.stats['start_time']
        self.stats['elapsed_time'] = elapsed
        self.stats['throughput'] = self.stats['tokens_generated'] / elapsed if elapsed > 0 else 0
        
        return self.stats


# DatasetManager removed - loading dataset directly for better performance


def process_dataset_distributed(args):
    """Main distributed processing with dynamic work-stealing."""
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    resources = ray.available_resources()
    num_gpus = args.num_gpus or int(resources.get('GPU', 1))
    
    logging.info(f"Starting dynamic distributed processing with {num_gpus} GPUs")
    
    # Auto-create output directory with config name
    output_dir = os.path.join(args.output_dir, args.config_name)
    logging.info(f"Output directory: {output_dir}")
    
    # Load dataset (non-streaming for better performance)
    logging.info(f"Loading dataset {args.dataset_name}/{args.config_name}...")
    dataset = load_dataset(
        args.dataset_name,
        name=args.config_name,
        split=args.split,
        cache_dir=args.cache_dir,  # Optional cache directory
        num_proc=128  
    )
    total_samples = len(dataset)
    
    if args.max_samples:
        total_samples = min(total_samples, args.max_samples)
        dataset = dataset.select(range(total_samples))
    
    logging.info(f"Dataset loaded: {total_samples} samples")
    
    # Create dataset info dict to pass to workers
    dataset_info = {
        'name': args.dataset_name,
        'config': args.config_name,
        'split': args.split,
        'cache_dir': args.cache_dir,
        'total_samples': total_samples
    }
    
    # Create work queue
    work_queue = WorkQueue.remote(total_samples, args.batch_size)
    
    # Create workers with custom pixel limits if provided
    workers = []
    for i in range(num_gpus):
        worker = DynamicTokenizerWorker.remote(
            tokenizer_path=args.tokenizer_path,
            output_dir=output_dir,  # Use the auto-created output directory
            worker_id=i,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels
        )
        workers.append(worker)
    
    # Create progress actor for efficient progress tracking
    progress_actor = ProgressActor.remote(total_samples)
    
    # Start all workers with progress actor
    futures = [worker.run.remote(work_queue, dataset_info, progress_actor) 
               for worker in workers]
    
    logging.info(f"Processing {total_samples} samples with {num_gpus} workers...")
    
    # Wait for all workers to complete - no polling, workers push updates
    results = ray.get(futures)
    
    # Close progress bar
    ray.get(progress_actor.close.remote())
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    
    total_samples_processed = sum(r['samples_processed'] for r in results)
    total_tokens = sum(r['tokens_generated'] for r in results)
    total_errors = sum(r['errors'] for r in results)
    max_time = max(r['elapsed_time'] for r in results)
    avg_tokens_per_image = total_tokens / total_samples_processed if total_samples_processed > 0 else 0
    
    for i, r in enumerate(results):
        worker_avg = r['tokens_generated'] / r['samples_processed'] if r['samples_processed'] > 0 else 0
        print(f"Worker {i}: {r['samples_processed']} samples, "
              f"{r['tokens_generated']} tokens ({worker_avg:.1f} avg/img), "
              f"{r['throughput']:.0f} tok/s")
    
    print("-"*60)
    print(f"Total: {total_samples_processed} samples, {total_tokens} tokens")
    print(f"Average tokens per image: {avg_tokens_per_image:.1f}")
    print(f"Errors: {total_errors}")
    print(f"Time: {max_time:.1f}s")
    print(f"Overall throughput: {total_tokens/max_time:.0f} tokens/sec")
    print("="*60)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "dataset_info.json")
    metadata = {
        'dataset_name': args.dataset_name,
        'config_name': args.config_name,
        'split': args.split,
        'total_samples': total_samples_processed,
        'total_tokens': total_tokens,
        'num_workers': num_gpus,
        'processing_time': max_time,
        'errors': total_errors,
        'tokenizer_path': args.tokenizer_path
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Saved metadata to {metadata_path}")
    
    ray.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize HuggingFace image datasets with EMU3"
    )
    
    # Dataset arguments
    parser.add_argument("--dataset-name", required=True,
                       help="HuggingFace dataset name (e.g., 'HuggingFaceM4/FineVision')")
    parser.add_argument("--config-name", required=True,
                       help="Dataset configuration (e.g., 'CoSyn_400k_chart')")
    parser.add_argument("--split", default="train",
                       help="Dataset split to process")
    
    # Output arguments
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for tokenized data")
    parser.add_argument("--tokenizer-path", required=True,
                       help="Path to EMU3 tokenizer")
    
    # Processing arguments
    parser.add_argument("--num-gpus", type=int,
                       help="Number of GPUs for distributed processing (0 for single GPU)")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Samples per batch for distributed processing")
    parser.add_argument("--max-samples", type=int,
                       help="Maximum samples to process (for testing)")
    parser.add_argument("--cache-dir", type=str,
                       help="Cache directory for downloaded datasets")
    
    # Vision tokenizer parameters
    parser.add_argument("--min-pixels", type=int, default=None,
                       help="Minimum pixels for image preprocessing (default: 512*512)")
    parser.add_argument("--max-pixels", type=int, default=None,
                       help="Maximum pixels for image preprocessing (default: 1024*1024)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Always use distributed mode
    process_dataset_distributed(args)


if __name__ == "__main__":
    main()