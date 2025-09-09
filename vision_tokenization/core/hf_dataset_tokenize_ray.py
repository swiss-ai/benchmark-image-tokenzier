#!/usr/bin/env python3
"""
HuggingFace dataset tokenization with Ray dynamic work-stealing.
Workers pull work as they become available for optimal GPU utilization.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import ray
import torch
from datasets import load_dataset, get_dataset_config_info
from tqdm import tqdm

# Add vision tokenization to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.indexed_dataset_megatron import DType, IndexedDatasetBuilder
from utils.tokenization_emu3_image_only import EMU3ImageOnlyTokenizer


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
    
    def __init__(self, tokenizer_path: str, output_dir: str, worker_id: int):
        self.worker_id = worker_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Setup logging
        self.logger = logging.getLogger(f"Worker{worker_id:02d}")
        self.logger.setLevel(logging.INFO)
        
        # Initialize tokenizer
        self.tokenizer = EMU3ImageOnlyTokenizer(
            text_tokenizer_path=tokenizer_path,
            device=self.device
        )
        
        # Setup output
        os.makedirs(output_dir, exist_ok=True)
        self.output_path = os.path.join(output_dir, f"rank_{worker_id:03d}")
        self.builder = IndexedDatasetBuilder(
            f"{self.output_path}.bin",
            dtype=DType.optimal_dtype(self.tokenizer.text_tokenizer.vocab_size)
        )
        
        self.stats = {
            'batches_processed': 0,
            'samples_processed': 0,
            'tokens_generated': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        self.logger.info(f"Worker {worker_id} initialized on {self.device}")
    
    def process_batch(self, batch_info: Dict, dataset_info: Dict) -> Dict:
        """Process a batch of samples from the dataset."""
        batch_id = batch_info['batch_id']
        indices = batch_info['indices']
        
        self.logger.info(f"Processing {batch_id} ({len(indices)} samples)")
        
        batch_tokens = 0
        batch_samples = 0
        batch_errors = 0
        start_time = time.time()
        
        # Load only the needed samples using HF's partial loading
        from datasets import load_dataset
        
        # Load specific indices directly - HF will use cached parquet files
        start_idx = indices[0]
        end_idx = indices[-1] + 1
        samples = load_dataset(
            dataset_info['name'],
            name=dataset_info['config'],
            split=f"{dataset_info['split']}[{start_idx}:{end_idx}]",
            cache_dir=dataset_info['cache_dir']
        )
        
        for sample in samples:
            try:
                # Get image from sample (handle different key names)
                image = None
                for key in ['image', 'img', 'images']:
                    if key in sample:
                        value = sample[key]
                        # Handle single image in a list (common in some datasets)
                        if isinstance(value, list) and len(value) == 1:
                            image = value[0]
                        else:
                            image = value
                        break
                
                if image is None:
                    self.logger.warning(f"No image found in sample")
                    batch_errors += 1
                    continue
                
                # Tokenize image
                img_tokens = self.tokenizer.tokenize_image(image)
                tokens_np = img_tokens.cpu().numpy() if torch.is_tensor(img_tokens) else img_tokens
                
                # Add to indexed dataset
                self.builder.add_document(tokens_np, [len(tokens_np)])
                
                batch_samples += 1
                batch_tokens += len(tokens_np)
                
            except Exception as e:
                self.logger.warning(f"Error processing sample: {e}")
                batch_errors += 1
        
        # Update stats
        self.stats['batches_processed'] += 1
        self.stats['samples_processed'] += batch_samples
        self.stats['tokens_generated'] += batch_tokens
        self.stats['errors'] += batch_errors
        
        elapsed = time.time() - start_time
        self.logger.info(f"Completed {batch_id}: {batch_samples} samples, {batch_tokens} tokens in {elapsed:.1f}s")
        
        return {
            'batch_id': batch_id,
            'samples': batch_samples,
            'tokens': batch_tokens,
            'errors': batch_errors,
            'time': elapsed
        }
    
    def run(self, work_queue, dataset_info) -> Dict:
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
    
    # Load dataset (non-streaming for better performance)
    logging.info(f"Loading dataset {args.dataset_name}/{args.config_name}...")
    dataset = load_dataset(
        args.dataset_name,
        name=args.config_name,
        split=args.split,
        cache_dir=args.cache_dir  # Optional cache directory
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
    
    # Create workers
    workers = []
    for i in range(num_gpus):
        worker = DynamicTokenizerWorker.remote(
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
            worker_id=i
        )
        workers.append(worker)
    
    # Start all workers (they'll pull work dynamically)
    futures = [worker.run.remote(work_queue, dataset_info) for worker in workers]
    
    # Monitor progress
    pbar = tqdm(total=total_samples, desc="Samples processed")
    last_processed = 0
    
    while True:
        # Check queue status
        status = ray.get(work_queue.get_status.remote())
        
        # Update progress bar
        processed = status['processed']
        if processed > last_processed:
            pbar.update(processed - last_processed)
            last_processed = processed
        
        # Check if done
        if status['processed'] >= total_samples and status['in_progress'] == 0:
            break
        
        time.sleep(1)  # Check every second
    
    pbar.close()
    
    # Get results from all workers
    results = ray.get(futures)
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    
    total_samples_processed = sum(r['samples_processed'] for r in results)
    total_tokens = sum(r['tokens_generated'] for r in results)
    total_errors = sum(r['errors'] for r in results)
    max_time = max(r['elapsed_time'] for r in results)
    
    for i, r in enumerate(results):
        print(f"Worker {i}: {r['samples_processed']} samples, "
              f"{r['tokens_generated']} tokens, "
              f"{r['throughput']:.0f} tok/s")
    
    print("-"*60)
    print(f"Total: {total_samples_processed} samples, {total_tokens} tokens")
    print(f"Errors: {total_errors}")
    print(f"Time: {max_time:.1f}s")
    print(f"Overall throughput: {total_tokens/max_time:.0f} tokens/sec")
    print("="*60)
    
    # Save metadata
    metadata_path = os.path.join(args.output_dir, "dataset_info.json")
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


def process_dataset_single(args):
    """Single GPU processing (no Ray)."""
    
    logging.info("Starting single GPU processing")
    
    # Load dataset (non-streaming)
    logging.info(f"Loading dataset {args.dataset_name}/{args.config_name}...")
    dataset = load_dataset(
        args.dataset_name,
        name=args.config_name,
        split=args.split,
        cache_dir=args.cache_dir
    )
    total_samples = len(dataset)
    
    if args.max_samples:
        total_samples = min(total_samples, args.max_samples)
        dataset = dataset.select(range(total_samples))
    
    logging.info(f"Dataset loaded: {total_samples} samples")
    
    # Initialize tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = EMU3ImageOnlyTokenizer(
        text_tokenizer_path=args.tokenizer_path,
        device=device
    )
    
    # Setup output
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "rank_000")
    builder = IndexedDatasetBuilder(
        f"{output_path}.bin",
        dtype=DType.optimal_dtype(tokenizer.text_tokenizer.vocab_size)
    )
    
    # Process samples
    samples_processed = 0
    tokens_generated = 0
    errors = 0
    start_time = time.time()
    
    pbar = tqdm(dataset, total=total_samples, desc="Processing samples")
    
    for sample in pbar:
        if samples_processed >= total_samples:
            break
        
        try:
            # Get image
            image = None
            for key in ['image', 'img', 'images']:
                if key in sample:
                    image = sample[key]
                    break
            
            if image is None:
                errors += 1
                continue
            
            # Tokenize
            img_tokens = tokenizer.tokenize_image(image)
            tokens_np = img_tokens.cpu().numpy() if torch.is_tensor(img_tokens) else img_tokens
            
            # Add to dataset
            builder.add_document(tokens_np, [len(tokens_np)])
            
            samples_processed += 1
            tokens_generated += len(tokens_np)
            
            if samples_processed % 100 == 0:
                pbar.set_postfix({
                    'tokens': tokens_generated,
                    'errors': errors
                })
            
        except Exception as e:
            logging.warning(f"Error processing sample: {e}")
            errors += 1
    
    pbar.close()
    
    # Finalize
    builder.finalize(f"{output_path}.idx")
    
    elapsed = time.time() - start_time
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Samples processed: {samples_processed}")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Throughput: {tokens_generated/elapsed:.0f} tokens/sec")
    print("="*60)
    
    # Save metadata
    metadata_path = os.path.join(args.output_dir, "dataset_info.json")
    metadata = {
        'dataset_name': args.dataset_name,
        'config_name': args.config_name,
        'split': args.split,
        'total_samples': samples_processed,
        'total_tokens': tokens_generated,
        'processing_time': elapsed,
        'errors': errors,
        'tokenizer_path': args.tokenizer_path
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


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
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Samples per batch for distributed processing")
    parser.add_argument("--max-samples", type=int,
                       help="Maximum samples to process (for testing)")
    parser.add_argument("--cache-dir", type=str,
                       help="Cache directory for downloaded datasets")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Choose processing mode
    if args.num_gpus and args.num_gpus > 1:
        process_dataset_distributed(args)
    else:
        process_dataset_single(args)


if __name__ == "__main__":
    main()