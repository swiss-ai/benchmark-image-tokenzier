#!/usr/bin/env python3
"""
HuggingFace dataset tokenization for EMU3 SFT data (image-text pairs).
Reuses components from tokenize_hf_datasets.py for maximum code reuse.
"""

import argparse
#!/usr/bin/env python3
"""
HuggingFace dataset tokenization for EMU3 SFT data (image-text pairs).
Optimized for single-sample processing due to arbitrary image shapes.
"""

import argparse
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Dict

import ray
import torch
from datasets import load_dataset

# Add paths for imports
import sys
base_dir = Path(__file__).parent.parent
sys.path.append(str(base_dir))
sys.path.append(str(base_dir / 'Tokenizer'))

# Import shared components
from tokenize_hf_datasets import ProgressActor, WorkQueue
from pipelines.indexed_dataset_megatron import DType, IndexedDatasetBuilder
from utils.parse_utils import parse_resolution
from vokenizers.emu3 import EMU3SftTokenizer


@ray.remote(num_gpus=1)
class DynamicSFTTokenizerWorker:
    """GPU worker that processes image-text pairs dynamically from queue."""

    def __init__(self, tokenizer_path: str, output_dir: str, worker_id: int,
                 min_tokenizer_res: int = None, max_tokenizer_res: int = None,
                 min_image_res: int = None, max_image_res: int = None,
                 image_field: str = "image", text_field: str = "text"):
        self.worker_id = worker_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_field = image_field
        self.text_field = text_field
        self.min_image_res = min_image_res
        self.max_image_res = max_image_res

        # Setup logging
        self.logger = logging.getLogger(f"Worker{worker_id:02d}")
        self.logger.setLevel(logging.INFO)

        # Initialize tokenizer with custom pixel limits
        # Use tokenizer resolution settings for the tokenizer itself
        self.tokenizer = EMU3SftTokenizer(
            text_tokenizer_path=tokenizer_path,
            device=self.device,
            min_pixels=min_tokenizer_res,
            max_pixels=max_tokenizer_res
        )

        # Cache frequently used token IDs
        self.img_end_id = self.tokenizer.text_tokenizer.convert_tokens_to_ids("<|img_end|>")

        # Setup output
        os.makedirs(output_dir, exist_ok=True)
        self.output_path = os.path.join(output_dir, f"rank_{worker_id:03d}")
        self.builder = IndexedDatasetBuilder(
            f"{self.output_path}.bin",
            dtype=DType.optimal_dtype(self.tokenizer.text_tokenizer.vocab_size)
        )

        # Initialize prefetch queue for async loading
        # Note: Since batch_size=1 for arbitrary shapes, we optimize for single-sample throughput
        self.sample_queue = queue.Queue(maxsize=32)

        self.stats = {
            'batches_processed': 0,
            'samples_processed': 0,
            'tokens_generated': 0,
            'image_tokens': 0,
            'text_tokens': 0,
            'errors': 0,
            'samples_skipped': 0,
            'resolution_skipped': 0,
            'start_time': time.time()
        }

        self.logger.info(f"Worker {worker_id} initialized on {self.device}")

    def _extract_image(self, sample: Dict):
        """Extract image from sample, handling various formats."""
        # Try the configured field first
        if self.image_field in sample:
            value = sample[self.image_field]
            # Handle single image in list
            return value[0] if isinstance(value, list) and len(value) == 1 else value

        # Fallback to common field names
        for key in ['image', 'img', 'images']:
            if key not in sample:
                continue
            value = sample[key]
            # Handle single image in list
            return value[0] if isinstance(value, list) and len(value) == 1 else value
        return None

    def _should_process_resolution(self, image):
        """Check if image meets resolution criteria for filtering.

        Note: image.size is O(1) - just returns stored dimensions without decoding.
        """
        if not self.min_image_res and not self.max_image_res:
            return True

        width, height = image.size
        resolution = width * height

        # Filter based on configured thresholds
        if self.min_image_res and resolution < self.min_image_res:
            return False
        if self.max_image_res and resolution > self.max_image_res:
            return False

        return True

    def _extract_text(self, sample: Dict):
        """Extract text from sample, handling FineVision and other formats."""
        # FineVision format: 'texts' field contains conversation
        if 'texts' in sample:
            return sample['texts']  # List of {"user": ..., "assistant": ...} dicts

        # Standard format: use configured field
        if self.text_field in sample:
            return sample[self.text_field]

        return None

    def _load_samples_async(self, indices, dataset_info):
        """Background thread to load image-text pairs into queue."""
        try:
            # Load dataset slice
            samples = load_dataset(
                dataset_info['name'],
                name=dataset_info['config'],
                split=f"{dataset_info['split']}[{indices[0]}:{indices[-1]+1}]",
                cache_dir=dataset_info.get('cache_dir')
            )

            # Queue image-text pairs for processing
            for sample in samples:
                image = self._extract_image(sample)
                text = self._extract_text(sample)

                if image is not None and text:
                    # Check resolution filtering
                    if self._should_process_resolution(image):
                        self.sample_queue.put((image, text, 'ok'))
                    else:
                        # Resolution filtered
                        self.sample_queue.put((None, None, 'resolution_skip'))
                else:
                    # Track skipped samples (missing data)
                    self.sample_queue.put((None, None, 'data_skip'))
        except Exception as e:
            self.logger.error(f"Loader thread error: {e}")
        finally:
            self.sample_queue.put((None, None, 'sentinel'))  # Sentinel

    def _process_sample_stream(self):
        """Process image-text pairs from queue as GPU becomes available."""
        stats = {'samples': 0, 'tokens': 0, 'image_tokens': 0, 'text_tokens': 0, 'errors': 0, 'skipped': 0, 'resolution_skipped': 0}

        while True:
            item = self.sample_queue.get()
            if len(item) == 3:
                image, text, status = item
            else:
                # Handle old format for compatibility
                image, text = item
                status = 'ok' if image is not None else 'data_skip'

            if status == 'sentinel':
                if self.sample_queue.empty():
                    break  # Sentinel
                else:
                    continue
            elif status == 'resolution_skip':
                stats['resolution_skipped'] += 1
                continue
            elif status == 'data_skip':
                stats['skipped'] += 1
                continue

            try:
                # Process with EMU3 tokenizer (handles arbitrary image shapes)
                # Note: Processing single samples due to variable image dimensions
                tokens = self.tokenizer.process_finevision_sample(text, image)

                # Efficiently find token boundaries
                if torch.is_tensor(tokens):
                    # Tensor operations are faster for finding token boundaries
                    img_end_mask = (tokens == self.img_end_id)
                    if img_end_mask.any():
                        img_end_idx = img_end_mask.nonzero(as_tuple=True)[0][0].item()
                        image_token_count = img_end_idx + 1
                        text_token_count = len(tokens) - image_token_count
                    else:
                        image_token_count = 0
                        text_token_count = len(tokens)
                    tokens_np = tokens.cpu().numpy()
                else:
                    # Handle numpy arrays
                    tokens_np = tokens
                    tokens_list = tokens_np.tolist()
                    if self.img_end_id in tokens_list:
                        img_end_idx = tokens_list.index(self.img_end_id)
                        image_token_count = img_end_idx + 1
                        text_token_count = len(tokens_list) - image_token_count
                    else:
                        image_token_count = 0
                        text_token_count = len(tokens_list)

                # Save results
                self.builder.add_document(tokens_np, [len(tokens_np)])

                stats['samples'] += 1
                stats['tokens'] += len(tokens_np)
                stats['image_tokens'] += image_token_count
                stats['text_tokens'] += text_token_count
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
            target=self._load_samples_async,
            args=(indices, dataset_info),
            daemon=True
        )
        loader.start()

        # Process samples as they arrive
        stats = self._process_sample_stream()

        # Ensure loader completes
        loader.join()

        # Update global stats
        self.stats['batches_processed'] += 1
        self.stats['samples_processed'] += stats['samples']
        self.stats['tokens_generated'] += stats['tokens']
        self.stats['image_tokens'] += stats['image_tokens']
        self.stats['text_tokens'] += stats['text_tokens']
        self.stats['errors'] += stats['errors']
        self.stats['samples_skipped'] += stats['skipped']
        self.stats['resolution_skipped'] += stats.get('resolution_skipped', 0)

        # Log completion with token distribution
        elapsed = time.time() - start_time
        if stats['samples'] > 0:
            avg_img = stats['image_tokens'] / stats['samples']
            avg_txt = stats['text_tokens'] / stats['samples']
            throughput = stats['samples'] / elapsed if elapsed > 0 else 0
            self.logger.info(
                f"Completed {batch_id}: {stats['samples']} samples, "
                f"{stats['tokens']} tokens (img: {avg_img:.1f}, txt: {avg_txt:.1f}), "
                f"{throughput:.1f} samples/s"
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


def process_dataset_distributed(args):
    """Main distributed processing with dynamic work-stealing for SFT data."""

    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    resources = ray.available_resources()
    num_gpus = args.num_gpus or int(resources.get('GPU', 1))

    logging.info(f"Starting SFT data processing with {num_gpus} GPUs")

    # Auto-create output directory with config name and SFT suffix
    output_dir = os.path.join(args.output_dir, f"{args.config_name}_sft")
    logging.info(f"Output directory: {output_dir}")

    # Load dataset
    logging.info(f"Loading dataset {args.dataset_name}/{args.config_name}...")
    dataset = load_dataset(
        args.dataset_name,
        name=args.config_name,
        split=args.split,
        cache_dir=args.cache_dir,
        num_proc=32
    )
    total_samples = len(dataset)

    if args.max_samples:
        total_samples = min(total_samples, args.max_samples)
        dataset = dataset.select(range(total_samples))

    logging.info(f"Dataset loaded: {total_samples} samples")

    # Create dataset info dict
    dataset_info = {
        'name': args.dataset_name,
        'config': args.config_name,
        'split': args.split,
        'cache_dir': args.cache_dir,
        'total_samples': total_samples
    }

    # Create work queue (reused from original)
    work_queue = WorkQueue.remote(total_samples, args.batch_size)

    # Create workers
    workers = []
    for i in range(num_gpus):
        worker = DynamicSFTTokenizerWorker.remote(
            tokenizer_path=args.tokenizer_path,
            output_dir=output_dir,
            worker_id=i,
            min_tokenizer_res=args.min_tokenizer_res,
            max_tokenizer_res=args.max_tokenizer_res,
            min_image_res=args.min_image_res,
            max_image_res=args.max_image_res,
            image_field=args.image_field,
            text_field=args.text_field
        )
        workers.append(worker)

    # Create progress actor (reused from original)
    progress_actor = ProgressActor.remote(total_samples)

    # Start all workers
    futures = [worker.run.remote(work_queue, dataset_info, progress_actor)
               for worker in workers]

    logging.info(f"Processing {total_samples} samples with {num_gpus} workers...")

    # Wait for completion
    results = ray.get(futures)

    # Close progress bar
    ray.get(progress_actor.close.remote())

    # Print summary
    print_sft_summary(results, output_dir, args, total_samples)

    ray.shutdown()


def print_sft_summary(results, output_dir, args, _total_samples):
    """Print summary statistics for SFT data processing."""
    print("\n" + "="*70)
    print("SFT DATA TOKENIZATION COMPLETE")
    print("="*70)

    # Calculate totals
    total_samples_processed = sum(r['samples_processed'] for r in results)
    total_tokens = sum(r['tokens_generated'] for r in results)
    total_image_tokens = sum(r.get('image_tokens', 0) for r in results)
    total_text_tokens = sum(r.get('text_tokens', 0) for r in results)
    total_errors = sum(r['errors'] for r in results)
    total_skipped = sum(r.get('samples_skipped', 0) for r in results)
    total_resolution_skipped = sum(r.get('resolution_skipped', 0) for r in results)
    max_time = max(r['elapsed_time'] for r in results)

    # Per-worker stats
    print("\nPer-Worker Statistics:")
    print("-"*70)
    for i, r in enumerate(results):
        if r['samples_processed'] > 0:
            worker_avg_img = r.get('image_tokens', 0) / r['samples_processed']
            worker_avg_txt = r.get('text_tokens', 0) / r['samples_processed']
            print(f"Worker {i:2d}: {r['samples_processed']:6d} samples | "
                  f"{r['tokens_generated']:10,d} tokens | "
                  f"img: {worker_avg_img:6.1f} | txt: {worker_avg_txt:6.1f} | "
                  f"{r['throughput']/1000:6.1f}K tok/s")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total samples processed: {total_samples_processed:10,d}")
    print(f"Total samples skipped:   {total_skipped:10,d}")
    if total_resolution_skipped > 0:
        print(f"Resolution filtered:     {total_resolution_skipped:10,d}")
    print(f"Total tokens generated:  {total_tokens:10,d}")

    if total_samples_processed > 0:
        # Token distribution
        print(f"\nToken Distribution:")
        print(f"  Image tokens: {total_image_tokens:12,d} ({total_image_tokens/total_tokens*100:5.1f}%)")
        print(f"  Text tokens:  {total_text_tokens:12,d} ({total_text_tokens/total_tokens*100:5.1f}%)")

        # Averages
        print(f"\nAverage Tokens per Sample:")
        print(f"  Total: {total_tokens/total_samples_processed:8.1f}")
        print(f"  Image: {total_image_tokens/total_samples_processed:8.1f}")
        print(f"  Text:  {total_text_tokens/total_samples_processed:8.1f}")

    # Performance
    print(f"\nPerformance:")
    print(f"  Processing time: {max_time:8.1f}s")
    if max_time > 0:
        print(f"  Samples/sec:     {total_samples_processed/max_time:8.1f}")
        print(f"  Tokens/sec:      {total_tokens/max_time:8.0f}")

    if total_errors > 0:
        print(f"\n⚠ Errors: {total_errors}")

    print("="*70)

    # Save metadata
    save_sft_metadata(results, output_dir, args, total_samples_processed, total_skipped,
                     total_tokens, total_image_tokens, total_text_tokens, total_errors, max_time,
                     total_resolution_skipped)


def save_sft_metadata(results, output_dir, args, total_samples_processed, total_skipped,
                      total_tokens, total_image_tokens, total_text_tokens, total_errors, max_time,
                      total_resolution_skipped=0):
    """Save metadata for SFT dataset processing."""
    metadata_path = os.path.join(output_dir, "dataset_info.json")
    metadata = {
        'dataset_type': 'SFT (image-text conversations)',
        'dataset_name': args.dataset_name,
        'config_name': args.config_name,
        'split': args.split,
        'image_field': args.image_field,
        'text_field': args.text_field,
        'statistics': {
            'total_samples_processed': total_samples_processed,
            'samples_skipped': total_skipped,
            'resolution_skipped': total_resolution_skipped,
            'total_tokens': total_tokens,
            'image_tokens': total_image_tokens,
            'text_tokens': total_text_tokens,
            'errors': total_errors
        },
        'token_distribution': {
            'image_percentage': total_image_tokens / total_tokens * 100 if total_tokens > 0 else 0,
            'text_percentage': total_text_tokens / total_tokens * 100 if total_tokens > 0 else 0
        },
        'averages': {
            'tokens_per_sample': total_tokens / total_samples_processed if total_samples_processed > 0 else 0,
            'image_tokens_per_sample': total_image_tokens / total_samples_processed if total_samples_processed > 0 else 0,
            'text_tokens_per_sample': total_text_tokens / total_samples_processed if total_samples_processed > 0 else 0
        },
        'processing': {
            'num_workers': len(results),
            'processing_time_seconds': max_time,
            'samples_per_second': total_samples_processed / max_time if max_time > 0 else 0,
            'tokens_per_second': total_tokens / max_time if max_time > 0 else 0
        },
        'tokenizer': {
            'path': args.tokenizer_path,
            'min_tokenizer_res': args.min_tokenizer_res,
            'max_tokenizer_res': args.max_tokenizer_res,
            'min_tokenizer_res_str': f"{args.min_tokenizer_res} pixels" if args.min_tokenizer_res else None,
            'max_tokenizer_res_str': f"{args.max_tokenizer_res} pixels" if args.max_tokenizer_res else None
        },
        'image_filtering': {
            'min_image_res': args.min_image_res,
            'max_image_res': args.max_image_res,
            'min_image_res_str': f"{args.min_image_res} pixels" if args.min_image_res else None,
            'max_image_res_str': f"{args.max_image_res} pixels" if args.max_image_res else None
        },
        'worker_details': [
            {
                'worker_id': i,
                'samples_processed': r['samples_processed'],
                'tokens_generated': r['tokens_generated'],
                'image_tokens': r.get('image_tokens', 0),
                'text_tokens': r.get('text_tokens', 0),
                'errors': r['errors'],
                'throughput': r['throughput']
            } for i, r in enumerate(results)
        ]
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize HuggingFace SFT datasets with EMU3 for image-text conversations"
    )

    # Dataset arguments
    parser.add_argument("--dataset-name", required=True,
                       help="HuggingFace dataset name")
    parser.add_argument("--config-name", required=True,
                       help="Dataset configuration")
    parser.add_argument("--split", default="train",
                       help="Dataset split to process")

    # Field names
    parser.add_argument("--image-field", default="image",
                       help="Field name for images (default: 'image')")
    parser.add_argument("--text-field", default="text",
                       help="Field name for text/captions (default: 'text')")

    # Output arguments
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for tokenized data")
    parser.add_argument("--tokenizer-path", required=True,
                       help="Path to EMU3 tokenizer")

    # Processing arguments
    parser.add_argument("--num-gpus", type=int,
                       help="Number of GPUs for distributed processing")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Samples per batch")
    parser.add_argument("--max-samples", type=int,
                       help="Maximum samples to process (for testing)")
    parser.add_argument("--cache-dir", type=str,
                       help="Cache directory for downloaded datasets")

    # Image resolution filtering parameters (which images to process)
    parser.add_argument("--min-image-res", type=parse_resolution, default=None,
                       help="Minimum image resolution to process (e.g., '256*256')")
    parser.add_argument("--max-image-res", type=parse_resolution, default=None,
                       help="Maximum image resolution to process (e.g., '2048*2048')")

    # Tokenizer resolution parameters (how to tokenize the images)
    parser.add_argument("--min-tokenizer-res", type=parse_resolution, default=None,
                       help="Minimum resolution for tokenizer (default: same as --min-image-res)")
    parser.add_argument("--max-tokenizer-res", type=parse_resolution, default=parse_resolution('1024*1024'),
                       help="Maximum resolution for tokenizer (default: 1024*1024)")

    args = parser.parse_args()

    # If min-tokenizer-res not specified, use min-image-res
    if args.min_tokenizer_res is None:
        args.min_tokenizer_res = args.min_image_res

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Process dataset
    process_dataset_distributed(args)


if __name__ == "__main__":
    main()