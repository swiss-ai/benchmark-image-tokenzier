#!/usr/bin/env python3
"""
Ray-based EMU3 image-text tokenization using tokenize_image_text_pair.
Uses ThreadPoolExecutor-based parallel tokenization (simpler than pipelined).
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import ray
import torch
import webdataset as wds

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "utils"))

# Import existing components
from webdataset_emu3_ray_dynamic import ShardQueue


# ============================================================================
# GPU Worker with Image-Text Pair Tokenization
# ============================================================================

@ray.remote(num_gpus=1)
class EMU3ImageTextWorker:
    """Worker that processes image-text pairs using tokenize_image_text_pair."""

    def __init__(self, config: Dict, worker_id: int):
        # Fix imports for Ray workers
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        sys.path.append(str(Path(__file__).parent.parent / "utils"))

        # Now import after path is set
        from utils.tokenization_emu3_image_only import EMU3ImageTextPairTokenizer
        from indexed_dataset_megatron import DType, IndexedDatasetBuilder

        self.config = config
        self.worker_id = worker_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Setup logging
        self.logger = logging.getLogger(f"Worker{worker_id:02d}")
        self.logger.setLevel(logging.INFO)

        # Initialize tokenizer with ThreadPoolExecutor for parallel processing
        self.tokenizer = EMU3ImageTextPairTokenizer(
            text_tokenizer_path=config['tokenizer_path'],
            device=self.device
        )

        # Initialize builder
        output_dir = config['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        self.output_path = os.path.join(output_dir, f"rank_{worker_id:03d}")
        self.builder = IndexedDatasetBuilder(
            f"{self.output_path}.bin",
            dtype=DType.optimal_dtype(self.tokenizer.text_tokenizer.vocab_size)
        )

        # Statistics
        self.stats = {
            'shards_processed': 0,
            'samples_processed': 0,
            'image_tokens': 0,
            'text_tokens': 0,
            'total_tokens': 0,
            'errors': 0,
            'start_time': time.time()
        }

        self.logger.info(f"Worker {worker_id} initialized on {self.device}")
        self.logger.info(f"Using tokenize_image_text_pair with ThreadPoolExecutor")

    def process_shard(self, shard_path: str) -> Dict[str, Any]:
        """Process a single shard using tokenize_image_text_pair."""
        shard_name = Path(shard_path).name
        self.logger.info(f"Starting {shard_name}")

        start_time = time.time()
        samples = 0
        total_tokens = 0
        skipped = 0

        try:
            # Create dataset for the shard
            dataset = (
                wds.WebDataset(shard_path, shardshuffle=False)
                .decode("pil")
                .to_tuple("jpg;png;jpeg;webp", "txt", "__key__")
                .batched(64)  # Pre-batch data
            )

            # Process each sample
            for img, text_data, key in dataset:
                # Text comes as string from .txt files
                text = text_data.strip() if text_data else ""

                if not text:
                    self.logger.debug(f"Skipping {key}: no text found")
                    skipped += 1
                    continue

                try:
                    # Tokenize image-text pair
                    tokens = self.tokenizer.tokenize_image_text_pair(img, text)
                    total_token_count = len(tokens)

                    # Estimate token counts based on structure
                    # We know the structure: [BOS] [img_start] [dims] [img_token_start] [vision_tokens+EOLs] [EOF] [img_end] [text] [EOS]
                    # Find img_end token position to separate image from text
                    tokens_list = tokens.tolist() if torch.is_tensor(tokens) else list(tokens)
                    img_end_id = self.tokenizer.text_tokenizer.convert_tokens_to_ids("<|img_end|>")

                    if img_end_id in tokens_list:
                        img_end_idx = tokens_list.index(img_end_id)
                        # Image tokens: from BOS to img_end (inclusive)
                        image_token_count = img_end_idx + 1
                        # Text tokens: from img_end+1 to EOS
                        text_token_count = total_token_count - image_token_count
                    else:
                        # Fallback: estimate based on typical sizes
                        image_token_count = total_token_count // 2  # rough estimate
                        text_token_count = total_token_count - image_token_count

                    # Convert to numpy and add to dataset
                    tokens_np = tokens.cpu().numpy() if torch.is_tensor(tokens) else tokens
                    self.builder.add_document(tokens_np, [len(tokens_np)])

                    samples += 1
                    total_tokens += total_token_count
                    self.stats['image_tokens'] += image_token_count
                    self.stats['text_tokens'] += text_token_count

                    # Log progress every 100 samples
                    if samples % 100 == 0:
                        elapsed = time.time() - start_time
                        self.logger.info(
                            f"  {shard_name}: {samples} samples processed, "
                            f"{samples/elapsed:.1f} samples/s"
                        )

                except Exception as e:
                    self.logger.error(f"Error processing {key}: {e}")
                    self.stats['errors'] += 1
                    continue

            # Update stats
            self.stats['shards_processed'] += 1
            self.stats['samples_processed'] += samples
            self.stats['total_tokens'] += total_tokens

            elapsed = time.time() - start_time
            self.logger.info(
                f"Completed {shard_name}: {samples} samples, {total_tokens} tokens in {elapsed:.1f}s "
                f"({skipped} skipped)"
            )

            return {
                'success': True,
                'shard': shard_name,
                'samples': samples,
                'tokens': total_tokens,
                'skipped': skipped,
                'time': elapsed
            }

        except Exception as e:
            self.logger.error(f"Failed to process {shard_name}: {e}")
            self.stats['errors'] += 1
            return {
                'success': False,
                'shard': shard_name,
                'error': str(e)
            }

    def _extract_text(self, metadata: Dict) -> str:
        """Extract text from metadata, checking common keys."""
        # Try common text keys in order of preference
        for key in ['caption', 'text', 'description', 'alt_text', 'title']:
            if key in metadata and metadata[key]:
                return str(metadata[key])
        return ""

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
                ray.get(shard_queue.mark_failed.remote(
                    shard_path,
                    result.get('error', 'Unknown')
                ))

        # No pipeline to stop with tokenize_image_text_pair

        # Finalize dataset
        self.builder.finalize(f"{self.output_path}.idx")

        # Return final statistics
        elapsed = time.time() - self.stats['start_time']
        self.stats['elapsed_time'] = elapsed
        self.stats['throughput'] = self.stats['total_tokens'] / elapsed if elapsed > 0 else 0

        return self.stats


# ============================================================================
# Modified Main Pipeline
# ============================================================================

def process_image_text_pairs(config: Dict):
    """Process image-text pairs using tokenize_image_text_pair (ThreadPoolExecutor parallel)."""

    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    resources = ray.available_resources()
    num_gpus = config.get('num_gpus') or int(resources.get('GPU', 1))

    logging.info(f"Starting with {num_gpus} GPU workers")
    logging.info(f"Each worker uses ThreadPoolExecutor for CPU-GPU parallel processing")

    # Get list of shards
    import glob
    input_files = sorted(glob.glob(config['input_pattern']))
    if not input_files:
        logging.error(f"No files found: {config['input_pattern']}")
        return

    # Apply range filter if specified
    if config.get('range'):
        range_str = config['range']
        try:
            if ':' in range_str:
                parts = range_str.split(':')
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if parts[1] else len(input_files)
            else:
                # Single number means process only that index
                start = int(range_str)
                end = start + 1

            # Validate multiples of 4
            assert (end - start) % 4 == 0, f"Range size {end - start} must be a multiple of 4"

            # Apply range
            original_count = len(input_files)
            input_files = input_files[start:end]
            logging.info(f"Range filter '{range_str}': selected {len(input_files)} out of {original_count} shards")

            if not input_files:
                logging.error(f"Range '{range_str}' resulted in no files to process")
                return

            logging.info(f"Processing shards from index {start} to {min(end, original_count)-1}")
            logging.info(f"  First shard: {Path(input_files[0]).name}")
            logging.info(f"  Last shard: {Path(input_files[-1]).name}")

        except (ValueError, IndexError) as e:
            logging.error(f"Invalid range format '{range_str}'. Use format like '0:4' or '8:16' or '100:'")
            return
    else:
        logging.info(f"Found {len(input_files)} shards to process")

    # Create shard queue (reuse from original)
    shard_queue = ShardQueue.remote(input_files)

    # Create workers with parallel tokenizer
    workers = []
    for i in range(num_gpus):
        worker = EMU3ImageTextWorker.remote(config, i)
        workers.append(worker)

    # Start all workers
    futures = [worker.run.remote(shard_queue) for worker in workers]

    # Monitor progress (reuse monitoring logic)
    from tqdm import tqdm
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

        time.sleep(0.1)  # Check every 100ms for more responsive updates

    pbar.close()

    # Get final results
    results = ray.get(futures)

    # Print summary
    print("\n" + "="*70)
    print("IMAGE-TEXT PARALLEL TOKENIZATION COMPLETE")
    print("="*70)

    total_shards = sum(r['shards_processed'] for r in results)
    total_samples = sum(r['samples_processed'] for r in results)
    total_tokens = sum(r['total_tokens'] for r in results)
    total_image_tokens = sum(r['image_tokens'] for r in results)
    total_text_tokens = sum(r['text_tokens'] for r in results)
    total_errors = sum(r['errors'] for r in results)
    max_time = max(r['elapsed_time'] for r in results)

    # Per-worker stats
    print("\nPer-Worker Statistics:")
    print("-"*70)
    for i, r in enumerate(results):
        print(f"Worker {i:2d}: {r['shards_processed']:3d} shards | "
              f"{r['samples_processed']:6d} samples | "
              f"{r['throughput']/1000:6.1f}K tok/s")

    # Summary stats
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"Shards:     {total_shards:8d}")
    print(f"Samples:    {total_samples:8d}")
    print(f"Tokens:     {total_tokens:8,d}")

    if total_samples > 0:
        avg_total_tokens = total_tokens / total_samples
        avg_image_tokens = total_image_tokens / total_samples
        avg_text_tokens = total_text_tokens / total_samples

        print(f"\nToken Distribution:")
        print(f"  Image:    {total_image_tokens:12,d} ({total_image_tokens/total_tokens*100:5.1f}%)")
        print(f"  Text:     {total_text_tokens:12,d} ({total_text_tokens/total_tokens*100:5.1f}%)")

        print(f"\nAverage Tokens per Sequence:")
        print(f"  Total:    {avg_total_tokens:8.1f}")
        print(f"  Image:    {avg_image_tokens:8.1f}")
        print(f"  Text:     {avg_text_tokens:8.1f}")

    print(f"\nPerformance:")
    print(f"  Time:     {max_time:8.1f}s")
    print(f"  Samples:  {total_samples/max_time:8.1f} samples/sec")
    print(f"  Tokens:   {total_tokens/max_time/1000:8.1f}K tokens/sec")

    if total_errors > 0:
        print(f"\n⚠ Errors:   {total_errors}")
    print("="*70)

    # Check for failed shards
    final_status = ray.get(shard_queue.get_status.remote())
    if final_status['failed'] > 0:
        print(f"\nWarning: {final_status['failed']} shards failed processing")

    ray.shutdown()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EMU3 image-text tokenization with parallel architecture (ThreadPoolExecutor)"
    )

    # Required arguments
    parser.add_argument("--input-pattern", required=True,
                       help="Input shard pattern (e.g., '/path/*.tar')")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for tokenized data")
    parser.add_argument("--tokenizer-path", required=True,
                       help="Path to EMU3 tokenizer")

    # Optional arguments
    parser.add_argument("--num-gpus", type=int,
                       help="Number of GPU workers (default: all available)")
    parser.add_argument("--range", type=str,
                       help="Process specific range of tar files. Must be multiples of 4. E.g., '0:4' (files 0-3), '8:16' (files 8-15), '100:104' (files 100-103)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    config = vars(args)
    process_image_text_pairs(config)


if __name__ == "__main__":
    main()