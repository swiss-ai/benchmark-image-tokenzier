#!/usr/bin/env python3
"""
Ray-based EMU3 image-text tokenization using parallel CPU-GPU processing.
Each Ray worker gets 1 GPU and uses ThreadPoolExecutor for efficient tokenization.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import ray
import torch
import webdataset as wds
from tqdm import tqdm

# Setup paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from utils.parse_utils import add_emu3_tokenization_args

# Import shared components
from webdataset_emu3_ray_dynamic_clean import ShardQueue

# ============================================================================
# GPU Worker for Image-Text Tokenization
# ============================================================================


@ray.remote(num_gpus=1)
class EMU3ImageTextWorker:
    """Worker that processes image-text pairs using parallel tokenization."""

    def __init__(self, config: Dict, worker_id: int):
        """Initialize worker with tokenizer and output builder."""
        # Setup imports for Ray worker
        self._setup_imports()

        # Initialize components
        self.config = config
        self.worker_id = worker_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Setup logging
        self._setup_logging()

        # Initialize tokenizer and builder
        self._initialize_tokenizer()
        self._initialize_builder()

        # Initialize statistics
        self._reset_stats()

        self.logger.info(f"Worker {worker_id} initialized on {self.device}")

    def _setup_imports(self):
        """Setup Python paths and imports for Ray worker."""
        import sys
        from pathlib import Path

        # Add paths for local imports
        sys.path.append(str(Path(__file__).parent.parent))
        sys.path.append(str(Path(__file__).parent.parent / "utils"))

    def _setup_logging(self):
        """Configure logging for this worker."""
        self.logger = logging.getLogger(f"Worker{self.worker_id:02d}")
        self.logger.setLevel(logging.INFO)

    def _initialize_tokenizer(self):
        """Initialize the EMU3 tokenizer."""
        # Import here after paths are set
        from utils.tokenization_emu3_image_only import EMU3ImageTextPairTokenizer

        self.tokenizer = EMU3ImageTextPairTokenizer(
            text_tokenizer_path=self.config["tokenizer_path"], device=self.device
        )

        # Cache frequently used token IDs
        self.img_end_id = self.tokenizer.text_tokenizer.convert_tokens_to_ids("<|img_end|>")

    def _initialize_builder(self):
        """Initialize builder setup - actual builders created per shard."""
        # Import here after paths are set
        from indexed_dataset_megatron import DType, IndexedDatasetBuilder

        # Store imports for later use
        self.DType = DType
        self.IndexedDatasetBuilder = IndexedDatasetBuilder

        # Create output directory
        self.output_dir = self.config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        # Store dtype for builders
        self.dtype = DType.optimal_dtype(self.tokenizer.text_tokenizer.vocab_size)

    def _reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            "shards_processed": 0,
            "samples_processed": 0,
            "samples_skipped": 0,
            "total_tokens": 0,
            "image_tokens": 0,
            "text_tokens": 0,
            "errors": 0,
            "start_time": time.time(),
        }

    def process_shard(self, shard_path: str) -> Dict[str, Any]:
        """
        Process a single shard file.

        Args:
            shard_path: Path to the tar shard

        Returns:
            Processing results and statistics
        """
        shard_name = Path(shard_path).name
        if self.worker_id == 0:
            self.logger.info(f"Processing {shard_name}")

        # Track shard-level metrics
        shard_start = time.time()
        shard_samples = 0
        shard_tokens = 0
        shard_image_tokens = 0
        shard_text_tokens = 0
        shard_skipped = 0

        # Create builder for this specific shard
        shard_base = shard_name.replace(".tar", "")
        shard_output_path = os.path.join(self.output_dir, shard_base)
        builder = self.IndexedDatasetBuilder(f"{shard_output_path}.bin", dtype=self.dtype)

        try:
            # Create WebDataset pipeline
            dataset = self._create_dataset(shard_path)

            # Process samples
            has_filtering = self.config.get("min_resolution") or self.config.get("max_resolution")

            for sample in dataset:
                # Unpack based on whether we have JSON metadata
                if has_filtering:
                    img, text, json_data, key = sample
                    # Apply resolution filtering if enabled
                    if not self._should_process_sample(json_data):
                        shard_skipped += 1
                        continue
                else:
                    img, text, key = sample

                if not text:
                    self.logger.debug(f"Skipping {key}: empty text")
                    shard_skipped += 1
                    continue

                # Process sample
                success = self._process_sample(img, text, key, builder)
                if success:
                    shard_samples += 1
                    shard_tokens += success["total_tokens"]
                    shard_image_tokens += success["image_tokens"]
                    shard_text_tokens += success["text_tokens"]

                # Log progress periodically (only on rank 0)
                if self.worker_id == 0 and shard_samples % 100 == 0 and shard_samples > 0:
                    self._log_progress(shard_name, shard_samples, shard_start)

            # Finalize the builder for this shard
            builder.finalize(f"{shard_output_path}.idx")

            # Update global stats
            self._update_stats(shard_samples, shard_tokens, shard_image_tokens, shard_text_tokens, shard_skipped)

            # Report completion (only on rank 0)
            elapsed = time.time() - shard_start
            if self.worker_id == 0:
                self.logger.info(
                    f"Completed {shard_name}: {shard_samples} samples, "
                    f"{shard_tokens} tokens in {elapsed:.1f}s -> {shard_base}.bin/idx"
                )

            return {
                "success": True,
                "shard": shard_name,
                "samples": shard_samples,
                "tokens": shard_tokens,
                "skipped": shard_skipped,
                "time": elapsed,
            }

        except Exception as e:
            self.logger.error(f"Failed to process {shard_name}: {e}")
            self.stats["errors"] += 1
            return {"success": False, "shard": shard_name, "error": str(e)}

    def _create_dataset(self, shard_path: str):
        """Create WebDataset pipeline with optimized I/O."""
        # Check if we have JSON metadata for filtering
        has_json = self.config.get("min_resolution") or self.config.get("max_resolution")

        if has_json:
            # Include JSON metadata for filtering
            return (
                wds.WebDataset(shard_path, shardshuffle=False)
                .decode("pil")
                .to_tuple("jpg;png;jpeg;webp", "txt", "json", "__key__")
                .batched(64)  # Batch for I/O efficiency
                .unbatched()  # Unbatch for clean iteration
            )
        else:
            # Original pipeline without JSON
            return (
                wds.WebDataset(shard_path, shardshuffle=False)
                .decode("pil")
                .to_tuple("jpg;png;jpeg;webp", "txt", "__key__")
                .batched(64)  # Batch for I/O efficiency
                .unbatched()  # Unbatch for clean iteration
            )

    def _should_process_sample(self, json_data: Dict) -> bool:
        """Check if sample meets filtering criteria."""
        min_res = self.config.get("min_resolution")
        max_res = self.config.get("max_resolution")

        # If no filtering configured, process everything
        if not min_res and not max_res:
            return True

        # Check if metadata has dimensions
        if "width" not in json_data or "height" not in json_data:
            return True  # Process if no metadata available

        # Apply resolution filtering
        resolution = json_data["width"] * json_data["height"]

        if min_res and resolution < min_res:
            return False
        if max_res and resolution > max_res:
            return False

        return True

    def _process_sample(self, image, text: str, key: str, builder) -> Optional[Dict]:
        """
        Process a single image-text pair.

        Args:
            image: PIL Image
            text: Text string
            key: Sample key
            builder: IndexedDatasetBuilder for this shard

        Returns:
            Dict with token counts if successful, None otherwise
        """
        try:
            # Clean text
            text = text.strip() if text else ""
            if not text:
                return None

            # Tokenize image-text pair
            tokens = self.tokenizer.tokenize_image_text_pair(image, text)
            total_tokens = len(tokens)

            # Separate image and text token counts
            image_tokens, text_tokens = self._count_tokens(tokens)

            # Save to dataset
            tokens_np = tokens.cpu().numpy() if torch.is_tensor(tokens) else tokens
            builder.add_document(tokens_np, [len(tokens_np)])

            # Don't update global stats here - will be done in _update_stats
            # Just return the counts

            return {"total_tokens": total_tokens, "image_tokens": image_tokens, "text_tokens": text_tokens}

        except Exception as e:
            self.logger.error(f"Error processing {key}: {e}")
            self.stats["errors"] += 1
            return None

    def _count_tokens(self, tokens) -> tuple:
        """
        Count image and text tokens separately.

        Returns:
            (image_token_count, text_token_count)
        """
        tokens_list = tokens.tolist() if torch.is_tensor(tokens) else list(tokens)

        # Find boundary between image and text
        assert self.img_end_id in tokens_list, f"img_end token {self.img_end_id} must be present in tokenized sequence"

        img_end_idx = tokens_list.index(self.img_end_id)
        image_count = img_end_idx + 1  # Include img_end token
        text_count = len(tokens_list) - image_count

        return image_count, text_count

    def _update_stats(self, samples: int, tokens: int, image_tokens: int, text_tokens: int, skipped: int):
        """Update global statistics."""
        self.stats["shards_processed"] += 1
        self.stats["samples_processed"] += samples
        self.stats["total_tokens"] += tokens
        self.stats["image_tokens"] += image_tokens
        self.stats["text_tokens"] += text_tokens
        self.stats["samples_skipped"] += skipped

    def _log_progress(self, shard_name: str, samples: int, start_time: float):
        """Log processing progress."""
        elapsed = time.time() - start_time
        rate = samples / elapsed if elapsed > 0 else 0
        self.logger.info(f"  {shard_name}: {samples} samples, {rate:.1f} samples/s")

    def run(self, shard_queue) -> Dict[str, Any]:
        """Main processing loop."""
        if self.worker_id == 0:
            self.logger.info("Starting processing loop")

        while True:
            # Get next shard
            shard_path = ray.get(shard_queue.get_next_shard.remote(self.worker_id))

            if shard_path is None:
                if self.worker_id == 0:
                    self.logger.info("No more shards to process")
                break

            # Process shard
            result = self.process_shard(shard_path)

            # Report result
            if result["success"]:
                ray.get(shard_queue.mark_completed.remote(shard_path))
            else:
                ray.get(shard_queue.mark_failed.remote(shard_path, result.get("error", "Unknown error")))

        # Calculate final metrics
        elapsed = time.time() - self.stats["start_time"]
        self.stats["elapsed_time"] = elapsed
        self.stats["throughput"] = self.stats["total_tokens"] / elapsed if elapsed > 0 else 0

        return self.stats


# ============================================================================
# Main Processing Pipeline
# ============================================================================


def process_image_text_pairs(config: Dict):
    """
    Main processing pipeline with Ray distributed workers.

    Args:
        config: Configuration dictionary
    """
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    resources = ray.available_resources()
    num_gpus = config.get("num_gpus") or int(resources.get("GPU", 1))

    logging.info(f"Starting with {num_gpus} GPU workers")

    # Get input files
    input_files = get_input_files(config)
    if not input_files:
        return

    # Create work queue
    shard_queue = ShardQueue.remote(input_files)

    # Launch workers
    workers = [EMU3ImageTextWorker.remote(config, i) for i in range(num_gpus)]

    # Start processing
    futures = [worker.run.remote(shard_queue) for worker in workers]

    # Monitor progress
    monitor_progress(shard_queue, len(input_files))

    # Collect results
    results = ray.get(futures)

    # Print summary and save dataset info
    print_summary(results, shard_queue, config, input_files)

    # Shutdown
    ray.shutdown()


def get_input_files(config: Dict) -> list:
    """Get list of input files with optional range filtering."""
    import glob

    # Get all matching files
    all_files = sorted(glob.glob(config["input_pattern"]))

    if not all_files:
        logging.error(f"No files found: {config['input_pattern']}")
        return []

    # Apply range filter if specified
    if config.get("range"):
        all_files = apply_range_filter(all_files, config["range"])

    if not all_files:
        return []

    logging.info(f"Processing {len(all_files)} shards")
    return all_files


def apply_range_filter(files: list, range_str: str) -> list:
    """Apply range filter to file list."""
    try:
        # Parse range
        if ":" in range_str:
            parts = range_str.split(":")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else len(files)
        else:
            start = int(range_str)
            end = start + 1

        # Validate range
        assert (end - start) % 4 == 0, f"Range size must be multiple of 4"

        # Apply range
        filtered = files[start:end]

        if filtered:
            logging.info(f"Range {range_str}: {len(filtered)} files")
            logging.info(f"  First: {Path(filtered[0]).name}")
            logging.info(f"  Last: {Path(filtered[-1]).name}")

        return filtered

    except Exception as e:
        logging.error(f"Invalid range '{range_str}': {e}")
        return []


def monitor_progress(shard_queue, total_shards: int):
    """Monitor processing progress with progress bar."""
    pbar = tqdm(total=total_shards, desc="Shards processed")
    last_completed = 0

    while True:
        # Check status
        status = ray.get(shard_queue.get_status.remote())

        # Update progress
        completed = status["completed"]
        if completed > last_completed:
            pbar.update(completed - last_completed)
            last_completed = completed

        # Check if done
        if status["pending"] == 0 and status["in_progress"] == 0:
            break

        time.sleep(0.1)

    pbar.close()


def print_summary(results: list, shard_queue, config: Dict, input_files: list):
    """Print processing summary and save dataset info."""
    # Header
    print("\n" + "=" * 70)
    print("IMAGE-TEXT TOKENIZATION COMPLETE")
    print("=" * 70)

    # Calculate totals
    totals = calculate_totals(results)

    # Per-worker stats
    print("\nPer-Worker Statistics:")
    print("-" * 70)

    for i, r in enumerate(results):
        print(
            f"Worker {i:2d}: {r['shards_processed']:3d} shards | "
            f"{r['samples_processed']:6d} samples | "
            f"{r['throughput']/1000:6.1f}K tok/s"
        )

    # Summary
    print_totals(totals)

    # Check for failures
    check_failures(shard_queue)

    # Save dataset info JSON
    save_dataset_info(results, totals, config, input_files)


def calculate_totals(results: list) -> Dict:
    """Calculate total statistics from worker results."""
    return {
        "shards": sum(r["shards_processed"] for r in results),
        "samples": sum(r["samples_processed"] for r in results),
        "skipped": sum(r.get("samples_skipped", 0) for r in results),
        "tokens": sum(r["total_tokens"] for r in results),
        "image_tokens": sum(r["image_tokens"] for r in results),
        "text_tokens": sum(r["text_tokens"] for r in results),
        "errors": sum(r["errors"] for r in results),
        "time": max(r["elapsed_time"] for r in results),
    }


def print_totals(totals: Dict):
    """Print summary totals."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Basic stats
    print(f"Shards:     {totals['shards']:8d}")
    print(f"Samples:    {totals['samples']:8d}")

    if totals["skipped"] > 0:
        print(f"Skipped:    {totals['skipped']:8d}")

    print(f"Tokens:     {totals['tokens']:8,d}")

    # Token distribution
    if totals["samples"] > 0:
        print(f"\nToken Distribution:")
        print(f"  Image:    {totals['image_tokens']:12,d} " f"({totals['image_tokens']/totals['tokens']*100:5.1f}%)")
        print(f"  Text:     {totals['text_tokens']:12,d} " f"({totals['text_tokens']/totals['tokens']*100:5.1f}%)")

        # Averages
        print(f"\nAverage per Sequence:")
        print(f"  Total:    {totals['tokens']/totals['samples']:8.1f}")
        print(f"  Image:    {totals['image_tokens']/totals['samples']:8.1f}")
        print(f"  Text:     {totals['text_tokens']/totals['samples']:8.1f}")

    # Performance
    print(f"\nPerformance:")
    print(f"  Time:     {totals['time']:8.1f}s")

    if totals["time"] > 0:
        print(f"  Samples:  {totals['samples']/totals['time']:8.1f} samples/sec")
        print(f"  Tokens:   {totals['tokens']/totals['time']/1000:8.1f}K tokens/sec")

    # Errors
    if totals["errors"] > 0:
        print(f"\n⚠ Errors:   {totals['errors']}")

    print("=" * 70)


def check_failures(shard_queue):
    """Check for failed shards."""
    status = ray.get(shard_queue.get_status.remote())
    if status["failed"] > 0:
        print(f"\n⚠ Warning: {status['failed']} shards failed processing")


def save_dataset_info(results: list, totals: Dict, config: Dict, input_files: list):
    """Save dataset information to JSON file."""
    import datetime

    # Prepare dataset info
    dataset_info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "configuration": {
            "input_pattern": config["input_pattern"],
            "output_dir": config["output_dir"],
            "tokenizer_path": config["tokenizer_path"],
            "num_gpus": len(results),
            "range": config.get("range", "all"),
            "num_shards": len(input_files),
        },
        "statistics": {
            "total_shards": totals["shards"],
            "total_samples": totals["samples"],
            "samples_skipped": totals["skipped"],
            "total_tokens": totals["tokens"],
            "image_tokens": totals["image_tokens"],
            "text_tokens": totals["text_tokens"],
            "errors": totals["errors"],
            "processing_time_seconds": totals["time"],
        },
        "averages": {
            "tokens_per_sequence": totals["tokens"] / totals["samples"] if totals["samples"] > 0 else 0,
            "image_tokens_per_sequence": totals["image_tokens"] / totals["samples"] if totals["samples"] > 0 else 0,
            "text_tokens_per_sequence": totals["text_tokens"] / totals["samples"] if totals["samples"] > 0 else 0,
        },
        "performance": {
            "samples_per_second": totals["samples"] / totals["time"] if totals["time"] > 0 else 0,
            "tokens_per_second_k": totals["tokens"] / totals["time"] / 1000 if totals["time"] > 0 else 0,
        },
        "worker_details": [],
    }

    # Add per-worker details
    for i, r in enumerate(results):
        worker_info = {
            "worker_id": i,
            "shards_processed": r["shards_processed"],
            "samples_processed": r["samples_processed"],
            "total_tokens": r["total_tokens"],
            "image_tokens": r["image_tokens"],
            "text_tokens": r["text_tokens"],
            "errors": r["errors"],
            "elapsed_time": r["elapsed_time"],
            "throughput": r["throughput"],
        }
        dataset_info["worker_details"].append(worker_info)

    # Add shard list
    dataset_info["shards_processed"] = [Path(f).name for f in input_files]

    # Save to JSON file
    output_path = Path(config["output_dir"]) / "dataset_info.json"
    try:
        with open(output_path, "w") as f:
            json.dump(dataset_info, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save dataset info: {e}")


# ============================================================================
# Command Line Interface
# ============================================================================


def main():
    """Main entry point."""
    # Use shared parser with filtering and range
    parser = add_emu3_tokenization_args(description="EMU3 image-text tokenization with Ray distributed processing")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # Process
    config = vars(args)
    process_image_text_pairs(config)


if __name__ == "__main__":
    main()
