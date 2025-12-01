#!/usr/bin/env python3
"""
Ray-based EMU3 tokenization with dynamic work scheduling.
GPUs pull work as they become available (work stealing pattern).
"""

import argparse
import glob
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import ray
from tqdm import tqdm

# Add parent directory to path for local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.parse_utils import add_emu3_tokenization_args

# ============================================================================
# Shard Queue for Dynamic Scheduling
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

    def get_status(self) -> Dict:
        """Get current processing status."""
        return {
            "pending": len(self.pending),
            "in_progress": len(self.in_progress),
            "completed": len(self.completed),
            "failed": len(self.failed),
        }


# ============================================================================
# GPU Worker with Dynamic Work Pulling
# ============================================================================


@ray.remote(num_gpus=1)
class EMU3DynamicWorker:
    """Worker that pulls shards dynamically from queue."""

    def __init__(self, config: Dict, worker_id: int):
        """Initialize worker with all necessary components."""
        self.config = config
        self.worker_id = worker_id

        # Setup environment and all components
        self._setup_environment()
        self._initialize_components()

    def _setup_environment(self):
        """Setup Python paths and import all required modules."""
        import sys
        from pathlib import Path

        # Add paths for local imports
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        sys.path.insert(0, str(parent_dir / "utils"))

        # Import all required modules once
        import torch
        import webdataset as wds
        from indexed_dataset_megatron import DType, IndexedDatasetBuilder
        from tokenization_emu3_image_only import EMU3ImageOnlyTokenizer

        # Store as class attributes for later use
        self.torch = torch
        self.wds = wds
        self.EMU3ImageOnlyTokenizer = EMU3ImageOnlyTokenizer
        self.DType = DType
        self.IndexedDatasetBuilder = IndexedDatasetBuilder

    def _initialize_components(self):
        """Initialize all worker components."""
        # Setup logging
        self.logger = logging.getLogger(f"Worker{self.worker_id:02d}")
        self.logger.setLevel(logging.INFO)

        # Setup device
        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        self.logger.info(f"Worker {self.worker_id} initialized on {self.device}")

        # Initialize tokenizer
        self.tokenizer = self.EMU3ImageOnlyTokenizer(
            text_tokenizer_path=self.config["tokenizer_path"],
            device=self.device,
            min_pixels=self.config.get("min_resolution"),
        )

        # Setup output directory
        self.output_dir = self._build_output_path()
        os.makedirs(self.output_dir, exist_ok=True)

        # Store dtype for dataset builders
        self.dtype = self.DType.optimal_dtype(self.tokenizer.text_tokenizer.vocab_size)

        # Initialize statistics
        self.stats = {
            "shards_processed": 0,
            "samples_processed": 0,
            "tokens_generated": 0,
            "errors": 0,
            "start_time": time.time(),
        }

        self.logger.info(f"Output directory: {self.output_dir}")

    def _build_output_path(self) -> str:
        """Build output directory path with range and resolution info."""
        output_dir = self.config["output_dir"]

        # Add range info if specified
        if self.config.get("range"):
            range_str = self.config["range"].replace(":", "-")
            output_dir = f"{output_dir}_range_{range_str}"

        # Add resolution info if filtering is enabled
        min_res = self.config.get("min_resolution")
        max_res = self.config.get("max_resolution")
        if min_res or max_res:
            min_str = str(min_res) if min_res else "0"
            max_str = str(max_res) if max_res else "inf"
            output_dir = f"{output_dir}_res_{min_str}_{max_str}"

        return output_dir

    def process_shard(self, shard_path: str) -> Dict[str, Any]:
        """Process a single shard with filtering."""
        shard_name = Path(shard_path).name
        self.logger.info(f"Starting {shard_name}")

        start_time = time.time()
        samples = 0
        tokens = 0

        # Create builder for this specific shard
        shard_base = shard_name.replace(".tar", "")
        shard_output_path = os.path.join(self.output_dir, shard_base)
        builder = self.IndexedDatasetBuilder(f"{shard_output_path}.bin", dtype=self.dtype)

        try:
            # Create dataset pipeline with efficient batching
            dataset = (
                self.wds.WebDataset(shard_path, shardshuffle=False)
                .decode("pil")
                .to_tuple("jpg;png;jpeg;webp", "json", "__key__")
                .batched(64)  # Batch for I/O efficiency
                .unbatched()  # Unbatch for individual processing
            )

            # Process samples
            for img, json_data, key in dataset:
                # Apply resolution filtering if enabled
                if not self._should_process_sample(json_data):
                    continue

                try:
                    # Tokenize image
                    img_tokens = self.tokenizer.tokenize_image(img)
                    tokens_np = img_tokens.cpu().numpy() if self.torch.is_tensor(img_tokens) else img_tokens

                    # Add to dataset
                    builder.add_document(tokens_np, [len(tokens_np)])

                    samples += 1
                    tokens += len(tokens_np)

                except Exception as e:
                    self.logger.warning(f"Error processing {key}: {e}")
                    self.stats["errors"] += 1

            # Finalize the builder for this shard
            builder.finalize(f"{shard_output_path}.idx")

            # Update statistics
            self.stats["shards_processed"] += 1
            self.stats["samples_processed"] += samples
            self.stats["tokens_generated"] += tokens

            elapsed = time.time() - start_time
            self.logger.info(
                f"Completed {shard_name}: {samples} samples, "
                f"{tokens} tokens in {elapsed:.1f}s -> {shard_base}.bin/idx"
            )

            return {"success": True, "shard": shard_name, "samples": samples, "tokens": tokens, "time": elapsed}

        except Exception as e:
            self.logger.error(f"Failed to process {shard_name}: {e}")
            self.stats["errors"] += 1
            return {"success": False, "shard": shard_name, "error": str(e)}

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
            if result["success"]:
                ray.get(shard_queue.mark_completed.remote(shard_path))
            else:
                ray.get(shard_queue.mark_failed.remote(shard_path, result.get("error", "Unknown")))

        # Calculate and return final statistics
        elapsed = time.time() - self.stats["start_time"]
        self.stats["elapsed_time"] = elapsed
        self.stats["throughput"] = self.stats["tokens_generated"] / elapsed if elapsed > 0 else 0

        return self.stats


# ============================================================================
# Main Pipeline
# ============================================================================


def process_with_dynamic_scheduling(config: Dict):
    """Main processing with dynamic work distribution."""

    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    resources = ray.available_resources()
    num_gpus = config.get("num_gpus") or int(resources.get("GPU", 1))

    logging.info(f"Starting with {num_gpus} GPU workers")

    # Get list of shards
    input_files = sorted(glob.glob(config["input_pattern"]))
    if not input_files:
        logging.error(f"No files found: {config['input_pattern']}")
        return

    # Apply range filter if specified
    if config.get("range"):
        try:
            start, end = map(int, config["range"].split(":"))
            input_files = input_files[start:end]
            logging.info(f"Processing range {start}:{end} of shards")
        except (ValueError, IndexError) as e:
            logging.error(f"Invalid range format: {config['range']}. Use format 'start:end'")
            return

    logging.info(f"Found {len(input_files)} shards to process")

    # Create shard queue
    shard_queue = ShardQueue.remote(input_files)

    # Create and start workers
    workers = [EMU3DynamicWorker.remote(config, i) for i in range(num_gpus)]
    futures = [worker.run.remote(shard_queue) for worker in workers]

    # Monitor progress
    monitor_progress(shard_queue, len(input_files))

    # Get final results
    results = ray.get(futures)

    # Print summary and save dataset info
    print_summary(results)
    save_dataset_info(results, config, input_files, final_status=ray.get(shard_queue.get_status.remote()))

    # Check for failed shards
    final_status = ray.get(shard_queue.get_status.remote())
    if final_status["failed"] > 0:
        logging.warning(f"{final_status['failed']} shards failed processing")

    ray.shutdown()


def monitor_progress(shard_queue, total_shards: int):
    """Monitor processing progress with progress bar."""
    pbar = tqdm(total=total_shards, desc="Shards processed")
    last_completed = 0

    while True:
        # Check queue status
        status = ray.get(shard_queue.get_status.remote())

        # Update progress bar
        new_completed = status["completed"]
        if new_completed > last_completed:
            pbar.update(new_completed - last_completed)
            last_completed = new_completed

        # Check if done
        if status["pending"] == 0 and status["in_progress"] == 0:
            break

        # Check every 2 minutes for long-running shards
        time.sleep(120)

    pbar.close()


def print_summary(results: List[Dict]):
    """Print processing summary."""
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    total_shards = sum(r["shards_processed"] for r in results)
    total_samples = sum(r["samples_processed"] for r in results)
    total_tokens = sum(r["tokens_generated"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    max_time = max(r["elapsed_time"] for r in results)

    for i, r in enumerate(results):
        print(
            f"Worker {i}: {r['shards_processed']} shards, "
            f"{r['samples_processed']} samples, "
            f"{r['throughput']:.0f} tok/s"
        )

    print("-" * 60)
    print(f"Total: {total_shards} shards, {total_samples} samples, {total_tokens} tokens")
    print(f"Errors: {total_errors}")
    print(f"Time: {max_time:.1f}s")
    print(f"Overall throughput: {total_tokens/max_time:.0f} tokens/sec")
    print("=" * 60)


def save_dataset_info(results: List[Dict], config: Dict, input_files: List[str], final_status: Dict):
    """Save dataset information to JSON file."""
    import datetime
    import json
    from pathlib import Path

    # Calculate totals
    total_shards = sum(r["shards_processed"] for r in results)
    total_samples = sum(r["samples_processed"] for r in results)
    total_tokens = sum(r["tokens_generated"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    max_time = max(r["elapsed_time"] for r in results) if results else 0

    # Build actual output directory path
    output_dir = config["output_dir"]
    if config.get("range"):
        range_str = config["range"].replace(":", "-")
        output_dir = f"{output_dir}_range_{range_str}"

    min_res = config.get("min_resolution")
    max_res = config.get("max_resolution")
    if min_res or max_res:
        min_str = str(min_res) if min_res else "0"
        max_str = str(max_res) if max_res else "inf"
        output_dir = f"{output_dir}_res_{min_str}_{max_str}"

    # Prepare dataset info
    dataset_info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "configuration": {
            "input_pattern": config["input_pattern"],
            "output_dir": output_dir,
            "tokenizer_path": config["tokenizer_path"],
            "num_gpus": len(results),
            "range": config.get("range", "all"),
            "min_resolution": config.get("min_resolution"),
            "max_resolution": config.get("max_resolution"),
            "num_shards": len(input_files),
        },
        "statistics": {
            "total_shards": total_shards,
            "total_samples": total_samples,
            "total_tokens": total_tokens,
            "errors": total_errors,
            "processing_time_seconds": max_time,
            "failed_shards": final_status.get("failed", 0),
        },
        "averages": {
            "tokens_per_sequence": total_tokens / total_samples if total_samples > 0 else 0,
            "samples_per_shard": total_samples / total_shards if total_shards > 0 else 0,
        },
        "performance": {
            "samples_per_second": total_samples / max_time if max_time > 0 else 0,
            "tokens_per_second": total_tokens / max_time if max_time > 0 else 0,
        },
        "shards_processed": [Path(f).name for f in input_files[:total_shards]],
    }

    # Save to JSON file
    output_path = Path(output_dir) / "dataset_info.json"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(dataset_info, f, indent=2)
        logging.info(f"Dataset info saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save dataset info: {e}")


def main():
    """Main entry point."""
    parser = add_emu3_tokenization_args(description="EMU3 tokenization with dynamic Ray scheduling")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    config = vars(args)
    process_with_dynamic_scheduling(config)


if __name__ == "__main__":
    main()
