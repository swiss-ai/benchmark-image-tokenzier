#!/usr/bin/env python3
"""
Base pipeline class for tokenization and shared utilities.
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Optional, Tuple, Union

import ray
import torch
from tqdm import tqdm

from vision_tokenization.utils.time_utils import format_duration, get_slurm_time_limit
from vision_tokenization.vokenizers.transforms import TransformPipeline


class BasePipeline(ABC):
    """Abstract base class for tokenization pipelines."""

    def __init__(self, tokenizer_path: str, output_dir: str, num_gpus: int, device: str, **kwargs):
        """
        Initialize base pipeline.

        Args:
            tokenizer_path: Path to tokenizer
            output_dir: Output directory for tokenized data
            num_gpus: Number of parallel workers
            device: Device for tokenization (cuda/cpu)
            **kwargs: Additional arguments
        """
        self.tokenizer_path = tokenizer_path
        self.output_dir = output_dir
        self.num_gpus = num_gpus
        self.device = device

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Store additional kwargs
        self.kwargs = kwargs

    @abstractmethod
    def setup(self):
        """Setup the pipeline (initialize workers, etc.)."""
        pass

    @abstractmethod
    def process(self):
        """Main processing logic."""
        pass

    def cleanup(self):
        """Cleanup Ray resources."""
        if ray.is_initialized():
            ray.shutdown()

    def _get_completed_shards(self) -> set:
        """Get list of already completed shards by checking for BOTH .bin and .idx files.

        Requires self.num_shards to be set before calling.
        """
        import re
        import sys
        from pathlib import Path

        completed = set()
        output_path = Path(self.output_dir)

        if not output_path.exists():
            return completed

        # Pattern: rank_X_shard_Y_Z.idx where Y is shard_id and Z is total_shards
        pattern = re.compile(r"rank_\d+_shard_(\d+)_(\d+)\.idx")

        # Collect all shard counts found
        shard_counts_found = set()
        files_by_shard_count = {}

        for idx_file in output_path.glob("*.idx"):
            match = pattern.match(idx_file.name)
            if match:
                shard_id = int(match.group(1))
                total_shards = int(match.group(2))

                # Check if corresponding .bin file also exists
                bin_file = idx_file.with_suffix(".bin")
                if not bin_file.exists():
                    self.logger.warning(
                        f"Found {idx_file.name} but missing corresponding .bin file - "
                        f"shard {shard_id} will be reprocessed"
                    )
                    continue

                shard_counts_found.add(total_shards)
                if total_shards not in files_by_shard_count:
                    files_by_shard_count[total_shards] = []
                files_by_shard_count[total_shards].append(idx_file.name)

                if total_shards == self.num_shards:
                    completed.add(shard_id)

        # Check for inconsistency
        if shard_counts_found and self.num_shards not in shard_counts_found:
            # No files match the expected shard count
            self.logger.error(
                f"ERROR: No existing shards match expected count ({self.num_shards}). "
                f"Found shard counts: {sorted(shard_counts_found)}"
            )
            for count in sorted(shard_counts_found):
                self.logger.error(f"  {count} total shards: {len(files_by_shard_count[count])} files")
            self.logger.error("Clean the output directory or use a different output path")
            sys.exit(1)

        if len(shard_counts_found) > 1:
            # Multiple different shard counts found
            self.logger.error(f"ERROR: Inconsistent total shard counts found: {sorted(shard_counts_found)}")
            for count in sorted(shard_counts_found):
                self.logger.error(f"  {count} total shards: {len(files_by_shard_count[count])} files")
            self.logger.error("Clean the output directory or use a different output path")
            sys.exit(1)

        return completed

    def _load_completed_shard_stats(self) -> Dict[str, Any]:
        """Load statistics from completed shards in shard_stats/ subfolder.

        Requires self.num_shards to be set before calling.

        Returns:
            Dictionary with aggregated stats from all completed shards
        """
        import json
        import re
        from pathlib import Path

        stats_dir = Path(self.output_dir) / "shard_stats"

        # Initialize aggregated stats
        aggregated = {
            "samples_processed": 0,
            "tokens_generated": 0,
            "image_tokens": 0,
            "text_tokens": 0,
            "errors": 0,
            "samples_skipped": 0,
            "resolution_skipped": 0,
            "transform_errors": 0,
            "cuda_oom_errors": 0,
            "shard_count": 0,
            "per_shard": [],  # Keep individual shard stats
        }

        # If shard_stats directory doesn't exist, no stats to load
        if not stats_dir.exists():
            return aggregated

        # Pattern: shard_Y.json where Y is shard_id
        pattern = re.compile(r"shard_(\d+)\.json")

        for stats_file in stats_dir.glob("shard_*.json"):
            match = pattern.match(stats_file.name)
            if match:
                try:
                    with open(stats_file, "r") as f:
                        shard_stats = json.load(f)

                    # Validate shard count matches (if present in stats)
                    if "num_shards" in shard_stats and shard_stats["num_shards"] != self.num_shards:
                        self.logger.warning(
                            f"Skipping {stats_file.name}: shard count mismatch "
                            f"(expected {self.num_shards}, got {shard_stats['num_shards']})"
                        )
                        continue

                    # Aggregate
                    aggregated["samples_processed"] += shard_stats.get("samples", 0)
                    aggregated["tokens_generated"] += shard_stats.get("tokens", 0)
                    aggregated["image_tokens"] += shard_stats.get("image_tokens", 0)
                    aggregated["text_tokens"] += shard_stats.get("text_tokens", 0)
                    aggregated["errors"] += shard_stats.get("errors", 0)
                    aggregated["samples_skipped"] += shard_stats.get("skipped", 0)
                    aggregated["resolution_skipped"] += shard_stats.get("resolution_skipped", 0)
                    aggregated["transform_errors"] += shard_stats.get("transform_errors", 0)
                    aggregated["cuda_oom_errors"] += shard_stats.get("cuda_oom_errors", 0)
                    aggregated["shard_count"] += 1
                    aggregated["per_shard"].append(shard_stats)

                except (json.JSONDecodeError, IOError) as e:
                    self.logger.warning(f"Failed to load {stats_file}: {e}")

        if aggregated["shard_count"] > 0:
            self.logger.info(
                f"Loaded statistics from {aggregated['shard_count']} completed shards:\n"
                f"  - Samples: {aggregated['samples_processed']:,}\n"
                f"  - Tokens: {aggregated['tokens_generated']:,}\n"
                f"  - Errors: {aggregated['errors']}"
            )

        return aggregated

    def generate_metadata(self, results: list, processing_time: float, **kwargs) -> Dict[str, Any]:
        """
        Generate standardized metadata from worker results.

        Args:
            results: List of worker statistics
            processing_time: Total processing time in seconds
            **kwargs: Additional metadata fields (dataset_name, config_name, etc.)

        Returns:
            Metadata dictionary
        """
        # Calculate aggregate statistics
        total_samples = sum(r["samples_processed"] for r in results)
        total_tokens = sum(r["tokens_generated"] for r in results)
        total_errors = sum(r["errors"] for r in results)
        total_skipped = sum(r.get("samples_skipped", 0) for r in results)
        total_resolution_skipped = sum(r.get("resolution_skipped", 0) for r in results)
        total_transform_errors = sum(r.get("transform_errors", 0) for r in results)
        total_cuda_oom_errors = sum(r.get("cuda_oom_errors", 0) for r in results)
        total_image_tokens = sum(r.get("image_tokens", 0) for r in results)
        total_text_tokens = sum(r.get("text_tokens", 0) for r in results)

        metadata = {
            "statistics": {
                "total_samples_processed": total_samples,
                "samples_skipped": total_skipped,
                "resolution_skipped": total_resolution_skipped,
                "transform_errors": total_transform_errors,
                "total_tokens": total_tokens,
                "image_tokens": total_image_tokens,
                "text_tokens": total_text_tokens,
                "errors": total_errors,
                "cuda_oom_errors": total_cuda_oom_errors,
            },
            "averages": {
                "tokens_per_sample": total_tokens / total_samples if total_samples > 0 else 0,
                "image_tokens_per_sample": total_image_tokens / total_samples if total_samples > 0 else 0,
                "text_tokens_per_sample": total_text_tokens / total_samples if total_samples > 0 else 0,
            },
            "token_distribution": {
                "image_percentage": total_image_tokens / total_tokens * 100 if total_tokens > 0 else 0,
                "text_percentage": total_text_tokens / total_tokens * 100 if total_tokens > 0 else 0,
            },
            "processing": {
                "num_gpus": len(results),
                "processing_time_seconds": processing_time,
                "samples_per_second": total_samples / processing_time if processing_time > 0 else 0,
                "tokens_per_second": total_tokens / processing_time if processing_time > 0 else 0,
            },
            "worker_details": [
                {
                    "worker_id": i,
                    "samples_processed": r["samples_processed"],
                    "tokens_generated": r["tokens_generated"],
                    "image_tokens": r.get("image_tokens", 0),
                    "text_tokens": r.get("text_tokens", 0),
                    "errors": r["errors"],
                    "samples_skipped": r.get("samples_skipped", 0),
                    "resolution_skipped": r.get("resolution_skipped", 0),
                    "transform_errors": r.get("transform_errors", 0),
                    "cuda_oom_errors": r.get("cuda_oom_errors", 0),
                    "throughput_tokens": r.get("throughput_tokens", r.get("throughput", 0)),
                    "throughput_samples": r.get("throughput_samples", 0),
                }
                for i, r in enumerate(results)
            ],
        }

        # Add any additional kwargs to metadata
        metadata.update(kwargs)

        return metadata

    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Returns:
            Dictionary with processing results
        """
        try:
            self.logger.info(f"Starting {self.__class__.__name__}")

            # Setup
            self.setup()

            # Process
            result = self.process()

            # Cleanup
            self.cleanup()

            self.logger.info(f"Completed {self.__class__.__name__}")
            return result

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.cleanup()
            raise


def setup_logger_basic(loglevel, loggerName, formatter):
    # Get the root logger
    logger = logging.getLogger(loggerName)
    logger.setLevel(loglevel)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Add a console handler (stderr)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def setup_worker_logging(loglevel=logging.INFO):
    """Configure logging for Ray workers with worker identification."""
    worker_id = ray.get_runtime_context().get_worker_id()
    node_id = ray.get_runtime_context().get_node_id()

    # Create a custom formatter that includes worker info
    formatter = logging.Formatter(
        f"%(asctime)s - [Worker:{worker_id[:8]} / Node:{node_id[:8]}] - " f"%(name)s - %(levelname)s - %(message)s"
    )

    return setup_logger_basic(loglevel, loggerName=f"RayWorker[{worker_id}]", formatter=formatter)


@ray.remote
class ProgressActor:
    """Lightweight actor for collecting progress updates without polling."""

    def __init__(
        self,
        total_samples: int,
        log_interval: Optional[int] = None,
        slurm_time_limit: Optional[int] = None,
    ):
        """
        Initialize progress tracker.

        Args:
            total_samples: Total number of samples to process (-1 for streaming/unknown)
            log_interval: Log progress every N samples when output is piped to file (None = log every update)
            desc: Description for progress bar
            slurm_time_limit: SLURM time limit in seconds (None = auto-detect)
        """
        # Detect if output is being piped to a file
        self.log_interval = log_interval if log_interval is not None else 1000
        self.samples_processed = 0
        self.last_logged = 0
        self.total = total_samples if total_samples > 0 else None

        # Add throughput tracking
        self.start_time = time.time()

        # Sliding window tracking (last 20 updates)
        # Each entry is (timestamp, cumulative_samples_at_that_time)
        self.sliding_window = deque(maxlen=20)

        logFmt = logging.Formatter(f"%(asctime)s ::: [%(name)s] - %(levelname)s -> %(message)s")
        self.logger = setup_logger_basic(logging.INFO, loggerName="ProgressActor", formatter=logFmt)

        self.logger.warning(f"Log every {self.log_interval} steps!")
        if self.total:
            self.logger.warning(f"Total samples to process: {self.total:,}")

        # SLURM time limit detection
        if slurm_time_limit is not None:
            self.slurm_time_limit = slurm_time_limit
            self.logger.info(f"SLURM time limit override: {format_duration(slurm_time_limit)}")
        else:
            self.slurm_time_limit = get_slurm_time_limit()
            if self.slurm_time_limit:
                self.logger.info(
                    f"SLURM time limit detected: {format_duration(self.slurm_time_limit)} "
                    f"({self.slurm_time_limit} seconds)"
                )

    def update(self, samples: int):
        """Update progress with completed samples."""
        self.samples_processed += samples

        # Record this update in the sliding window
        current_time = time.time()
        self.sliding_window.append((current_time, self.samples_processed))

        if self.log_interval == 0:
            # Log every update (every batch)
            self._log_progress()
        elif (self.samples_processed - self.last_logged) >= self.log_interval:
            self._log_progress()
            self.last_logged = self.samples_processed

    def _calculate_remaining_time(self) -> Optional[float]:
        """
        Calculate estimated remaining time based on overall throughput.

        Returns:
            Estimated seconds remaining, or None if cannot estimate
            (streaming mode or insufficient data)
        """
        # Cannot estimate in streaming mode
        if self.total is None:
            return None

        # Calculate remaining samples
        remaining_samples = self.total - self.samples_processed
        if remaining_samples <= 0:
            return 0.0

        # Calculate overall throughput
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return None

        overall_throughput = self.samples_processed / elapsed

        # Need meaningful throughput (avoid division by zero or very small values)
        if overall_throughput < 0.01:  # Less than 0.01 samples/sec
            return None

        # Estimate remaining time
        estimated_seconds = remaining_samples / overall_throughput
        return estimated_seconds

    def _check_timeout_risk(self, estimated_remaining: float) -> bool:
        """
        Check if job is at risk of timeout.

        Args:
            estimated_remaining: Estimated seconds remaining

        Returns:
            True if timeout risk detected, False otherwise
        """
        if self.slurm_time_limit is None:
            return False

        elapsed = time.time() - self.start_time
        slurm_remaining = self.slurm_time_limit - elapsed

        # Risk if estimated time exceeds remaining SLURM time
        # Add 5% buffer for safety margin (1.05x)
        return estimated_remaining > (slurm_remaining / 1.05)

    def _log_progress(self):
        """Log progress in a file-friendly format with runtime estimation."""
        # Calculate overall throughput (from start)
        overall_throughput = self.samples_processed / (time.time() - self.start_time)

        # Calculate sliding window throughput (last 20 updates)
        sliding_window_throughput = None
        if len(self.sliding_window) >= 2:
            oldest_time, oldest_samples = self.sliding_window[0]
            latest_time, latest_samples = self.sliding_window[-1]
            time_diff = latest_time - oldest_time
            if time_diff > 0:
                samples_diff = latest_samples - oldest_samples
                sliding_window_throughput = samples_diff / time_diff

        # Format throughput message
        if sliding_window_throughput is not None:
            throughput_msg = f"Overall: {overall_throughput:.2f} samples/s, Recent (last {len(self.sliding_window)} updates): {sliding_window_throughput:.2f} samples/s"
        else:
            throughput_msg = f"{overall_throughput:.2f} samples/s"

        # Calculate estimated remaining time
        estimated_remaining = self._calculate_remaining_time()

        # Build time estimation message
        time_msg = ""
        timeout_risk = False
        if estimated_remaining is not None:
            time_msg = f" - [Est. remaining: {format_duration(estimated_remaining)}]"
            timeout_risk = self._check_timeout_risk(estimated_remaining)

        # Add timeout risk marker
        risk_msg = " [TIMEOUT_RISK]" if timeout_risk else ""

        if self.total:
            percentage = (self.samples_processed / self.total) * 100
            self.logger.info(
                f"{self.samples_processed:,}/{self.total:,} samples ({percentage:.1f}%) - "
                f"[Throughput: {throughput_msg}]{time_msg}{risk_msg}"
            )
        else:
            # Streaming mode: no percentage or time estimation
            self.logger.info(f"{self.samples_processed:,} samples [Throughput: {throughput_msg}]")

    def close(self):
        """Close the progress tracker and return total processed samples."""
        # Log any remaining un-logged progress before closing
        if self.samples_processed > self.last_logged:
            self._log_progress()
            self.last_logged = self.samples_processed

        self.logger.info(f"Progress complete: {self.samples_processed:,} total samples processed")
        return self.samples_processed


class BaseTokenizerWorker:
    """
    Base class for tokenization workers.
    Contains shared tokenizer initialization and processing logic.
    Subclasses handle output file creation and work distribution.
    """

    def __init__(
        self,
        tokenizer_path: str,
        worker_id: int,
        mode: str,  # "image_only", "image2text", "text2image", or "sft"
        min_pixels: int,  # For tokenizer preprocessing
        max_pixels: int,  # For tokenizer preprocessing
        batch_mode: Optional[str] = None,  # "simple", "sorted", "clustered"
        batch_size: int = 1,
        buffer_size: Optional[int] = None,
        resize_size: Union[int, Tuple[int, int], str] = "avg",
        image_field: str = "image",
        text_field: str = "text",
        device: str = None,
        min_image_pixels: Optional[int] = None,  # For filtering images
        max_image_pixels: Optional[int] = None,  # For filtering images
        transform_pipeline: Optional[TransformPipeline] = None,  # For data transforms
        conversation_transform: Optional[str] = None,  # For SFT conversation transforms
    ):
        self.worker_id = worker_id
        self.mode = mode
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_field = image_field
        self.text_field = text_field
        self.min_image_pixels = min_image_pixels
        self.max_image_pixels = max_image_pixels
        self.transform_pipeline = transform_pipeline

        # Setup logging
        self.logger = setup_worker_logging(logging.WARNING)

        # Import here to avoid circular imports
        from vision_tokenization.vokenizers.emu import create_tokenizer

        # Initialize tokenizer using factory function with conversation_transform
        # (only used for SFT mode, passed via **kwargs to EMUSftTokenizer)
        self.tokenizer = create_tokenizer(
            mode=mode,
            text_tokenizer_path=tokenizer_path,
            device=self.device,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            conversation_transform=conversation_transform,
        )

        # Initialize batching function if batch_mode is provided
        if batch_mode is not None:
            if batch_mode == "simple":
                from vision_tokenization.pipelines.batching import SimpleBatcher

                self.batcher = SimpleBatcher(batch_size, resize_size)
                self.buffer_size = buffer_size or batch_size
            elif batch_mode == "sorted":
                from vision_tokenization.pipelines.batching import SortedBatcher

                self.batcher = SortedBatcher(batch_size, resize_size)
                self.buffer_size = buffer_size or batch_size * 8
            elif batch_mode == "clustered":
                from vision_tokenization.pipelines.batching import ClusteredBatcher

                self.batcher = ClusteredBatcher(batch_size, resize_size, device=self.device)
                self.buffer_size = buffer_size or batch_size * 8
            else:
                raise ValueError(f"Unrecognized batching mode {batch_mode}")

            assert self.buffer_size >= batch_size, "Buffer size must be at least as big as batch size"
        else:
            self.batcher = None
            self.buffer_size = None

        # Initialize stats
        import time

        self.stats = {
            "batches_processed": 0,
            "samples_processed": 0,
            "tokens_generated": 0,
            "image_tokens": 0,  # For multimodal modes (sft, image2text, text2image)
            "text_tokens": 0,  # For multimodal modes (sft, image2text, text2image)
            "errors": 0,
            "samples_skipped": 0,  # Samples skipped due to missing data
            "resolution_skipped": 0,  # Samples skipped due to resolution filtering
            "transform_errors": 0,  # Samples skipped due to transform errors
            "cuda_oom_errors": 0,  # CUDA Out-Of-Memory errors
            "start_time": time.time(),
        }

        self.logger.warning(f"Worker {worker_id} initialized on {self.device} in {mode} mode")

    def should_process_resolution(self, image) -> bool:
        """
        Check if image meets resolution criteria for filtering.
        Note: image.size is O(1) - just returns stored dimensions without decoding.

        Args:
            image: PIL Image object

        Returns:
            True if image should be processed, False if it should be skipped
        """
        if not self.min_image_pixels and not self.max_image_pixels:
            return True

        width, height = image.size
        resolution = width * height

        # Filter based on configured thresholds
        if self.min_image_pixels and resolution < self.min_image_pixels:
            return False
        if self.max_image_pixels and resolution > self.max_image_pixels:
            return False

        return True

    def get_sample_status(self, image, text) -> str:
        """
        Determine the processing status of a sample.

        Args:
            image: Image data (may be None)
            text: Text data (may be None)

        Returns:
            Status string: 'ok', 'data_skip', or 'resolution_skip'
        """
        # Check for missing image
        if image is None:
            # TODO: this will prevent txt-only sft tokenization form working!
            return "data_skip"

        # Check if mode requires text and it's missing
        if self.mode in ["image2text", "text2image", "sft"] and not text:
            return "data_skip"

        # Check resolution filtering
        if not self.should_process_resolution(image):
            return "resolution_skip"

        return "ok"

    def tokenize_sample(self, image, text) -> Optional[Any]:
        """
        Tokenize a single sample (shared logic).

        Args:
            image: Input image
            text: Input text (may be None for image-only mode)

        Returns:
            Tokens (as tensor or numpy array), or None if error
        """
        try:
            import torch

            # Use unified tokenize method
            tokens = self.tokenizer.tokenize(image, text)

            # Convert to numpy if needed
            tokens_np = tokens.cpu().numpy() if torch.is_tensor(tokens) else tokens

            return tokens_np

        except Exception as e:
            self.logger.warning(f"Error processing sample: {e}")
            return None

    def tokenize_batch(self, images, resize_size, text=None) -> Optional[Any]:
        """
        Tokenize a batch of samples (shared logic for batching mode).

        Args:
            images: List of input images
            resize_size: Target size for resizing images in the batch
            text: Optional list of text (may be None for image-only mode)

        Returns:
            Tokens (as tensor or numpy array), or None if error
        """
        try:
            import torch

            # Use unified tokenize_images method (for batched processing)
            tokens = self.tokenizer.tokenize_batch(images, resize_size, text)
            tokens_np = tokens.cpu().numpy() if torch.is_tensor(tokens) else tokens

            return tokens_np

        except Exception as e:
            self.logger.warning(f"Error processing batch: {e}")
            # TODO: might add error logging and error skip!
            raise

    def _extract_data(self, sample: Dict) -> tuple:
        """
        Extract image and/or text from sample based on mode.
        Can be overridden by specific implementations.

        Text loaded here can be either string only or any format of messages (ex. list of objects with {"content": ..., "role": ...})."})
        """
        try:
            # Extract image
            image = sample[self.image_field]
            # Unwrap single-item lists
            if (
                isinstance(image, list) and len(image) == 1
            ):  # TODO: only supports single image!, might want to support multiple images in future
                image = image[0]

            # Extract text (only for modes that need it)
            text = sample[self.text_field] if self.mode in ["image2text", "text2image", "sft"] else None

            return image, text

        except KeyError as e:
            field = e.args[0]
            raise KeyError(
                f"Required field '{field}' not found in sample. " f"Available fields: {', '.join(sample.keys())}"
            ) from None

    def apply_transforms(self, image, text):
        """
        Apply transform pipeline to image and text if configured.

        Args:
            image: PIL Image or None
            text: Text string or None

        Returns:
            Tuple of (transformed_image, transformed_text)

        Raises:
            TransformError: If any transform fails
        """
        if self.transform_pipeline is None:
            return image, text
        return self.transform_pipeline.apply(image, text)

    def _is_cuda_oom_error(self, exception: Exception) -> bool:
        """
        Check if an exception is a CUDA Out-Of-Memory error.

        Args:
            exception: The exception to check

        Returns:
            True if this is a CUDA OOM error, False otherwise
        """
        import torch

        # Check if it's the explicit CUDA OOM exception type
        if isinstance(exception, torch.cuda.OutOfMemoryError):
            return True

        # Check if it's a RuntimeError with OOM in the message
        if isinstance(exception, RuntimeError):
            error_str = str(exception).lower()
            # Common CUDA OOM error message patterns
            oom_patterns = [
                "out of memory",
                "cuda out of memory",
                "cuda error: out of memory",
            ]
            return any(pattern in error_str for pattern in oom_patterns)

        return False

    def update_stats(
        self,
        samples: int = 0,
        tokens: int = 0,
        errors: int = 0,
        skipped: int = 0,
        resolution_skipped: int = 0,
        transform_errors: int = 0,
        cuda_oom_errors: int = 0,
        image_tokens: int = 0,
        text_tokens: int = 0,
    ):
        """Update worker statistics."""
        self.stats["samples_processed"] += samples
        self.stats["tokens_generated"] += tokens
        self.stats["image_tokens"] += image_tokens
        self.stats["text_tokens"] += text_tokens
        self.stats["errors"] += errors
        self.stats["samples_skipped"] += skipped
        self.stats["resolution_skipped"] += resolution_skipped
        self.stats["transform_errors"] += transform_errors
        self.stats["cuda_oom_errors"] += cuda_oom_errors

    def format_stats_message(self, prefix: str, stats: Dict, elapsed: float = None) -> str:
        """Format statistics into a clean log message."""
        samples = stats.get("samples", stats.get("samples_processed", 0))
        tokens = stats.get("tokens", stats.get("tokens_generated", 0))
        avg_tokens = tokens / samples if samples > 0 else 0

        # Base message
        parts = [f"{samples} samples", f"{tokens} tokens ({avg_tokens:.1f} avg)"]

        # Throughput - show both metrics clearly labeled
        if elapsed and elapsed > 0:
            samples_per_sec = samples / elapsed
            tokens_per_sec = tokens / elapsed
            parts.append(f"throughput(samples): {samples_per_sec:.1f}/s, throughput(tokens): {tokens_per_sec:.1f}/s")

        # Token breakdown (always show for non-image_only modes)
        if self.mode != "image_only":
            img_tok = stats.get("image_tokens", 0)
            txt_tok = stats.get("text_tokens", 0)
            if img_tok > 0 or txt_tok > 0:
                parts.append(f"[img: {img_tok}, txt: {txt_tok}]")

        # Skip counts
        skips = []
        if stats.get("skipped", stats.get("samples_skipped", 0)) > 0:
            skips.append(f"{stats.get('skipped', stats.get('samples_skipped', 0))} data_skip")
        if stats.get("resolution_skipped", 0) > 0:
            skips.append(f"{stats['resolution_skipped']} res_skip")
        if stats.get("transform_errors", 0) > 0:
            skips.append(f"{stats['transform_errors']} transform_err")
        if stats.get("cuda_oom_errors", 0) > 0:
            skips.append(f"{stats['cuda_oom_errors']} cuda_oom")
        if skips:
            parts.append(f"({', '.join(skips)})")

        return f"{prefix}: {', '.join(parts)}"

    def get_final_stats(self) -> Dict:
        """Get final statistics for the worker."""
        import time

        elapsed = time.time() - self.stats["start_time"]
        self.stats["elapsed_time"] = elapsed

        # Calculate both throughput metrics
        self.stats["throughput_tokens"] = self.stats["tokens_generated"] / elapsed if elapsed > 0 else 0
        self.stats["throughput_samples"] = self.stats["samples_processed"] / elapsed if elapsed > 0 else 0

        # Keep old_apertus "throughput" key for backward compatibility (same as throughput_tokens)
        self.stats["throughput"] = self.stats["throughput_tokens"]

        msg = self.format_stats_message(f"Worker {self.worker_id} finished", self.stats, elapsed)
        self.logger.info(msg)

        return self.stats

    def batch_iterable_shard(self, shard, stats: Dict):
        """
        Extract, transform, filter samples from a shard and yield batches.

        Iterates over shard samples, applies transforms, checks skip conditions,
        buffers valid samples, and yields batches from the batcher.

        Args:
            shard: Iterable of samples (HF dataset shard or WDS decoded samples)
            stats: Stats tracker dictionary (modified in-place)

        Yields:
            Batch dictionaries with 'images', 'text', and 'resize_size' keys
        """
        buffer = []
        for sample in shard:
            image, text = self._extract_data(sample)
            try:
                image, text = self.apply_transforms(image, text)
            except Exception as e:
                self.logger.warning(f"Transform error: {e}")
                stats["transform_errors"] += 1
                continue

            # Check sample status and skip if necessary
            status = self.get_sample_status(image, text)
            if status == "resolution_skip":
                stats["resolution_skipped"] += 1
                continue
            elif status == "data_skip":
                stats["skipped"] += 1
                continue

            # Add sample to buffer
            if text is None:
                buffer.append({"image": image})
            else:
                buffer.append({"image": image, "text": text})

            if len(buffer) == self.buffer_size:
                for batch in self.batcher(buffer):
                    yield batch
                buffer.clear()

        # Flush remaining samples
        if buffer:
            for batch in self.batcher(buffer):
                yield batch

    def _process_shard_data(self, shard, builder, stats: Dict, log_interval: int, progress_actor=None):
        """
        Process all samples from a shard iterable into an IndexedDataset builder.

        Shared processing loop for both batched and sample-by-sample modes.
        Handles tokenization, progress updates, and token counting.

        Args:
            shard: Iterable of samples (HF dataset shard or WDS decoded samples)
            builder: IndexedDatasetBuilder to write tokens into
            stats: Stats tracker dictionary (modified in-place)
            log_interval: Log interval for progress updates
            progress_actor: Optional Ray progress tracking actor
        """
        import time

        update_interval = max(log_interval // 2, 10)
        samples_since_update = 0

        if self.batcher is not None:
            import torch

            batch_loader = self.batch_iterable_shard(shard, stats)

            for batch in batch_loader:
                try:
                    images, text = batch["images"], batch["text"]
                    tokens_batched = self.tokenize_batch(images, batch["resize_size"], text if text else None)

                    if tokens_batched is not None:
                        batch_size = len(tokens_batched)

                        all_tokens = []
                        lengths_list = []
                        for seq in tokens_batched:
                            seq_np = seq.cpu().numpy() if torch.is_tensor(seq) else seq
                            all_tokens.append(seq_np)
                            lengths_list.append(len(seq_np))

                        import numpy as np

                        tokens_np = np.concatenate(all_tokens)

                        builder.add_document(tokens_np, lengths_list)
                        stats["samples"] += batch_size
                        stats["tokens"] += len(tokens_np)
                        samples_since_update += batch_size

                        if progress_actor and samples_since_update >= update_interval:
                            progress_actor.update.remote(samples_since_update)
                            samples_since_update = 0

                        if self.mode in ["sft", "image2text", "text2image"]:
                            text_mask = tokens_np < self.tokenizer.vision_token_offset
                            stats["text_tokens"] += int(text_mask.sum())
                            stats["image_tokens"] += int((~text_mask).sum())

                        del tokens_batched, tokens_np, all_tokens, lengths_list
                    else:
                        stats["errors"] += 1

                except Exception as e:
                    import traceback

                    error_msg = f"Failed to process batch: {e}\n{traceback.format_exc()}"
                    self.logger.warning(error_msg)
                    print(f"[Worker {self.worker_id}] {error_msg}", flush=True)
                    if self._is_cuda_oom_error(e):
                        stats["cuda_oom_errors"] += 1
                    else:
                        stats["errors"] += 1
        else:
            # Sample-by-sample processing
            for sample in shard:
                image, text = self._extract_data(sample)
                try:
                    image, text = self.apply_transforms(image, text)
                except Exception as e:
                    self.logger.warning(f"Transform error: {e}")
                    stats["transform_errors"] += 1
                    continue

                status = self.get_sample_status(image, text)
                if status == "resolution_skip":
                    stats["resolution_skipped"] += 1
                    continue
                elif status == "data_skip":
                    stats["skipped"] += 1
                    continue

                try:
                    tokens_np = self.tokenize_sample(image, text)
                    if tokens_np is not None:
                        builder.add_document(tokens_np, [len(tokens_np)])
                        stats["samples"] += 1
                        stats["tokens"] += len(tokens_np)
                        samples_since_update += 1

                        if progress_actor and samples_since_update >= update_interval:
                            progress_actor.update.remote(samples_since_update)
                            samples_since_update = 0

                        if self.mode in ["sft", "image2text", "text2image"]:
                            text_mask = tokens_np < self.tokenizer.vision_token_offset
                            stats["text_tokens"] += int(text_mask.sum())
                            stats["image_tokens"] += int((~text_mask).sum())
                    else:
                        stats["errors"] += 1
                except Exception as e:
                    import traceback

                    error_msg = f"Failed to process sample: {e}\n{traceback.format_exc()}"
                    self.logger.warning(error_msg)
                    print(f"[Worker {self.worker_id}] {error_msg}", flush=True)
                    if self._is_cuda_oom_error(e):
                        stats["cuda_oom_errors"] += 1
                    else:
                        stats["errors"] += 1

        # Send final progress update for any remaining samples
        if progress_actor and samples_since_update > 0:
            progress_actor.update.remote(samples_since_update)

    def save_shard_stats(self, shard_id: int, num_shards: int, stats: Dict, output_dir: str):
        """Save statistics for a completed shard atomically.

        Args:
            shard_id: The shard ID that was processed
            num_shards: Total number of shards
            stats: Statistics dictionary from process_shard()
            output_dir: Output directory (stats saved to output_dir/shard_stats/)
        """
        import json
        from datetime import datetime
        from pathlib import Path

        stats_dir = Path(output_dir) / "shard_stats"
        stats_dir.mkdir(parents=True, exist_ok=True)

        stats_file = stats_dir / f"shard_{shard_id}.json"

        shard_stats = {
            "shard_id": shard_id,
            "worker_id": self.worker_id,
            "num_shards": num_shards,
            "samples": stats.get("samples", 0),
            "tokens": stats.get("tokens", 0),
            "image_tokens": stats.get("image_tokens", 0),
            "text_tokens": stats.get("text_tokens", 0),
            "errors": stats.get("errors", 0),
            "skipped": stats.get("skipped", 0),
            "resolution_skipped": stats.get("resolution_skipped", 0),
            "transform_errors": stats.get("transform_errors", 0),
            "cuda_oom_errors": stats.get("cuda_oom_errors", 0),
            "processing_time": stats.get("time", 0),
            "timestamp": datetime.now().isoformat(),
        }

        # Write atomically (write to temp file, then rename)
        temp_file = stats_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(shard_stats, f, indent=2)
        temp_file.rename(stats_file)

        self.logger.debug(f"Saved shard {shard_id} statistics to {stats_file}")

    def run_shards(self, shard_queue, dataset_info, num_shards, progress_actor=None) -> Dict:
        """
        Main worker loop for shard-based processing (work-stealing pattern).

        Pulls shards from the queue, processes them, saves stats, and updates
        global statistics. Used by both HF and WDS workers.

        Args:
            shard_queue: Ray remote ShardQueue for work distribution
            dataset_info: Dataset metadata passed to process_shard()
            num_shards: Total number of shards
            progress_actor: Optional progress tracking actor

        Returns:
            Final worker statistics
        """
        self.logger.info("Starting shard processing loop")

        while True:
            shard_id = ray.get(shard_queue.get_next_shard.remote(self.worker_id))

            if shard_id is None:
                self.logger.info("No more shards, finishing")
                break

            try:
                result = self.process_shard(shard_id, dataset_info, num_shards, progress_actor)
                ray.get(shard_queue.mark_completed.remote(shard_id, result))

                # Save shard statistics immediately
                self.save_shard_stats(
                    shard_id=shard_id, num_shards=num_shards, stats=result, output_dir=self.output_dir
                )

                # Update global statistics
                self.update_stats(
                    samples=result["samples"],
                    tokens=result["tokens"],
                    errors=result["errors"],
                    skipped=result.get("skipped", 0),
                    resolution_skipped=result.get("resolution_skipped", 0),
                    transform_errors=result.get("transform_errors", 0),
                    cuda_oom_errors=result.get("cuda_oom_errors", 0),
                    image_tokens=result.get("image_tokens", 0),
                    text_tokens=result.get("text_tokens", 0),
                )

            except Exception as e:
                import traceback

                error_msg = f"Failed to process shard {shard_id}: {e}\n{traceback.format_exc()}"
                self.logger.error(error_msg)
                print(f"[Worker {self.worker_id}] {error_msg}", flush=True)
                ray.get(shard_queue.mark_failed.remote(shard_id, str(e)))
