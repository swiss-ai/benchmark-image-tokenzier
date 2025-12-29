#!/usr/bin/env python3
"""
Base pipeline class for tokenization and shared utilities.
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Optional, Union, Tuple

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

    @abstractmethod
    def cleanup(self):
        """Cleanup resources."""
        pass

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
        desc: str = "Samples processed",
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
        resize_size: Union[int, Tuple[int, int], str] = 'avg',
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
            "image_tokens": 0,  # Only for SFT mode
            "text_tokens": 0,  # Only for SFT mode
            "errors": 0,
            "samples_skipped": 0,  # Samples skipped due to missing data
            "resolution_skipped": 0,  # Samples skipped due to resolution filtering
            "transform_errors": 0,  # Samples skipped due to transform errors
            "cuda_oom_errors": 0,  # CUDA Out-Of-Memory errors
            "start_time": time.time(),
        }

        # Cache img_end token ID for SFT mode
        if mode == "sft":
            #TODO: Hardcoded img end token - might be changed in the future
            self.img_end_id = self.tokenizer.text_tokenizer.convert_tokens_to_ids("<|img_end|>")
        else:
            self.img_end_id = None

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
            tokens = self.tokenizer.tokenize_images(images, resize_size, text)

            # Convert to numpy if needed
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

        # Keep old "throughput" key for backward compatibility (same as throughput_tokens)
        self.stats["throughput"] = self.stats["throughput_tokens"]

        msg = self.format_stats_message(f"Worker {self.worker_id} finished", self.stats, elapsed)
        self.logger.info(msg)

        return self.stats
