#!/usr/bin/env python3
"""
Base pipeline class for tokenization and shared utilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import ray
import torch
from tqdm import tqdm

from vision_tokenization.pipelines.transforms import TransformPipeline, TransformError


class BasePipeline(ABC):
    """Abstract base class for tokenization pipelines."""

    def __init__(
        self,
        tokenizer_path: str,
        output_dir: str,
        num_gpus: int,
        device: str,
        **kwargs
    ):
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
        total_samples = sum(r['samples_processed'] for r in results)
        total_tokens = sum(r['tokens_generated'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        total_skipped = sum(r.get('samples_skipped', 0) for r in results)
        total_resolution_skipped = sum(r.get('resolution_skipped', 0) for r in results)
        total_transform_errors = sum(r.get('transform_errors', 0) for r in results)
        total_image_tokens = sum(r.get('image_tokens', 0) for r in results)
        total_text_tokens = sum(r.get('text_tokens', 0) for r in results)

        metadata = {
            'statistics': {
                'total_samples_processed': total_samples,
                'samples_skipped': total_skipped,
                'resolution_skipped': total_resolution_skipped,
                'transform_errors': total_transform_errors,
                'total_tokens': total_tokens,
                'image_tokens': total_image_tokens,
                'text_tokens': total_text_tokens,
                'errors': total_errors
            },
            'averages': {
                'tokens_per_sample': total_tokens / total_samples if total_samples > 0 else 0,
                'image_tokens_per_sample': total_image_tokens / total_samples if total_samples > 0 else 0,
                'text_tokens_per_sample': total_text_tokens / total_samples if total_samples > 0 else 0
            },
            'token_distribution': {
                'image_percentage': total_image_tokens / total_tokens * 100 if total_tokens > 0 else 0,
                'text_percentage': total_text_tokens / total_tokens * 100 if total_tokens > 0 else 0
            },
            'processing': {
                'num_gpus': len(results),
                'processing_time_seconds': processing_time,
                'samples_per_second': total_samples / processing_time if processing_time > 0 else 0,
                'tokens_per_second': total_tokens / processing_time if processing_time > 0 else 0
            },
            'worker_details': [
                {
                    'worker_id': i,
                    'samples_processed': r['samples_processed'],
                    'tokens_generated': r['tokens_generated'],
                    'image_tokens': r.get('image_tokens', 0),
                    'text_tokens': r.get('text_tokens', 0),
                    'errors': r['errors'],
                    'samples_skipped': r.get('samples_skipped', 0),
                    'resolution_skipped': r.get('resolution_skipped', 0),
                    'transform_errors': r.get('transform_errors', 0),
                    'throughput': r.get('throughput', 0)
                } for i, r in enumerate(results)
            ]
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


@ray.remote
class ProgressActor:
    """Lightweight actor for collecting progress updates without polling."""

    def __init__(self, total_samples: int, desc: str = "Samples processed"):
        self.total_samples = total_samples
        self.processed = 0
        self.pbar = tqdm(total=total_samples, desc=desc)

    def update(self, samples: int):
        """Update progress bar with completed samples."""
        self.processed += samples
        self.pbar.update(samples)

    def close(self):
        """Close the progress bar."""
        self.pbar.close()
        return self.processed


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
        image_field: str = "image",
        text_field: str = "text",
        device: str = None,
        min_image_pixels: Optional[int] = None,  # For filtering images
        max_image_pixels: Optional[int] = None,  # For filtering images
        transform_pipeline: Optional[TransformPipeline] = None  # For data transforms
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
        self.logger = logging.getLogger(f"Worker{worker_id:02d}")
        self.logger.setLevel(logging.INFO)

        # Import here to avoid circular imports
        from vision_tokenization.vokenizers.emu import create_tokenizer

        # Initialize tokenizer using factory function
        self.tokenizer = create_tokenizer(
            mode=mode,
            text_tokenizer_path=tokenizer_path,
            device=self.device,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )

        # Initialize stats
        import time
        self.stats = {
            'batches_processed': 0,
            'samples_processed': 0,
            'tokens_generated': 0,
            'image_tokens': 0,  # Only for SFT mode
            'text_tokens': 0,   # Only for SFT mode
            'errors': 0,
            'samples_skipped': 0,  # Samples skipped due to missing data
            'resolution_skipped': 0,  # Samples skipped due to resolution filtering
            'transform_errors': 0,  # Samples skipped due to transform errors
            'start_time': time.time()
        }

        # Cache img_end token ID for SFT mode
        if mode == "sft":
            self.img_end_id = self.tokenizer.text_tokenizer.convert_tokens_to_ids("<|img_end|>")
        else:
            self.img_end_id = None

        self.logger.info(f"Worker {worker_id} initialized on {self.device} in {mode} mode")

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
            return 'data_skip'

        # Check if mode requires text and it's missing
        if self.mode in ["image2text", "text2image", "sft"] and not text:
            return 'data_skip'

        # Check resolution filtering
        if not self.should_process_resolution(image):
            return 'resolution_skip'

        return 'ok'

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
            if isinstance(image, list) and len(image) == 1: # TODO: only supports single image!, might want to support multiple images in future
                image = image[0]

            # Extract text (only for modes that need it)
            text = sample[self.text_field] if self.mode in ["image2text", "text2image", "sft"] else None

            return image, text

        except KeyError as e:
            field = e.args[0]
            raise KeyError(
                f"Required field '{field}' not found in sample. "
                f"Available fields: {', '.join(sample.keys())}"
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

    def update_stats(self, samples: int = 0, tokens: int = 0, errors: int = 0,
                     skipped: int = 0, resolution_skipped: int = 0,
                     transform_errors: int = 0,
                     image_tokens: int = 0, text_tokens: int = 0):
        """Update worker statistics."""
        self.stats['samples_processed'] += samples
        self.stats['tokens_generated'] += tokens
        self.stats['image_tokens'] += image_tokens
        self.stats['text_tokens'] += text_tokens
        self.stats['errors'] += errors
        self.stats['samples_skipped'] += skipped
        self.stats['resolution_skipped'] += resolution_skipped
        self.stats['transform_errors'] += transform_errors

    def format_stats_message(self, prefix: str, stats: Dict, elapsed: float = None) -> str:
        """Format statistics into a clean log message."""
        samples = stats.get('samples', stats.get('samples_processed', 0))
        tokens = stats.get('tokens', stats.get('tokens_generated', 0))
        avg_tokens = tokens / samples if samples > 0 else 0

        # Base message
        parts = [f"{samples} samples", f"{tokens} tokens ({avg_tokens:.1f} avg)"]

        # Throughput
        if elapsed and elapsed > 0:
            parts.append(f"{samples / elapsed:.1f} samples/s")

        # Token breakdown (always show for non-image_only modes)
        if self.mode != "image_only":
            img_tok = stats.get('image_tokens', 0)
            txt_tok = stats.get('text_tokens', 0)
            if img_tok > 0 or txt_tok > 0:
                parts.append(f"[img: {img_tok}, txt: {txt_tok}]")

        # Skip counts
        skips = []
        if stats.get('skipped', stats.get('samples_skipped', 0)) > 0:
            skips.append(f"{stats.get('skipped', stats.get('samples_skipped', 0))} data_skip")
        if stats.get('resolution_skipped', 0) > 0:
            skips.append(f"{stats['resolution_skipped']} res_skip")
        if stats.get('transform_errors', 0) > 0:
            skips.append(f"{stats['transform_errors']} transform_err")
        if skips:
            parts.append(f"({', '.join(skips)})")

        return f"{prefix}: {', '.join(parts)}"

    def get_final_stats(self) -> Dict:
        """Get final statistics for the worker."""
        import time

        elapsed = time.time() - self.stats['start_time']
        self.stats['elapsed_time'] = elapsed
        self.stats['throughput'] = self.stats['tokens_generated'] / elapsed if elapsed > 0 else 0

        msg = self.format_stats_message(f"Worker {self.worker_id} finished", self.stats, elapsed)
        self.logger.info(msg)

        return self.stats