#!/usr/bin/env python3
"""
Unified Ray workers for distributed tokenization.
Handles all tokenization modes with a single flexible worker class.
"""

import queue
import threading
import time
from typing import Dict, Optional, Tuple, Any

import ray
from datasets import load_dataset


@ray.remote
class WorkQueue:
    """Dynamic work queue for distributing batches to workers."""

    def __init__(self, total_samples: int, batch_size: int = 1):
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


from vision_tokenization.pipelines.base import BaseTokenizerWorker


@ray.remote(num_gpus=1)
class Worker(BaseTokenizerWorker):
    """
    HuggingFace dataset worker that extends BaseTokenizerWorker.
    Adds HF-specific data loading, work queue processing, and rank-based output.
    """

    def __init__(
        self,
        tokenizer_path: str,
        output_dir: str,
        worker_id: int,
        mode: str,
        min_pixels: int,
        max_pixels: int,
        image_field: str = "image",
        text_field: str = "text",
        min_image_pixels: Optional[int] = None,
        max_image_pixels: Optional[int] = None
    ):
        """
        Initialize HF worker with tokenizer and output configuration.

        Args:
            tokenizer_path: Path to tokenizer
            output_dir: Directory for output files
            worker_id: Unique worker identifier
            mode: Tokenization mode ('image_only', 'image_text_pair', or 'sft')
            min_pixels: Min pixels for tokenizer preprocessing
            max_pixels: Max pixels for tokenizer preprocessing
            image_field: Field name for images in dataset
            text_field: Field name for text in dataset
            min_image_pixels: Min pixels to filter images (optional)
            max_image_pixels: Max pixels to filter images (optional)
        """
        # Initialize base tokenizer with resolution filtering parameters
        super().__init__(
            tokenizer_path=tokenizer_path,
            worker_id=worker_id,
            mode=mode,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            image_field=image_field,
            text_field=text_field,
            min_image_pixels=min_image_pixels,
            max_image_pixels=max_image_pixels
        )

        # Setup HF-specific output (rank-based)
        self._setup_output(output_dir)

        # Initialize HF-specific prefetch queue for async loading
        self.data_queue = queue.Queue(maxsize=64)

    def _setup_output(self, output_dir: str):
        """Setup indexed dataset builder for output."""
        from vision_tokenization.pipelines.indexed_dataset_megatron import DType, IndexedDatasetBuilder
        from pathlib import Path

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.output_path = Path(output_dir) / f"rank_{self.worker_id:03d}"
        self.builder = IndexedDatasetBuilder(
            f"{self.output_path}.bin",
            dtype=DType.optimal_dtype(len(self.tokenizer.text_tokenizer))
        )

    def _load_data_async(self, indices: list, dataset_info: Dict):
        """
        Background thread to load data into queue.

        Args:
            indices: Sample indices to load
            dataset_info: Dataset metadata (name, config, split, cache_dir)
        """
        try:
            # Load dataset slice
            samples = load_dataset(
                dataset_info['name'],
                name=dataset_info.get('config'),
                split=f"{dataset_info['split']}[{indices[0]}:{indices[-1]+1}]",
                cache_dir=dataset_info.get('cache_dir')
            )

            # Process and queue samples
            for sample in samples:
                self._queue_sample(sample)

        except Exception as e:
            self.logger.error(f"Loader thread error: {e}")
        finally:
            # Add sentinel to mark end of data
            self.data_queue.put((None, None, 'sentinel'))

    def _queue_sample(self, sample: Dict):
        """
        Extract and queue a sample with appropriate status.

        Args:
            sample: Raw sample from dataset
        """
        image, text = self._extract_data(sample)

        # Use base class method to determine status
        status = self.get_sample_status(image, text)

        # Queue with appropriate data based on status
        if status == 'ok':
            self.data_queue.put((image, text, status))
        else:
            # Skip samples don't need data
            self.data_queue.put((None, None, status))

    def _process_data_stream(self) -> Dict:
        """
        Process data from queue as GPU becomes available.

        Returns:
            Statistics dictionary with samples, tokens, errors, and skip counts
        """
        stats = {
            'samples': 0,
            'tokens': 0,
            'image_tokens': 0,
            'text_tokens': 0,
            'errors': 0,
            'skipped': 0,
            'resolution_skipped': 0
        }

        while True:
            # Get next item from queue
            item = self.data_queue.get()

            # Parse queue item (supports 3-tuple format)
            image, text, status = self._parse_queue_item(item)

            # Handle different statuses
            if status == 'sentinel':
                if self.data_queue.empty():
                    break
                continue
            elif status == 'resolution_skip':
                stats['resolution_skipped'] += 1
                continue
            elif status == 'data_skip':
                stats['skipped'] += 1
                continue

            # Process valid sample
            self._process_and_save(image, text, stats)

        return stats

    def _parse_queue_item(self, item) -> Tuple[Any, Any, str]:
        """
        Parse item from queue, handling different formats.

        Args:
            item: Queue item (could be tuple or None)

        Returns:
            Tuple of (image, text, status)
        """
        if isinstance(item, tuple) and len(item) == 3:
            return item

        # Backward compatibility for old format
        if item is None:
            return None, None, 'sentinel'

        # Assume it's (image, text) tuple
        image, text = item
        return image, text, 'ok'

    def _process_and_save(self, image, text, stats: Dict):
        """
        Tokenize and save a single sample.

        Args:
            image: Image to tokenize
            text: Text to tokenize (may be None)
            stats: Statistics dictionary to update
        """
        try:
            # Tokenize using base class method
            tokens_np = self.tokenize_sample(image, text)

            if tokens_np is not None:
                # Save to indexed dataset
                self.builder.add_document(tokens_np, [len(tokens_np)])
                stats['samples'] += 1
                stats['tokens'] += len(tokens_np)

                # For SFT mode, count image vs text tokens
                if self.mode == "sft" and self.img_end_id is not None:
                    import torch
                    if torch.is_tensor(tokens_np):
                        tokens_list = tokens_np.cpu().numpy().tolist()
                    else:
                        tokens_list = tokens_np.tolist()

                    # Find img_end token
                    if self.img_end_id in tokens_list:
                        img_end_idx = tokens_list.index(self.img_end_id)
                        image_token_count = img_end_idx + 1
                        text_token_count = len(tokens_list) - image_token_count
                    else:
                        # No image tokens (shouldn't happen in SFT)
                        image_token_count = 0
                        text_token_count = len(tokens_list)

                    stats['image_tokens'] = stats.get('image_tokens', 0) + image_token_count
                    stats['text_tokens'] = stats.get('text_tokens', 0) + text_token_count
            else:
                stats['errors'] += 1

        except Exception as e:
            self.logger.warning(f"Failed to process sample: {e}")
            stats['errors'] += 1

    def process_batch(self, batch_info: Dict, dataset_info: Dict) -> Dict:
        """
        Process a batch of samples with async prefetching.

        Args:
            batch_info: Batch metadata (batch_id, indices)
            dataset_info: Dataset metadata

        Returns:
            Processing statistics
        """
        batch_id = batch_info['batch_id']
        indices = batch_info['indices']

        self.logger.info(f"Processing {batch_id} ({len(indices)} samples)")
        start_time = time.time()

        # Start async data loading
        loader = threading.Thread(
            target=self._load_data_async,
            args=(indices, dataset_info),
            daemon=True
        )
        loader.start()

        # Process data stream as it arrives
        stats = self._process_data_stream()

        # Wait for loader to complete
        loader.join()

        # Update global statistics
        self._update_global_stats(stats)

        # Log completion
        self._log_batch_completion(batch_id, stats, start_time)

        return {
            'batch_id': batch_id,
            'time': time.time() - start_time,
            **stats
        }

    def _update_global_stats(self, batch_stats: Dict):
        """Update worker's global statistics with batch results."""
        self.stats['batches_processed'] += 1
        self.update_stats(
            samples=batch_stats['samples'],
            tokens=batch_stats['tokens'],
            errors=batch_stats['errors'],
            skipped=batch_stats.get('skipped', 0),
            resolution_skipped=batch_stats.get('resolution_skipped', 0),
            image_tokens=batch_stats.get('image_tokens', 0),
            text_tokens=batch_stats.get('text_tokens', 0)
        )

    def _log_batch_completion(self, batch_id: str, stats: Dict, start_time: float):
        """Log batch completion with statistics."""
        elapsed = time.time() - start_time
        msg = self.format_stats_message(f"Completed {batch_id}", stats, elapsed)
        self.logger.info(msg)

    def run(self, work_queue, dataset_info, progress_actor=None) -> Dict:
        """
        Main worker loop: pull and process batches until done.

        Args:
            work_queue: Ray remote work queue for batch distribution
            dataset_info: Dataset metadata
            progress_actor: Optional progress tracking actor

        Returns:
            Final worker statistics
        """
        self.logger.info("Starting work loop")

        while True:
            # Get next batch from queue
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

        # Finalize output files
        self._finalize()

        # Return final statistics
        return self.get_final_stats()

    def _finalize(self):
        """Finalize the indexed dataset."""
        self.builder.finalize(f"{self.output_path}.idx")
        self.logger.info(f"Finalized output: {self.output_path}")