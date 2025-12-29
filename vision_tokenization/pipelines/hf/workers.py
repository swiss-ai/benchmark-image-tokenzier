#!/usr/bin/env python3
"""
Unified Ray workers for distributed tokenization.
Handles all tokenization modes with a single flexible worker class.
"""

import time
from typing import Dict, Optional, Union, Tuple

import ray


@ray.remote
class ShardQueue:
    """Dynamic queue for distributing shards to workers."""

    def __init__(self, num_shards: int, initial_shards: Optional[list] = None):
        self.num_shards = num_shards
        if initial_shards is not None:
            # Resume mode: only process specified shards
            self.remaining_shards = list(initial_shards)
        else:
            # Normal mode: process all shards (0 to num_shards-1)
            self.remaining_shards = list(range(num_shards))
        self.current_index = 0  # Index to track position in remaining_shards
        self.in_progress = {}  # shard_id -> (worker_id, start_time)
        self.completed = []
        self.failed = []

    def get_next_shard(self, worker_id: int) -> Optional[int]:
        """Get next shard index for a worker (work-stealing)."""
        if self.current_index >= len(self.remaining_shards):
            return None

        shard_id = self.remaining_shards[self.current_index]
        self.current_index += 1
        self.in_progress[shard_id] = (worker_id, time.time())

        return shard_id

    def mark_completed(self, shard_id: int, stats: Dict):
        """Mark shard as completed."""
        if shard_id in self.in_progress:
            del self.in_progress[shard_id]
        self.completed.append((shard_id, stats))

    def mark_failed(self, shard_id: int, error: str):
        """Mark shard as failed."""
        if shard_id in self.in_progress:
            del self.in_progress[shard_id]
        self.failed.append((shard_id, error))

    def get_status(self) -> Dict:
        """Get current processing status."""
        return {
            "processed": self.next_shard,
            "total": self.num_shards,
            "in_progress": len(self.in_progress),
            "completed": len(self.completed),
            "failed": len(self.failed),
        }


from vision_tokenization.pipelines.base import BaseTokenizerWorker
from vision_tokenization.pipelines.hf.dataset_loader import load_hf_dataset, log_hf_environment_info
from vision_tokenization.vokenizers.transforms import TransformError, TransformPipeline


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
        batch_mode: Optional[str] = None,
        batch_size: int = 1,
        buffer_size: Optional[int] = None,
        resize_size: Union[int, Tuple[int, int], str] = 'avg',
        image_field: str = "image",
        text_field: str = "text",
        min_image_pixels: Optional[int] = None,
        max_image_pixels: Optional[int] = None,
        transform_pipeline: Optional[TransformPipeline] = None,
        conversation_transform: Optional[str] = None,
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
            transform_pipeline: Transform pipeline for image/text transforms (optional)
            conversation_transform: Conversation transform for SFT mode (optional)
        """
        # Initialize base tokenizer with resolution filtering and batching parameters
        super().__init__(
            tokenizer_path=tokenizer_path,
            worker_id=worker_id,
            mode=mode,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            batch_mode=batch_mode,
            batch_size=batch_size,
            buffer_size=buffer_size,
            resize_size=resize_size,
            image_field=image_field,
            text_field=text_field,
            min_image_pixels=min_image_pixels,
            max_image_pixels=max_image_pixels,
            transform_pipeline=transform_pipeline,
            conversation_transform=conversation_transform,
        )

        # Store output directory for per-shard files
        self.output_dir = output_dir

        # Log HuggingFace environment info (once per worker at initialization)
        log_hf_environment_info(self.logger, worker_id=self.worker_id)

    def batch_iterable_shard(self, shard, stats: Dict):
        """
        A generator which acts as an iterator over batches of a shard.

        Args:
            shard: The dataset shard
            stats (Dict): Stats tracker for tokenization

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

            # Check sample status and skip if necessary (missing field, out of range distribution)
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

    def process_shard(self, shard_id: int, dataset_info: Dict, num_shards: int, progress_actor=None) -> Dict:
        """
        Process a complete shard and save to a separate output file.

        Args:
            shard_id: Index of the shard to process
            dataset_info: Dataset metadata (includes 'log_interval' for progress updates)
            num_shards: Total number of shards
            progress_actor: Optional progress tracking actor for periodic updates

        Returns:
            Processing statistics for this shard
        """
        self.logger.info(f"Processing shard {shard_id}/{num_shards}")
        start_time = time.time()

        # Calculate progress update interval (half of log_interval, minimum 10)
        log_interval = dataset_info.get("log_interval", 1000)
        update_interval = max(log_interval // 2, 10)
        samples_since_update = 0  # Track samples since last progress update

        # Load the shard using HuggingFace's efficient shard method
        dataset = load_hf_dataset(
            dataset_name=dataset_info["name"],
            config_name=dataset_info.get("config"),
            split=dataset_info["split"],
            cache_dir=dataset_info.get("cache_dir"),
            method=dataset_info.get("load_method", "default"),
            streaming=dataset_info["dataset_streamed"],
        )

        # Get this specific shard
        shard = dataset.shard(num_shards=num_shards, index=shard_id)

        # Create per-shard output file with total shards in filename
        from pathlib import Path

        from vision_tokenization.pipelines.indexed_dataset_megatron import DType, IndexedDatasetBuilder

        shard_output_path = Path(self.output_dir) / f"rank_{self.worker_id}_shard_{shard_id}_{num_shards}"
        builder = IndexedDatasetBuilder(
            f"{shard_output_path}.bin", dtype=DType.optimal_dtype(len(self.tokenizer.text_tokenizer))
        )

        # Process all samples in the shard
        stats = {
            "samples": 0,
            "tokens": 0,
            "image_tokens": 0,
            "text_tokens": 0,
            "errors": 0,
            "skipped": 0,
            "resolution_skipped": 0,
            "transform_errors": 0,
            "cuda_oom_errors": 0,
        }

        # Use batching if configured, otherwise process samples individually
        if self.batcher is not None:
            # Batched processing
            import torch

            batch_loader = self.batch_iterable_shard(shard, stats)

            for batch in batch_loader:
                try:
                    images, text = batch["images"], batch["text"]
                    # Tokenize the batch - returns a list of variable-length sequences
                    tokens_batched = self.tokenize_batch(images, batch["resize_size"], text if text else None)

                    if tokens_batched is not None:
                        # Process list of sequences (works for both image-only and image-text pairs)
                        batch_size = len(tokens_batched)

                        # Collect all tokens and their lengths
                        all_tokens = []
                        lengths_list = []
                        for seq in tokens_batched:
                            seq_np = seq.cpu().numpy() if torch.is_tensor(seq) else seq
                            all_tokens.append(seq_np)
                            lengths_list.append(len(seq_np))

                        # Concatenate all sequences
                        import numpy as np
                        tokens_np = np.concatenate(all_tokens)

                        # Add to builder with individual lengths
                        builder.add_document(tokens_np, lengths_list)
                        stats["samples"] += batch_size
                        stats["tokens"] += len(tokens_np)
                        samples_since_update += batch_size

                        # Send periodic progress updates during shard processing
                        if progress_actor and samples_since_update >= update_interval:
                            progress_actor.update.remote(samples_since_update)
                            samples_since_update = 0  # Reset counter

                        # Count image vs text tokens for SFT mode
                        # Note: Each sample may have different numbers of image/text tokens
                        if self.mode == "sft" and self.img_end_id is not None:
                            for sample_tokens in all_tokens:
                                sample_tokens_list = sample_tokens.tolist()
                                if self.img_end_id in sample_tokens_list:
                                    img_end_idx = sample_tokens_list.index(self.img_end_id)
                                    stats["image_tokens"] += img_end_idx + 1
                                    stats["text_tokens"] += len(sample_tokens_list) - (img_end_idx + 1)

                        # Memory cleanup
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
            # Sample-by-sample processing (original behavior)
            for sample in shard:
                image, text = self._extract_data(sample)
                try:
                    image, text = self.apply_transforms(image, text)
                except Exception as e:
                    self.logger.warning(f"Transform error: {e}")
                    stats["transform_errors"] += 1
                    continue

                # Check sample status
                status = self.get_sample_status(image, text)

                if status == "resolution_skip":
                    stats["resolution_skipped"] += 1
                    continue
                elif status == "data_skip":
                    stats["skipped"] += 1
                    continue

                # Tokenize and save
                try:
                    tokens_np = self.tokenize_sample(image, text)
                    if tokens_np is not None:
                        builder.add_document(tokens_np, [len(tokens_np)])
                        stats["samples"] += 1
                        stats["tokens"] += len(tokens_np)
                        samples_since_update += 1

                        # Send periodic progress updates during shard processing
                        if progress_actor and samples_since_update >= update_interval:
                            progress_actor.update.remote(samples_since_update)
                            samples_since_update = 0  # Reset counter

                        # Count image vs text tokens for SFT mode
                        if self.mode == "sft" and self.img_end_id is not None:
                            import torch

                            if torch.is_tensor(tokens_np):
                                tokens_list = tokens_np.cpu().numpy().tolist()
                            else:
                                tokens_list = tokens_np.tolist()

                            if self.img_end_id in tokens_list:
                                img_end_idx = tokens_list.index(self.img_end_id)
                                stats["image_tokens"] += img_end_idx + 1
                                stats["text_tokens"] += len(tokens_list) - (img_end_idx + 1)
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

        # Finalize the shard file
        builder.finalize(f"{shard_output_path}.idx")

        # Send final progress update for any remaining samples
        if progress_actor and samples_since_update > 0:
            progress_actor.update.remote(samples_since_update)

        elapsed = time.time() - start_time
        self.logger.info(
            f"Completed shard {shard_id}: {stats['samples']} samples, " f"{stats['tokens']} tokens in {elapsed:.1f}s"
        )

        return {"shard_id": shard_id, "time": elapsed, **stats}

    def run_shards(self, shard_queue, dataset_info, num_shards, progress_actor=None) -> Dict:
        """
        Main worker loop for shard-based processing.

        Args:
            shard_queue: Ray remote shard queue for distribution
            dataset_info: Dataset metadata (includes 'log_interval' passed to process_shard)
            num_shards: Total number of shards
            progress_actor: Optional progress tracking actor

        Returns:
            Final worker statistics
        """
        self.logger.info("Starting shard processing loop")

        while True:
            # Get next shard from queue
            shard_id = ray.get(shard_queue.get_next_shard.remote(self.worker_id))

            if shard_id is None:
                self.logger.info("No more shards, finishing")
                break

            # Process the shard
            try:
                result = self.process_shard(shard_id, dataset_info, num_shards, progress_actor)
                ray.get(shard_queue.mark_completed.remote(shard_id, result))

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

        # Return final statistics
        return self.get_final_stats()
