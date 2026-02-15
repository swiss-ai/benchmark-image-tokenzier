#!/usr/bin/env python3
"""
Unified Ray workers for distributed tokenization.
Handles all tokenization modes with a single flexible worker class.
"""

import time
from typing import Dict, Optional, Tuple, Union

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
            "dispatched": self.current_index,
            "total": self.num_shards,
            "in_progress": len(self.in_progress),
            "completed": len(self.completed),
            "failed": len(self.failed),
        }

    def get_final_summary(self) -> Dict:
        """Get final shard processing summary with completed/failed details."""
        return {
            "shards_completed": len(self.completed),
            "shards_failed": len(self.failed),
            "failed_shard_details": list(self.failed),
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
        resize_size: Union[int, Tuple[int, int], str] = "avg",
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

    def process_shard(self, shard_id: int, dataset_info: Dict, num_shards: int, progress_actor=None) -> Dict:
        """
        Process a complete HF dataset shard and save to a separate output file.

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

        # Load dataset shard
        dataset = load_hf_dataset(
            dataset_name=dataset_info["name"],
            config_name=dataset_info.get("config"),
            split=dataset_info["split"],
            cache_dir=dataset_info.get("cache_dir"),
            method=dataset_info.get("load_method", "default"),
            streaming=dataset_info["dataset_streamed"],
            data_files=dataset_info.get("data_files"),
        )
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

        log_interval = dataset_info.get("log_interval", 1000)
        self._process_shard_data(shard, builder, stats, log_interval, progress_actor)

        # Finalize the shard file
        builder.finalize(f"{shard_output_path}.idx")

        elapsed = time.time() - start_time
        self.logger.info(
            f"Completed shard {shard_id}: {stats['samples']} samples, " f"{stats['tokens']} tokens in {elapsed:.1f}s"
        )

        return {"shard_id": shard_id, "time": elapsed, **stats}