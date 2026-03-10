#!/usr/bin/env python3
"""
HuggingFace datasets tokenization pipeline.
Handles both image-only and SFT tokenization modes.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import ray

from vision_tokenization.pipelines.base import BasePipeline, ProgressActor
from vision_tokenization.pipelines.hf.dataset_loader import (
    _convert_percentage_to_absolute,
    _parse_split_slice,
    check_memory_mapping_limits,
    get_builder_split_info,
    load_hf_dataset,
    log_hf_environment_info,
)
from vision_tokenization.pipelines.hf.workers import ShardQueue, Worker
from vision_tokenization.vokenizers.transforms import create_transform_pipeline


class HFDatasetPipeline(BasePipeline):
    """Pipeline for tokenizing HuggingFace datasets."""

    def __init__(
        self,
        tokenizer_path: str,
        output_dir: str,
        dataset_name: str,
        dataset_split: str,
        mode: str,  # "image_only", "image2text", "text2image", or "sft"
        num_gpus: int,
        device: str,
        num_shards: int,  # Required for checkpointing
        config_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        num_proc: int = 32,
        min_tokenizer_pixels: Optional[int] = None,
        max_tokenizer_pixels: Optional[int] = None,
        min_image_pixels: Optional[int] = None,
        max_image_pixels: Optional[int] = None,
        max_samples: Optional[int] = None,
        batch_size: int = 64,
        batch_mode: str = "sorted",
        buffer_size: Optional[int] = None,
        resize_size: Union[int, Tuple[int, int], str] = "avg",
        image_field: str = "images",
        text_field: str = "texts",
        resume: bool = False,
        image_transforms: Optional[str] = None,
        text_transforms: Optional[str] = None,
        transform_params: Optional[Dict[str, Dict[str, Any]]] = None,
        conversation_transform: Optional[str] = None,
        dataset_load_method: str = "default",
        dataset_streamed: bool = False,
        data_files: Optional[str] = None,
        data_dir: Optional[str] = None,
        image_field_pattern: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(tokenizer_path, output_dir, num_gpus, device, **kwargs)

        self.image_field_pattern = image_field_pattern
        if self.image_field_pattern:
            self.logger.info(f"Multi-image mode enabled with pattern: '{self.image_field_pattern}'")

        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.config_name = config_name
        self.cache_dir = cache_dir
        self.num_proc = num_proc
        self.mode = mode
        self.num_shards = num_shards
        self.dataset_load_method = dataset_load_method
        self.dataset_streamed = dataset_streamed
        self.data_files = data_files
        self.data_dir = data_dir

        # Set tokenizer pixels with intelligent defaults
        # If min_image_pixels is set but min_tokenizer_pixels is not, use min_image_pixels
        if min_tokenizer_pixels is None:
            if min_image_pixels is not None:
                min_tokenizer_pixels = min_image_pixels
                self.logger.info(f"Using min_image_pixels ({min_image_pixels:,} pixels) as min_tokenizer_pixels")
            else:
                min_tokenizer_pixels = 512 * 512
                self.logger.warning("No min_tokenizer_pixels provided, using default: 512*512 (262,144 pixels)")

        # Max tokenizer pixels always defaults to 1024*1024
        if max_tokenizer_pixels is None:
            max_tokenizer_pixels = 1024 * 1024
            self.logger.warning("No max_tokenizer_pixels provided, using default: 1024*1024 (1,048,576 pixels)")

        self.min_tokenizer_pixels = min_tokenizer_pixels
        self.max_tokenizer_pixels = max_tokenizer_pixels
        self.min_image_pixels = min_image_pixels
        self.max_image_pixels = max_image_pixels
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.batch_mode = batch_mode
        self.buffer_size = buffer_size
        self.resize_size = resize_size
        self.image_field = image_field
        self.text_field = text_field
        self.resume = resume

        # Validate mode
        valid_modes = ["image_only", "image2text", "text2image", "sft"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

        # For webdataset with streaming, cap num_shards to the number of tar files
        # since IterableDataset.shard() distributes underlying files, not samples
        if self.dataset_streamed and self.data_files:
            import glob as glob_mod

            parts = [p.strip() for p in self.data_files.split(",")]
            num_files = sum(len(glob_mod.glob(p)) for p in parts)
            if num_files > 0 and self.num_shards > num_files:
                self.logger.warning(
                    f"num_shards ({self.num_shards}) > number of data files ({num_files}). "
                    f"Capping num_shards to {num_files} to avoid empty shards."
                )
                self.num_shards = num_files

        # Validate num_shards is sufficient for workers
        if self.num_shards < self.num_gpus:
            self.logger.warning(
                f"num_shards ({self.num_shards}) < num_gpus ({self.num_gpus}). "
                f"Adjusting num_shards to {self.num_gpus} to ensure all workers have work."
            )
            self.num_shards = self.num_gpus

        # Create transform pipeline if transforms are configured
        self.transform_pipeline = create_transform_pipeline(
            image_transforms=image_transforms, text_transforms=text_transforms, transform_params=transform_params
        )
        if self.transform_pipeline:
            self.logger.info(
                f"Transform pipeline configured with "
                f"image_transforms='{image_transforms}', text_transforms='{text_transforms}'"
            )

        # Store conversation transform configuration (will be passed to workers)
        self.conversation_transform = conversation_transform
        if self.conversation_transform:
            self.logger.info(f"Conversation transform configured: '{conversation_transform}'")

    def setup(self):
        """Setup Ray and load dataset."""
        self.logger.info(f"Initializing Ray with {self.num_gpus} workers")

        # DEBUG: Log environment
        self.logger.info(f"DEBUG: RAY_ADDRESS env var = {os.environ.get('RAY_ADDRESS', 'NOT SET')}")
        self.logger.info(f"DEBUG: CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
        self.logger.info(f"DEBUG: NUM_GPUS env var = {os.environ.get('NUM_GPUS', 'NOT SET')}")
        self.logger.info(f"DEBUG: self.num_gpus = {self.num_gpus}")

        # Initialize Ray with GPU support
        # Check if Ray should connect to existing cluster or start local
        if not ray.is_initialized():
            ray_address = os.environ.get("RAY_ADDRESS", None)

            if ray_address:
                # Multi-node mode: Connect to existing cluster
                self.logger.info(f"Connecting to existing Ray cluster at {ray_address}")
                self.logger.info(f"DEBUG: Using multinode mode - will call ray.init(address='auto')")
                ray.init(address="auto")  # Auto-detect from environment
                self.logger.info(f"DEBUG: Connected to Ray cluster")
            else:
                # Local mode: Start new cluster with explicit resources
                self.logger.info(f"Starting local Ray cluster with {self.num_gpus} GPUs")
                self.logger.info(f"DEBUG: Using local mode - will call ray.init(num_gpus={self.num_gpus})")
                ray.init(num_cpus=self.num_gpus + 2, num_gpus=self.num_gpus)
                self.logger.info(f"DEBUG: Started local Ray cluster")

        # DEBUG: Log Ray cluster state
        resources = ray.cluster_resources()
        available = ray.available_resources()
        self.logger.info(f"DEBUG: Ray cluster resources: {resources}")
        self.logger.info(f"DEBUG: Ray available resources: {available}")
        self.logger.info(f"DEBUG: Total GPUs in cluster: {resources.get('GPU', 0)}")
        self.logger.info(f"DEBUG: Available GPUs: {available.get('GPU', 0)}")

        log_hf_environment_info(self.logger)

        # Get dataset size without loading the full dataset
        if self.dataset_streamed:
            self.logger.info("Processing dataset in streaming mode (size unknown upfront)")
            self.dataset_size = -1
        else:
            # For non-streaming datasets, use get_builder_split_info to get metadata without loading
            self.logger.info("Getting dataset information without loading full dataset...")

            split_info = get_builder_split_info(
                dataset_name=self.dataset_name,
                config_name=self.config_name,
                cache_dir=self.cache_dir,
                dataset_load_method=self.dataset_load_method,
                data_files=self.data_files,
                data_dir=self.data_dir,
            )

            if split_info is not None:
                # Parse split string to handle slice notation (e.g., "train[:100]" or "train[:50%]")
                base_split, start, end, start_is_pct, end_is_pct = _parse_split_slice(self.dataset_split)

                # Check if base split exists in split_info
                if base_split not in split_info:
                    available_splits = list(split_info.keys())
                    raise ValueError(
                        f"Split '{base_split}' not found in dataset. " f"Available splits: {available_splits}"
                    )

                # Get total number of examples for this split
                num_examples = split_info[base_split]["num_examples"]

                # Apply slice bounds if present in split string
                if start is not None or end is not None:
                    # Convert percentages to absolute indices
                    abs_start = _convert_percentage_to_absolute(start, start_is_pct, num_examples)
                    abs_end = _convert_percentage_to_absolute(end, end_is_pct, num_examples)

                    effective_start = abs_start if abs_start is not None else 0
                    effective_end = abs_end if abs_end is not None else num_examples

                    num_examples = effective_end - effective_start
                    self.logger.info(
                        f"Split slice '{self.dataset_split}' will process "
                        f"{num_examples:,} samples (from {effective_start} to {effective_end})"
                    )

                # Apply max_samples limit if specified
                if self.max_samples:
                    original_num = num_examples
                    num_examples = min(self.max_samples, num_examples)
                    if num_examples < original_num:
                        self.logger.info(
                            f"Limiting to {num_examples:,} samples due to max_samples parameter "
                            f"(dataset has {original_num:,} samples)"
                        )

                self.logger.info("Checking system memory mapping limits...")
                check_memory_mapping_limits(split_info, self.dataset_split, self.dataset_streamed)

                self.dataset_size = num_examples
                self.logger.info(f"Will process {num_examples:,} samples")

            # Load dataset once on head node to:
            # 1. Trigger download_and_prepare() so Arrow cache is ready before workers start
            #    (avoids concurrent preparation race conditions on shared filesystems)
            # 2. Sanity check: verify the first sample is accessible and readable
            # 3. Determine dataset size when builder split info was not available
            self.logger.info("Loading dataset for preparation and sanity check...")
            dataset = load_hf_dataset(
                dataset_name=self.dataset_name,
                config_name=self.config_name,
                split=self.dataset_split,
                cache_dir=self.cache_dir,
                method=self.dataset_load_method,
                streaming=False,
                data_files=self.data_files,
                data_dir=self.data_dir,
            )

            if split_info is None:
                # Builder couldn't determine split info (e.g. webdataset from local tar files).
                # Now that we've loaded, we can get the actual size.
                num_examples = len(dataset)
                self.logger.info(
                    f"Split metadata was not available from builder. "
                    f"Determined dataset size after loading: {num_examples:,} samples"
                )
                self.dataset_size = num_examples
            else:
                self.logger.info("Dataset Arrow cache is ready.")

            # Sanity check: access first sample to verify data is readable
            self._sanity_check_sample(dataset)

            # Clean up — workers will mmap the cached Arrow files independently
            del dataset

        # Auto-create output subdirectory based on config_name and mode if config_name is provided
        if self.config_name:
            self.output_dir = str(Path(self.output_dir) / f"{self.config_name}_{self.mode}")

        # Add resolution subdirectory if filtering is enabled
        min_dims = self.kwargs.get("min_image_dims")
        max_dims = self.kwargs.get("max_image_dims")
        if min_dims or max_dims:
            parts = []
            if min_dims:
                parts.append(f"{min_dims[0]}x{min_dims[1]}")
            if max_dims:
                parts.append(f"{max_dims[0]}x{max_dims[1]}")
            res_dir = "_".join(parts)
            self.output_dir = str(Path(self.output_dir) / res_dir)

        self.logger.info(f"Output directory: {self.output_dir}")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self._setup_workers()

    def _sanity_check_sample(self, dataset):
        """Access the first sample to verify the dataset is readable."""
        try:
            sample = dataset[0]
            self.logger.info(f"Sanity check passed. Columns: {list(sample.keys())}")
        except Exception as e:
            raise RuntimeError(f"Sanity check failed: could not access first sample. Error: {e}") from e

    def _setup_workers(self):
        """Setup workers for shard-based tokenization.

        Creates a work queue that distributes shards to workers. Each shard
        will be processed completely by a worker and saved as a separate output file.
        """
        # Calculate remaining samples if resuming
        total_samples = self.dataset_size
        completed_samples = 0

        # Check for existing completed shards if resuming
        if self.resume:
            completed_shards = self._get_completed_shards()
            if completed_shards:
                self.logger.info(
                    f"[RESUME]: Found {len(completed_shards)} completed shards: {sorted(completed_shards)}"
                )

                # Count samples in completed shards by reading .idx files
                import re

                output_path = Path(self.output_dir)
                from vision_tokenization.pipelines.indexed_dataset_megatron import get_num_sequences

                for idx_file in output_path.glob("*.idx"):
                    # Match pattern: rank_X_shard_Y_Z.idx
                    match = re.match(r"rank_\d+_shard_(\d+)_\d+\.idx", idx_file.name)
                    if match:
                        shard_id = int(match.group(1))
                        if shard_id in completed_shards:
                            num_sequences = get_num_sequences(str(idx_file))
                            completed_samples += num_sequences

                if self.dataset_size > 0:
                    remaining_samples = self.dataset_size - completed_samples
                    self.logger.info(
                        f"[RESUME]: Already processed {completed_samples:,} samples in {len(completed_shards)} shards. "
                        f"Remaining: {remaining_samples:,} samples"
                    )
                    total_samples = remaining_samples
                else:
                    self.logger.info(
                        f"[RESUME]: Already processed {completed_samples:,} samples in {len(completed_shards)} shards. "
                        f"Remaining: unknown (dataset size not available upfront)"
                    )

                # Create work queue with only uncompleted shards
                uncompleted = [i for i in range(self.num_shards) if i not in completed_shards]
                self.logger.info(
                    f"[RESUME]: Will process {len(uncompleted)} remaining shards: {uncompleted[:10]}{'...' if len(uncompleted) > 10 else ''}"
                )
                self.work_queue = ShardQueue.remote(self.num_shards, initial_shards=uncompleted)
            else:
                self.logger.info("[RESUME]: No completed shards found, starting from beginning")
                self.work_queue = ShardQueue.remote(self.num_shards)
        else:
            # Create work queue for shard distribution
            self.work_queue = ShardQueue.remote(self.num_shards)

        # Create progress tracker with adjusted sample count
        log_interval = self.kwargs.get("log_interval", 1000)
        slurm_time_limit = self.kwargs.get("slurm_time_limit", None)
        self.progress_actor = ProgressActor.remote(
            total_samples,  # Now uses remaining samples in resume mode
            log_interval=log_interval,
            slurm_time_limit=slurm_time_limit,
        )

        # Start unified workers
        self.workers = []
        for i in range(self.num_gpus):
            worker = Worker.remote(
                tokenizer_path=self.tokenizer_path,
                output_dir=self.output_dir,
                worker_id=i,
                mode=self.mode,
                min_pixels=self.min_tokenizer_pixels,
                max_pixels=self.max_tokenizer_pixels,
                batch_mode=self.batch_mode,
                batch_size=self.batch_size,
                buffer_size=self.buffer_size,
                resize_size=self.resize_size,
                image_field=self.image_field,
                text_field=self.text_field,
                min_image_pixels=self.min_image_pixels,
                max_image_pixels=self.max_image_pixels,
                transform_pipeline=self.transform_pipeline,
                conversation_transform=self.conversation_transform,
                image_field_pattern=self.image_field_pattern,
            )
            self.workers.append(worker)

        self.logger.info(
            f"Setup {self.num_gpus} workers for {self.mode} mode "
            f"with batch_size={self.batch_size} (for I/O optimization)"
        )

    def process(self) -> Dict[str, Any]:
        """Process the dataset using shard-based approach."""
        # Create dataset info dict to pass to workers
        dataset_info = {
            "name": self.dataset_name,
            "config": self.config_name,
            "split": self.dataset_split,
            "cache_dir": self.cache_dir,
            "total_samples": self.dataset_size,
            "load_method": self.dataset_load_method,
            "dataset_streamed": self.dataset_streamed,
            "data_files": self.data_files,
            "data_dir": self.data_dir,
            "log_interval": self.kwargs.get("log_interval", 1000),
        }

        # Start workers processing shards
        worker_futures = [
            worker.run_shards.remote(self.work_queue, dataset_info, self.num_shards, self.progress_actor)
            for worker in self.workers
        ]

        # Wait for completion
        results = ray.get(worker_futures)

        # Get final progress (only reflects samples processed in this run, not resumed shards)
        current_run_processed = ray.get(self.progress_actor.close.remote())

        # Get shard-level summary from work queue
        shard_summary = ray.get(self.work_queue.get_final_summary.remote())

        # At the end, read ALL shard stats from shard_stats/ directory
        # This accumulates stats from both old shards (if resumed) and newly completed shards
        self.logger.info("Reading all shard statistics files...")
        all_shard_stats = self._load_completed_shard_stats()

        shards_incomplete = self.num_shards - all_shard_stats["shard_count"]
        self.logger.info(
            f"Final statistics ({all_shard_stats['shard_count']}/{self.num_shards} shards completed"
            f"{f', {shards_incomplete} incomplete' if shards_incomplete > 0 else ''}):\n"
            f"  - Total samples: {all_shard_stats['samples_processed']:,}\n"
            f"  - Total tokens: {all_shard_stats['tokens_generated']:,}\n"
            f"  - Total errors: {all_shard_stats['errors']}\n"
            f"  - This run: {shard_summary['shards_completed']} completed, "
            f"{shard_summary['shards_failed']} failed"
        )

        if shard_summary["shards_failed"] > 0:
            for shard_id, error in shard_summary["failed_shard_details"]:
                self.logger.error(f"  Failed shard {shard_id}: {error}")

        # Calculate max time from worker results for overall timing
        valid_results = [r for r in results if r is not None]
        max_time = max(r["elapsed_time"] for r in valid_results) if valid_results else 0

        # Save final metadata with complete accumulated statistics
        self._save_metadata(all_shard_stats, shard_summary, valid_results, max_time)

        return {
            "current_run_processed": current_run_processed,
            "total_samples": all_shard_stats["samples_processed"],
            "total_tokens": all_shard_stats["tokens_generated"],
            "total_errors": all_shard_stats["errors"],
            "shards_completed_total": all_shard_stats["shard_count"],
            "shards_incomplete_total": shards_incomplete,
            "current_run_shards_completed": shard_summary["shards_completed"],
            "current_run_shards_failed": shard_summary["shards_failed"],
            "processing_time": max_time,
            "workers": len(self.workers),
            "mode": self.mode,
            "output_dir": self.output_dir,
            "metadata_file": str(Path(self.output_dir) / "dataset_info.json"),
        }

    def _save_metadata(
        self,
        shard_stats: Dict[str, Any],
        shard_summary: Dict[str, Any],
        worker_results: list,
        processing_time: float,
    ):
        """Save dataset processing metadata to JSON file.

        Args:
            shard_stats: Accumulated statistics from all shard_stats/*.json files
            shard_summary: Shard completion/failure summary from ShardQueue
            worker_results: Results from workers (for additional context)
            processing_time: Overall processing time
        """
        import json
        from pathlib import Path

        # Use accumulated shard statistics for the metadata
        total_samples = shard_stats["samples_processed"]
        total_tokens = shard_stats["tokens_generated"]
        total_errors = shard_stats["errors"]

        # Create synthetic "results" format that generate_metadata expects
        # with a single aggregated entry from all shards
        aggregated_result = {
            "worker_id": -1,  # Special marker for aggregated stats
            "samples_processed": total_samples,
            "tokens_generated": total_tokens,
            "image_tokens": shard_stats["image_tokens"],
            "text_tokens": shard_stats["text_tokens"],
            "errors": total_errors,
            "samples_skipped": shard_stats["samples_skipped"],
            "resolution_skipped_min": shard_stats["resolution_skipped_min"],
            "resolution_skipped_max": shard_stats["resolution_skipped_max"],
            "transform_errors": shard_stats["transform_errors"],
            "cuda_oom_errors": shard_stats["cuda_oom_errors"],
            "elapsed_time": processing_time,
            "throughput_samples": total_samples / processing_time if processing_time > 0 else 0,
            "throughput_tokens": total_tokens / processing_time if processing_time > 0 else 0,
        }

        # Generate metadata using base pipeline method with aggregated stats
        metadata = self.generate_metadata(
            results=[aggregated_result],
            processing_time=processing_time,
            dataset_type=f"{self.mode} tokenization",
            dataset_name=self.dataset_name,
            config_name=self.config_name,
            split=self.dataset_split,
            mode=self.mode,
            image_field=self.image_field,
            text_field=self.text_field,
            batching={
                "batch_size": self.batch_size,
                "batch_mode": self.batch_mode,
                "buffer_size": self.buffer_size,
                "resize_size": self.resize_size,
            },
            tokenizer={
                "path": self.tokenizer_path,
                "min_pixels": self.min_tokenizer_pixels,
                "max_pixels": self.max_tokenizer_pixels,
            },
            image_filtering={"min_pixels": self.min_image_pixels, "max_pixels": self.max_image_pixels},
        )

        # Add per-shard details to metadata
        metadata["per_shard_details"] = shard_stats["per_shard"]
        metadata["processing"]["num_shards"] = self.num_shards
        # All-time count: derived from shard_stats files on disk (includes resumed shards)
        metadata["processing"]["shards_completed_total"] = shard_stats["shard_count"]
        metadata["processing"]["shards_incomplete_total"] = self.num_shards - shard_stats["shard_count"]
        # Current run counts: from the ShardQueue (this run only)
        metadata["processing"]["current_run"] = {
            "shards_completed": shard_summary["shards_completed"],
            "shards_failed": shard_summary["shards_failed"],
        }
        if shard_summary["failed_shard_details"]:
            metadata["processing"]["current_run"]["failed_shard_details"] = [
                {"shard_id": shard_id, "error": error} for shard_id, error in shard_summary["failed_shard_details"]
            ]

        # Save to file
        metadata_path = Path(self.output_dir) / "dataset_info.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved complete metadata to {metadata_path}")


def run_hf_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run HF dataset pipeline with configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Processing results
    """
    pipeline = HFDatasetPipeline(**config)
    return pipeline.run()
