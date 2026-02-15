#!/usr/bin/env python3
"""
WebDataset tokenization pipeline.
Handles tar-based datasets with Ray distributed processing and shard-level checkpointing.
"""

import glob
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import ray

from vision_tokenization.pipelines.base import BasePipeline, ProgressActor
from vision_tokenization.pipelines.hf.workers import ShardQueue
from vision_tokenization.pipelines.wds.workers import WDSWorker
from vision_tokenization.vokenizers.transforms import create_transform_pipeline


class WDSPipeline(BasePipeline):
    """Pipeline for tokenizing WebDataset tar files."""

    def __init__(
        self,
        tokenizer_path: str,
        output_dir: str,
        input_pattern: str,
        mode: str,
        num_gpus: int,
        device: str,
        min_tokenizer_pixels: Optional[int] = None,
        max_tokenizer_pixels: Optional[int] = None,
        min_image_pixels: Optional[int] = None,
        max_image_pixels: Optional[int] = None,
        batch_size: int = 64,
        batch_mode: str = "sorted",
        buffer_size: Optional[int] = None,
        resize_size: Union[int, Tuple[int, int], str] = "avg",
        image_field: str = "jpg;png;jpeg;webp",
        text_field: str = "txt",
        resume: bool = False,
        image_transforms: Optional[str] = None,
        text_transforms: Optional[str] = None,
        transform_params: Optional[Dict[str, Dict[str, Any]]] = None,
        conversation_transform: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(tokenizer_path, output_dir, num_gpus, device, **kwargs)

        self.input_pattern = input_pattern
        self.mode = mode
        self.resume = resume

        # Set tokenizer pixels with defaults
        if min_tokenizer_pixels is None:
            if min_image_pixels is not None:
                min_tokenizer_pixels = min_image_pixels
                self.logger.info(f"Using min_image_pixels ({min_image_pixels:,} pixels) as min_tokenizer_pixels")
            else:
                min_tokenizer_pixels = 512 * 512
                self.logger.warning("No min_tokenizer_pixels provided, using default: 512*512 (262,144 pixels)")

        if max_tokenizer_pixels is None:
            max_tokenizer_pixels = 1024 * 1024
            self.logger.warning("No max_tokenizer_pixels provided, using default: 1024*1024 (1,048,576 pixels)")

        self.min_tokenizer_pixels = min_tokenizer_pixels
        self.max_tokenizer_pixels = max_tokenizer_pixels
        self.min_image_pixels = min_image_pixels
        self.max_image_pixels = max_image_pixels
        self.batch_size = batch_size
        self.batch_mode = batch_mode
        self.buffer_size = buffer_size
        self.resize_size = resize_size
        self.image_field = image_field
        self.text_field = text_field

        # Validate mode
        valid_modes = ["image_only", "image2text", "text2image", "sft"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

        # Create transform pipeline
        self.transform_pipeline = create_transform_pipeline(
            image_transforms=image_transforms, text_transforms=text_transforms, transform_params=transform_params
        )
        if self.transform_pipeline:
            self.logger.info(
                f"Transform pipeline configured with "
                f"image_transforms='{image_transforms}', text_transforms='{text_transforms}'"
            )

        self.conversation_transform = conversation_transform
        if self.conversation_transform:
            self.logger.info(f"Conversation transform configured: '{conversation_transform}'")

    def _discover_shards(self):
        """
        Discover tar shards from the input pattern.

        Supports both braceexpand patterns (e.g., 'data_{000..100}.tar')
        and standard glob patterns (e.g., '/path/to/*.tar').

        Returns:
            Sorted list of existing tar file paths
        """
        tar_paths = []

        # Try braceexpand first for patterns like {000..100}
        if "{" in self.input_pattern and ".." in self.input_pattern:
            try:
                import braceexpand

                expanded = list(braceexpand.braceexpand(self.input_pattern))
                # Filter to only existing files
                tar_paths = sorted([p for p in expanded if os.path.isfile(p)])
                if tar_paths:
                    self.logger.info(
                        f"Braceexpand: {len(expanded)} paths expanded, {len(tar_paths)} existing tar files found"
                    )
                    return tar_paths
                else:
                    self.logger.warning(
                        f"Braceexpand produced {len(expanded)} paths but none exist. Falling back to glob."
                    )
            except ImportError:
                self.logger.warning("braceexpand not installed, falling back to glob")
            except Exception as e:
                self.logger.warning(f"braceexpand failed ({e}), falling back to glob")

        # Fall back to glob
        tar_paths = sorted(glob.glob(self.input_pattern))
        if not tar_paths:
            raise FileNotFoundError(f"No tar files found matching pattern: {self.input_pattern}")

        self.logger.info(f"Glob: {len(tar_paths)} tar files found matching '{self.input_pattern}'")
        return tar_paths

    def _count_tar_samples(self, tar_paths: list) -> int:
        """
        Count total samples across tar files by reading tar headers only (no decompression).

        Each WDS sample is a group of files sharing the same base key (e.g., '000001.jpg'
        and '000001.txt' are one sample). We count unique keys per tar.

        Args:
            tar_paths: List of tar file paths

        Returns:
            Total number of samples across all tars
        """
        import tarfile
        import time

        start = time.time()
        total = 0
        for i, tar_path in enumerate(tar_paths):
            try:
                seen_keys = set()
                with tarfile.open(tar_path) as tf:
                    for member in tf:
                        if member.isfile():
                            # WDS key is filename without extension
                            key = member.name.rsplit(".", 1)[0] if "." in member.name else member.name
                            seen_keys.add(key)
                total += len(seen_keys)
            except Exception as e:
                self.logger.warning(f"Failed to count samples in {tar_path}: {e}")

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                self.logger.info(
                    f"Counting samples: {i + 1}/{len(tar_paths)} tars scanned "
                    f"({total:,} samples so far, {elapsed:.1f}s)"
                )

        elapsed = time.time() - start
        self.logger.info(f"Counted {total:,} samples across {len(tar_paths)} tars in {elapsed:.1f}s")
        return total

    def setup(self):
        """Discover shards, initialize Ray, and setup workers."""
        # Discover tar shards
        tar_paths = self._discover_shards()
        self.num_shards = len(tar_paths)
        self.shard_mapping = {i: path for i, path in enumerate(tar_paths)}

        self.logger.info(f"Discovered {self.num_shards} tar shards")
        if self.num_shards < self.num_gpus:
            self.logger.warning(
                f"num_shards ({self.num_shards}) < num_gpus ({self.num_gpus}). "
                f"Some workers will be idle."
            )

        # Count total samples by scanning tar headers (no decompression)
        self.dataset_size = self._count_tar_samples(tar_paths)

        # Initialize Ray
        self.logger.info(f"Initializing Ray with {self.num_gpus} workers")
        self.logger.info(f"DEBUG: RAY_ADDRESS env var = {os.environ.get('RAY_ADDRESS', 'NOT SET')}")
        self.logger.info(f"DEBUG: CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")

        if not ray.is_initialized():
            ray_address = os.environ.get("RAY_ADDRESS", None)
            if ray_address:
                self.logger.info(f"Connecting to existing Ray cluster at {ray_address}")
                ray.init(address="auto")
            else:
                self.logger.info(f"Starting local Ray cluster with {self.num_gpus} GPUs")
                ray.init(num_cpus=self.num_gpus + 2, num_gpus=self.num_gpus)

        resources = ray.cluster_resources()
        available = ray.available_resources()
        self.logger.info(f"DEBUG: Ray cluster resources: {resources}")
        self.logger.info(f"DEBUG: Ray available resources: {available}")

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self._setup_workers()

    def _setup_workers(self):
        """Setup ShardQueue, ProgressActor, and WDSWorker instances."""
        total_samples = self.dataset_size
        completed_samples = 0

        if self.resume:
            completed_shards = self._get_completed_shards()
            if completed_shards:
                self.logger.info(
                    f"[RESUME]: Found {len(completed_shards)} completed shards: {sorted(completed_shards)}"
                )

                # Count samples in completed shards by reading .idx files
                import re

                from vision_tokenization.pipelines.indexed_dataset_megatron import get_num_sequences

                output_path = Path(self.output_dir)
                for idx_file in output_path.glob("*.idx"):
                    match = re.match(r"rank_\d+_shard_(\d+)_\d+\.idx", idx_file.name)
                    if match:
                        shard_id = int(match.group(1))
                        if shard_id in completed_shards:
                            completed_samples += get_num_sequences(str(idx_file))

                remaining_samples = total_samples - completed_samples
                self.logger.info(
                    f"[RESUME]: Already processed {completed_samples:,} samples in {len(completed_shards)} shards. "
                    f"Remaining: {remaining_samples:,} samples"
                )
                total_samples = remaining_samples

                uncompleted = [i for i in range(self.num_shards) if i not in completed_shards]
                self.logger.info(
                    f"[RESUME]: Will process {len(uncompleted)} remaining shards: "
                    f"{uncompleted[:10]}{'...' if len(uncompleted) > 10 else ''}"
                )
                self.work_queue = ShardQueue.remote(self.num_shards, initial_shards=uncompleted)
            else:
                self.logger.info("[RESUME]: No completed shards found, starting from beginning")
                self.work_queue = ShardQueue.remote(self.num_shards)
        else:
            self.work_queue = ShardQueue.remote(self.num_shards)

        self.logger.info(f"Will process {total_samples:,} samples")

        log_interval = self.kwargs.get("log_interval", 1000)
        slurm_time_limit = self.kwargs.get("slurm_time_limit", None)
        self.progress_actor = ProgressActor.remote(
            total_samples,
            log_interval=log_interval,
            slurm_time_limit=slurm_time_limit,
        )

        self.workers = []
        for i in range(self.num_gpus):
            worker = WDSWorker.remote(
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
            )
            self.workers.append(worker)

        self.logger.info(
            f"Setup {self.num_gpus} workers for {self.mode} mode "
            f"with batch_size={self.batch_size}"
        )

    def process(self) -> Dict[str, Any]:
        """Process all tar shards using distributed workers."""
        dataset_info = {
            "shard_mapping": self.shard_mapping,
            "log_interval": self.kwargs.get("log_interval", 1000),
        }

        worker_futures = [
            worker.run_shards.remote(self.work_queue, dataset_info, self.num_shards, self.progress_actor)
            for worker in self.workers
        ]

        results = ray.get(worker_futures)

        # Get final progress (only reflects samples processed in this run, not resumed shards)
        current_run_processed = ray.get(self.progress_actor.close.remote())

        # Get shard-level summary from work queue
        shard_summary = ray.get(self.work_queue.get_final_summary.remote())

        # Read all shard stats (includes previously completed shards if resumed)
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

        max_time = max(r["elapsed_time"] for r in results) if results else 0

        self._save_metadata(all_shard_stats, shard_summary, results, max_time)

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
        """Save dataset processing metadata to JSON file."""
        import json

        total_samples = shard_stats["samples_processed"]
        total_tokens = shard_stats["tokens_generated"]
        total_errors = shard_stats["errors"]

        aggregated_result = {
            "worker_id": -1,
            "samples_processed": total_samples,
            "tokens_generated": total_tokens,
            "image_tokens": shard_stats["image_tokens"],
            "text_tokens": shard_stats["text_tokens"],
            "errors": total_errors,
            "samples_skipped": shard_stats["samples_skipped"],
            "resolution_skipped": shard_stats["resolution_skipped"],
            "transform_errors": shard_stats["transform_errors"],
            "cuda_oom_errors": shard_stats["cuda_oom_errors"],
            "elapsed_time": processing_time,
            "throughput_samples": total_samples / processing_time if processing_time > 0 else 0,
            "throughput_tokens": total_tokens / processing_time if processing_time > 0 else 0,
        }

        metadata = self.generate_metadata(
            results=[aggregated_result],
            processing_time=processing_time,
            dataset_type=f"{self.mode} tokenization (WebDataset)",
            input_pattern=self.input_pattern,
            mode=self.mode,
            image_field=self.image_field,
            text_field=self.text_field,
            shard_mapping={str(k): v for k, v in self.shard_mapping.items()},
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
                {"shard_id": shard_id, "error": error}
                for shard_id, error in shard_summary["failed_shard_details"]
            ]

        metadata_path = Path(self.output_dir) / "dataset_info.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved complete metadata to {metadata_path}")

def run_wds_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run WebDataset pipeline with configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Processing results
    """
    pipeline = WDSPipeline(**config)
    return pipeline.run()
