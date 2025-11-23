#!/usr/bin/env python3
"""
HuggingFace datasets tokenization pipeline.
Handles both image-only and SFT tokenization modes.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import ray

from vision_tokenization.pipelines.base import BasePipeline, ProgressActor
from vision_tokenization.pipelines.hf.workers import ShardQueue, Worker


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
        batch_size: Optional[int] = None,
        image_field: str = "images",
        text_field: str = "texts",
        **kwargs
    ):
        super().__init__(tokenizer_path, output_dir, num_gpus, device, **kwargs)

        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.config_name = config_name
        self.cache_dir = cache_dir
        self.num_proc = num_proc
        self.mode = mode
        self.num_shards = num_shards

        # Set tokenizer pixels with intelligent defaults
        # If min_image_pixels is set but min_tokenizer_pixels is not, use min_image_pixels
        if min_tokenizer_pixels is None:
            if min_image_pixels is not None:
                min_tokenizer_pixels = min_image_pixels
                self.logger.info(
                    f"Using min_image_pixels ({min_image_pixels:,} pixels) as min_tokenizer_pixels"
                )
            else:
                min_tokenizer_pixels = 512 * 512
                self.logger.warning(
                    "No min_tokenizer_pixels provided, using default: 512*512 (262,144 pixels)"
                )

        # Max tokenizer pixels always defaults to 1024*1024
        if max_tokenizer_pixels is None:
            max_tokenizer_pixels = 1024 * 1024
            self.logger.warning(
                "No max_tokenizer_pixels provided, using default: 1024*1024 (1,048,576 pixels)"
            )

        self.min_tokenizer_pixels = min_tokenizer_pixels
        self.max_tokenizer_pixels = max_tokenizer_pixels
        self.min_image_pixels = min_image_pixels
        self.max_image_pixels = max_image_pixels
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.image_field = image_field
        self.text_field = text_field

        # Validate mode
        valid_modes = ["image_only", "image2text", "text2image", "sft"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

        # Validate num_shards is sufficient for workers
        if self.num_shards < self.num_gpus:
            self.logger.warning(
                f"num_shards ({self.num_shards}) < num_gpus ({self.num_gpus}). "
                f"Adjusting num_shards to {self.num_gpus} to ensure all workers have work."
            )
            self.num_shards = self.num_gpus

    def setup(self):
        """Setup Ray and load dataset."""
        self.logger.info(f"Initializing Ray with {self.num_gpus} workers")

        # Initialize Ray
        if not ray.is_initialized():
            ray.init(num_cpus=self.num_gpus + 2)

        # Load dataset
        config_info = f" (config: {self.config_name})" if self.config_name else ""
        self.logger.info(f"Loading dataset: {self.dataset_name}{config_info}/{self.dataset_split}")
        from datasets import load_dataset
        self.dataset = load_dataset(
            self.dataset_name,
            name=self.config_name,
            split=self.dataset_split,
            cache_dir=self.cache_dir,
            num_proc=self.num_proc
        )

        if self.max_samples:
            self.dataset = self.dataset.select(range(min(self.max_samples, len(self.dataset))))

        self.logger.info(f"Processing {len(self.dataset)} samples")

        # Auto-create output subdirectory based on config_name and mode if config_name is provided
        if self.config_name:
            self.output_dir = str(Path(self.output_dir) / f"{self.config_name}_{self.mode}")

        # Add resolution subdirectory if filtering is enabled
        min_dims = self.kwargs.get('min_image_dims')
        max_dims = self.kwargs.get('max_image_dims')
        if min_dims or max_dims:
            parts = []
            if min_dims:
                parts.append(f"{min_dims[0]}x{min_dims[1]}")
            if max_dims:
                parts.append(f"{max_dims[0]}x{max_dims[1]}")
            res_dir = "_".join(parts)
            self.output_dir = str(Path(self.output_dir) / res_dir)

        self.logger.info(f"Output directory: {self.output_dir}")

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Setup workers
        self._setup_workers()

    def _setup_workers(self):
        """Setup workers for shard-based tokenization.

        Creates a work queue that distributes shards to workers. Each shard
        will be processed completely by a worker and saved as a separate output file.
        """
        # Create progress tracker
        self.progress_actor = ProgressActor.remote(len(self.dataset))

        # Create work queue for shard distribution
        self.work_queue = ShardQueue.remote(self.num_shards)

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
                image_field=self.image_field,
                text_field=self.text_field,
                min_image_pixels=self.min_image_pixels,
                max_image_pixels=self.max_image_pixels
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
            'name': self.dataset_name,
            'config': self.config_name,
            'split': self.dataset_split,
            'cache_dir': self.cache_dir,
            'total_samples': len(self.dataset)
        }

        # Start workers processing shards
        worker_futures = [worker.run_shards.remote(self.work_queue, dataset_info,
                                                   self.num_shards, self.progress_actor)
                         for worker in self.workers]

        # Wait for completion
        results = ray.get(worker_futures)

        # Get final progress
        total_processed = ray.get(self.progress_actor.close.remote())

        # Calculate summary statistics
        total_samples_processed = sum(r['samples_processed'] for r in results)
        total_tokens = sum(r['tokens_generated'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        max_time = max(r['elapsed_time'] for r in results)

        # Save metadata
        self._save_metadata(results, max_time)

        # Merge index files if needed
        self._merge_results()

        return {
            "total_processed": total_processed,
            "total_samples": total_samples_processed,
            "total_tokens": total_tokens,
            "total_errors": total_errors,
            "processing_time": max_time,
            "workers": len(self.workers),
            "mode": self.mode,
            "output_dir": self.output_dir,
            "metadata_file": str(Path(self.output_dir) / 'dataset_info.json')
        }

    def _merge_results(self):
        """Merge results from all workers."""
        # Implementation depends on the specific format
        # This would merge the individual worker outputs
        self.logger.info(f"Merging results in {self.output_dir}")

    def _save_metadata(self, results: list, processing_time: float):
        """Save dataset processing metadata to JSON file."""
        import json
        from pathlib import Path

        # Generate metadata using base pipeline method
        metadata = self.generate_metadata(
            results=results,
            processing_time=processing_time,
            dataset_type=f'{self.mode} tokenization',
            dataset_name=self.dataset_name,
            config_name=self.config_name,
            split=self.dataset_split,
            mode=self.mode,
            image_field=self.image_field,
            text_field=self.text_field,
            batch_size=self.batch_size,
            tokenizer={
                'path': self.tokenizer_path,
                'min_pixels': self.min_tokenizer_pixels,
                'max_pixels': self.max_tokenizer_pixels
            },
            image_filtering={
                'min_pixels': self.min_image_pixels,
                'max_pixels': self.max_image_pixels
            }
        )

        # Save to file
        metadata_path = Path(self.output_dir) / 'dataset_info.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved metadata to {metadata_path}")

    def cleanup(self):
        """Cleanup Ray resources."""
        if ray.is_initialized():
            ray.shutdown()


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