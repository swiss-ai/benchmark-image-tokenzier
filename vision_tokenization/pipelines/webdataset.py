#!/usr/bin/env python3
"""
WebDataset tokenization pipeline.
Handles webdataset format with parallel processing.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import ray

from vision_tokenization.vokenizers.emu import EMUImageOnlyTokenizer, EMUImageTextPairTokenizer

from .base import BasePipeline


class WebDatasetPipeline(BasePipeline):
    """Pipeline for tokenizing WebDataset format data."""

    def __init__(
        self,
        tokenizer_path: str,
        output_dir: str,
        input_pattern: str,
        mode: str = "image_text_pair",  # "image_only" or "image_text_pair"
        num_workers: int = 8,
        device: str = "cuda",
        min_pixels: int = 384 * 384,
        max_pixels: int = 1024 * 1024,
        batch_size: int = 64,
        **kwargs,
    ):
        """
        Initialize WebDataset pipeline.

        Args:
            tokenizer_path: Path to tokenizer
            output_dir: Output directory
            input_pattern: Pattern for input webdataset files (e.g., "data_{000..100}.tar")
            mode: Tokenization mode
            num_workers: Number of parallel workers
            device: Device for tokenization
            min_pixels: Minimum pixels for tokenizer
            max_pixels: Maximum pixels for tokenizer
            batch_size: Batch size for processing
            **kwargs: Additional arguments
        """
        super().__init__(tokenizer_path, output_dir, num_workers, device, **kwargs)

        self.input_pattern = input_pattern
        self.mode = mode
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.batch_size = batch_size

    def setup(self):
        """Setup Ray and workers."""
        self.logger.info(f"Initializing WebDataset pipeline")

        # Initialize Ray
        # Check if Ray should connect to existing cluster or start local
        if not ray.is_initialized():
            ray_address = os.environ.get('RAY_ADDRESS', None)

            if ray_address:
                # Multi-node mode: Connect to existing cluster
                self.logger.info(f"Connecting to existing Ray cluster at {ray_address}")
                ray.init(address='auto')  # Auto-detect from environment
            else:
                # Local mode: Start new cluster with explicit resources
                self.logger.info(f"Starting local Ray cluster with {self.num_workers} workers")
                ray.init(num_cpus=self.num_workers + 2)

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Setup workers
        if self.mode == "image_text_pair":
            self._setup_parallel_workers()
        else:
            self._setup_image_only_workers()

    def _setup_parallel_workers(self):
        """Setup workers for parallel image-text tokenization."""
        from vision_tokenization.core.webdataset_emu3_image_text_parallel import EMU3WebDatasetProcessor

        # Create processor
        self.processor = EMU3WebDatasetProcessor(
            tokenizer_path=self.tokenizer_path,
            output_dir=self.output_dir,
            num_workers=self.num_workers,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

    def _setup_image_only_workers(self):
        """Setup workers for image-only tokenization."""
        from vision_tokenization.core.webdataset_emu3_ray_dynamic_clean import setup_ray_workers

        # Setup Ray workers
        self.workers = setup_ray_workers(
            num_workers=self.num_workers,
            tokenizer_path=self.tokenizer_path,
            output_dir=self.output_dir,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

    def process(self) -> Dict[str, Any]:
        """Process the webdataset."""
        if self.mode == "image_text_pair":
            result = self.processor.process(self.input_pattern, self.batch_size)
        else:
            result = self._process_image_only()

        return {"mode": self.mode, "output_dir": self.output_dir, "num_workers": self.num_workers, **result}

    def _process_image_only(self) -> Dict[str, Any]:
        """Process image-only webdataset."""
        # Implementation for image-only processing
        total_processed = 0
        # Process with workers
        # ...
        return {"total_processed": total_processed}

    def cleanup(self):
        """Cleanup Ray resources."""
        if ray.is_initialized():
            ray.shutdown()


def run_webdataset_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run WebDataset pipeline with configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Processing results
    """
    pipeline = WebDatasetPipeline(**config)
    return pipeline.run()
