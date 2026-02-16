#!/usr/bin/env python3
"""
Ray workers for distributed WebDataset tokenization.
Handles tar-based shard processing with the same work-stealing pattern as HF workers.
"""

import time
from typing import Dict, Optional, Tuple, Union

import ray

from vision_tokenization.pipelines.base import BaseTokenizerWorker
from vision_tokenization.vokenizers.transforms import TransformPipeline


@ray.remote(num_gpus=1)
class WDSWorker(BaseTokenizerWorker):
    """
    WebDataset worker that extends BaseTokenizerWorker.
    Loads data from tar files instead of HuggingFace datasets.
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
        image_field: str = "jpg;png;jpeg;webp",
        text_field: str = "txt",
        min_image_pixels: Optional[int] = None,
        max_image_pixels: Optional[int] = None,
        transform_pipeline: Optional[TransformPipeline] = None,
        conversation_transform: Optional[str] = None,
    ):
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

        self.output_dir = output_dir

        # Parse semicolon-separated image/text field keys into fallback lists
        self.image_keys = [k.strip() for k in image_field.split(";")]
        self.text_keys = [k.strip() for k in text_field.split(";")]

    def _extract_data(self, sample: Dict) -> tuple:
        """
        Extract image and text from a WebDataset sample.

        WDS samples have extension-based keys (e.g., 'jpg', 'png', 'txt', 'json', '__key__').
        Tries image keys in fallback order. For text, tries text keys then falls back to
        'json' sidecar for SFT mode.
        """
        # Extract image: try keys in fallback order
        image = None
        for key in self.image_keys:
            if key in sample:
                image = sample[key]
                break

        # Extract text (only for modes that need it)
        text = None
        if self.mode in ["image2text", "text2image", "sft"]:
            for key in self.text_keys:
                if key in sample:
                    text = sample[key]
                    break

            # For SFT, fall back to json sidecar if no text key found
            if text is None and self.mode == "sft" and "json" in sample:
                import json

                json_data = sample["json"]
                if isinstance(json_data, str):
                    json_data = json.loads(json_data)
                text = json_data

        return image, text

    def _load_tar_shard(self, tar_path: str):
        """
        Load a WebDataset tar file and return an iterable of decoded samples.

        Args:
            tar_path: Path to the tar file

        Returns:
            Iterable of decoded samples with PIL images
        """
        import webdataset as wds

        dataset = wds.WebDataset(tar_path).decode("pil")
        return dataset

    def process_shard(self, shard_id: int, dataset_info: Dict, num_shards: int, progress_actor=None) -> Dict:
        """
        Process a complete tar shard and save to IndexedDataset output file.

        Args:
            shard_id: Index of the shard to process
            dataset_info: Must contain 'shard_mapping' dict (shard_id -> tar_path)
            num_shards: Total number of shards
            progress_actor: Optional progress tracking actor

        Returns:
            Processing statistics for this shard
        """
        tar_path = dataset_info["shard_mapping"][shard_id]
        self.logger.info(f"Processing shard {shard_id}/{num_shards}: {tar_path}")
        start_time = time.time()

        # Load tar shard
        shard = self._load_tar_shard(tar_path)

        # Create per-shard output file
        from pathlib import Path

        from vision_tokenization.pipelines.indexed_dataset_megatron import DType, IndexedDatasetBuilder

        shard_output_path = Path(self.output_dir) / f"rank_{self.worker_id}_shard_{shard_id}_{num_shards}"
        builder = IndexedDatasetBuilder(
            f"{shard_output_path}.bin", dtype=DType.optimal_dtype(len(self.tokenizer.text_tokenizer))
        )

        stats = {
            "samples": 0,
            "tokens": 0,
            "image_tokens": 0,
            "text_tokens": 0,
            "errors": 0,
            "skipped": 0,
            "resolution_skipped_min": 0,
            "resolution_skipped_max": 0,
            "transform_errors": 0,
            "cuda_oom_errors": 0,
        }

        log_interval = dataset_info.get("log_interval", 1000)
        self._process_shard_data(shard, builder, stats, log_interval, progress_actor)

        # Finalize the shard file
        builder.finalize(f"{shard_output_path}.idx")

        elapsed = time.time() - start_time
        self.logger.info(
            f"Completed shard {shard_id} ({tar_path}): {stats['samples']} samples, "
            f"{stats['tokens']} tokens in {elapsed:.1f}s"
        )

        return {"shard_id": shard_id, "time": elapsed, **stats}