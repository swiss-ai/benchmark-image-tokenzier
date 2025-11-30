#!/usr/bin/env python3
"""
Utility functions for loading HuggingFace datasets.
Supports both standard load_dataset() and dataset builder approaches.
"""

from typing import Optional
from datasets import Dataset, load_dataset, load_dataset_builder
import logging

logger = logging.getLogger(__name__)


def load_hf_dataset(
    dataset_name: str,
    config_name: Optional[str] = None,
    split: str = "train",
    cache_dir: Optional[str] = None,
    num_proc: Optional[int] = None,
    method: str = "default"
) -> Dataset:
    """
    Load a HuggingFace dataset using specified method.

    Args:
        dataset_name: HF dataset name (e.g., "HuggingFaceM4/FineVision")
        config_name: Dataset configuration/subset name (e.g., "lrv_chart")
        split: Dataset split to load (e.g., "train", "test", "validation")
        cache_dir: Cache directory for dataset files
        num_proc: Number of processes (only used with "default" method)
        method: Loading method - "default" or "builder_load"

    Returns:
        Dataset object supporting len(), .shard(), .select(), iteration

    Raises:
        ValueError: If method is not "default" or "builder_load"
        FileNotFoundError: If "builder_load" is used but dataset is not prepared
    """
    if method not in ["default", "builder_load"]:
        raise ValueError(
            f"Invalid method: {method}. Must be 'default' or 'builder_load'"
        )

    if method == "default":
        config_info = f" (config: {config_name})" if config_name else ""
        logger.info(f"Loading dataset using load_dataset(): {dataset_name}{config_info}/{split}")

        dataset = load_dataset(
            dataset_name,
            name=config_name,
            split=split,
            cache_dir=cache_dir,
            num_proc=num_proc
        )

    elif method == "builder_load":
        config_info = f" (config: {config_name})" if config_name else ""
        logger.info(f"Loading dataset using builder.as_dataset(): {dataset_name}{config_info}/{split}")

        # Warn if num_proc provided (not supported)
        if num_proc is not None:
            logger.warning(
                f"num_proc parameter ({num_proc}) is ignored when using 'builder_load' method. "
                f"Dataset is already prepared."
            )

        # Warn if cache_dir not provided
        if cache_dir is None:
            logger.warning(
                "Using 'builder_load' without explicit cache_dir. "
                "Will use default: ~/.cache/huggingface/datasets"
            )

        builder = load_dataset_builder(dataset_name, name=config_name, cache_dir=cache_dir)

        try:
            dataset = builder.as_dataset(split=split)
        except FileNotFoundError as e:
            logger.error(
                f"Dataset '{dataset_name}' is not prepared. "
                f"When using 'builder_load', run download_and_prepare() first."
            )
            logger.error(
                f"To prepare:\n"
                f"  from datasets import load_dataset_builder\n"
                f"  builder = load_dataset_builder('{dataset_name}'"
                f"{f', name={config_name!r}' if config_name else ''}"
                f"{f', cache_dir={cache_dir!r}' if cache_dir else ''})\n"
                f"  builder.download_and_prepare()"
            )
            raise FileNotFoundError(
                f"Dataset not prepared. Use method='default' or prepare dataset first."
            ) from e

    logger.info(f"Loaded {len(dataset)} samples from {split} split")
    return dataset