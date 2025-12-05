#!/usr/bin/env python3
"""
Utility functions for loading HuggingFace datasets.
Supports both standard load_dataset() and dataset builder approaches.

Also contains methods to lookup num dataset samples based on the load_dataset_builder approach which does not map files to
virtual memory -> Works even if dataset exceeds limits of memory mapping.
"""

import logging
import re
from typing import Optional, Tuple, Dict

from datasets import Dataset, load_dataset, load_dataset_builder

logger = logging.getLogger(__name__)


def get_system_max_map_count() -> Optional[int]:
    """
    Get the system's maximum number of memory map areas (vm.max_map_count).
    return None if not found!
    """
    try:
        with open('/proc/sys/vm/max_map_count', 'r') as f:
            max_map_count = int(f.read().strip())
            return max_map_count
    except (FileNotFoundError, PermissionError, ValueError) as e:
        logger.debug(f"Could not read max_map_count: {e}")
        return None


def get_builder_split_info(
    dataset_name: str,
    config_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Load dataset builder and extract split information without loading the dataset.

    Args:
        dataset_name: HF dataset name (e.g., "HuggingFaceM4/FineVision")
        config_name: Dataset configuration/subset name (e.g., "lrv_chart")
        cache_dir: Cache directory for dataset files

    Returns:
        Dictionary mapping split names to their info:
        {
            "train": {"num_examples": 1000, "num_shards": 4},
            "validation": {"num_examples": 100, "num_shards": 1},
            ...
        }
    """
    config_info = f" (config: {config_name})" if config_name else ""
    logger.info(f"Loading builder for: {dataset_name}{config_info}")

    if cache_dir is None:
        logger.warning(
            "No explicit cache_dir provided. " "Will use default: ~/.cache/huggingface/datasets"
        )

    builder = load_dataset_builder(dataset_name, name=config_name, cache_dir=cache_dir)
    info = builder.info

    logger.info(f"=== Builder Split Info ===")

    split_info = {}
    for split_name, split_data in info.splits.items():
        # Extract num_examples and num_shards
        num_examples = split_data.num_examples
        num_shards = len(getattr(split_data, 'shard_lengths', [])) or None

        split_info[split_name] = {
            "num_examples": num_examples,
            "num_shards": num_shards,
        }

        logger.info(
            f"  Split '{split_name}': {num_examples:,} examples"
            + (f", {num_shards} shard(s)" if num_shards is not None else "")
        )

    return split_info


def check_memory_mapping_limits(split_info: Dict[str, Dict[str, int]], split: str) -> None:
    """
    Check if system memory mapping limits are sufficient for the dataset.

    Args:
        split_info: Dictionary with split information from get_builder_split_info()
        split: The split being loaded (may include slice notation like "train[:100]")
    """
    # Parse split to get base split name (without slice notation)
    base_split, _, _ = _parse_split_slice(split)

    # Get number of shards for this split
    if base_split not in split_info:
        logger.warning(f"Split '{base_split}' not found in dataset info. Skipping memory mapping check.")
        return

    num_shards = split_info[base_split].get("num_shards")

    if num_shards is None:
        logger.debug(f"Number of shards not available for split '{base_split}'. Skipping memory mapping check.")
        return

    # Get system max_map_count
    max_map_count = get_system_max_map_count()

    if max_map_count is None:
        logger.warning(
            f"⚠️  WARNING: Could not determine system's max memory mappings (vm.max_map_count). "
            f"Dataset split '{base_split}' has {num_shards} shard(s). "
        )
        return

    # Check if max_map_count is sufficient
    if max_map_count < num_shards:
        logger.warning(
            f"⚠️  WARNING: System max memory mappings ({max_map_count:,}) is lower than "
            f"the number of shards ({num_shards}) in split '{base_split}'. -> LOADING ISSUES POSSIBLE! "
        )
    else:
        logger.info(
            f"✓ System max memory mappings ({max_map_count:,}) is sufficient "
            f"for {num_shards} shard(s) in split '{base_split}'"
        )

def _parse_split_slice(split: str) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Parse split string that may contain slice notation.

    Args:
        split: Split string (e.g., "train", "train[:100]", "train[100:200]")

    Returns:
        Tuple of (base_split, start, end) where start/end are None if no slice

    Examples:
        "train" -> ("train", None, None)
        "train[:100]" -> ("train", None, 100)
        "train[100:200]" -> ("train", 100, 200)
        "train[100:]" -> ("train", 100, None)
    """
    # Pattern to match split[start:end] syntax
    match = re.match(r"^(\w+)(?:\[(\d*):(\d*)\])?$", split)
    if not match:
        # No slice notation or invalid format
        return split, None, None

    base_split = match.group(1)
    start_str = match.group(2)
    end_str = match.group(3)

    # Convert to int if present, otherwise None
    start = int(start_str) if start_str else None
    end = int(end_str) if end_str else None

    return base_split, start, end


def load_hf_dataset(
    dataset_name: str,
    config_name: Optional[str] = None,
    split: str = "train",
    cache_dir: Optional[str] = None,
    num_proc: Optional[int] = None,
    method: str = "default",
    streaming: bool = False,
) -> Dataset:
    """
    Load a HuggingFace dataset using specified method.

    Args:
        dataset_name: HF dataset name (e.g., "HuggingFaceM4/FineVision")
        config_name: Dataset configuration/subset name (e.g., "lrv_chart")
        split: Dataset split to load (e.g., "train", "test", "validation", "train[:100]")
        cache_dir: Cache directory for dataset files
        num_proc: Number of processes (only used with "default" method)
        method: Loading method - "default" or "builder_load"
        streaming: Whether to use streaming mode

    Returns:
        Dataset object supporting len(), .shard(), .select(), iteration

    Raises:
        ValueError: If method is not "default" or "builder_load"
        FileNotFoundError: If "builder_load" is used but dataset is not prepared
    """
    if method not in ["default", "builder_load"]:
        raise ValueError(f"Invalid method: {method}. Must be 'default' or 'builder_load'")

    if method == "default":
        config_info = f" (config: {config_name})" if config_name else ""
        logger.info(f"Loading dataset using load_dataset(): {dataset_name}{config_info}/{split}")

        if streaming:
            logger.warning(
                "Number of processes is set to 1 as streaming with multiple processes is not implemented in hf"
            )
            num_proc = None

            # Parse split to handle slice notation (not supported in streaming mode)
            base_split, start, end = _parse_split_slice(split)

            if start is not None or end is not None:
                logger.info(
                    f"Detected slice notation in split '{split}'. Loading base split '{base_split}' and applying slice via .skip()/.take()"
                )
                actual_split = base_split
            else:
                actual_split = split
        else:
            actual_split = split
            start, end = None, None

        dataset = load_dataset(
            dataset_name,
            name=config_name,
            split=actual_split,
            cache_dir=cache_dir,
            num_proc=num_proc,
            streaming=streaming,
        )

        # Apply slicing for streaming datasets
        if streaming and (start is not None or end is not None):
            if start is not None and start > 0:
                logger.info(f"Skipping first {start} samples")
                dataset = dataset.skip(start)
            if end is not None:
                count = end - (start or 0)
                logger.info(f"Taking {count} samples")
                dataset = dataset.take(count)
            elif start is not None:
                # start specified but no end - take all remaining
                logger.info(f"Taking all samples after skipping {start}")

        return dataset

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
                "Using 'builder_load' without explicit cache_dir. " "Will use default: ~/.cache/huggingface/datasets"
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
            raise FileNotFoundError(f"Dataset not prepared. Use method='default' or prepare dataset first.") from e

    # Log dataset size (streaming datasets don't have len())
    if streaming:
        logger.info(f"Loaded streaming dataset from {split} split")
    else:
        logger.info(f"Loaded {len(dataset)} samples from {split} split")
    return dataset
