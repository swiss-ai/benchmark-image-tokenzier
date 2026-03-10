#!/usr/bin/env python3
"""
Utility functions for loading HuggingFace datasets.
Supports both standard load_dataset() and dataset builder approaches.

Also contains methods to lookup num dataset samples based on the load_dataset_builder approach which does not map files to
virtual memory -> Works even if dataset exceeds limits of memory mapping.
"""

import logging
import os
import re
from typing import Dict, Optional, Tuple

from datasets import Dataset, load_dataset, load_dataset_builder, load_from_disk

logger = logging.getLogger(__name__)


def log_hf_environment_info(logger_instance, worker_id: Optional[int] = None):
    """
    Log HuggingFace environment configuration at WARNING level.

    Logs:
    - HF_HUB_CACHE location (or HF_HOME fallback)
    - HF_DATASETS_OFFLINE status

    Args:
        logger_instance: Logger instance to use
        worker_id: Optional worker ID for worker context logging
    """
    # Determine prefix for worker context
    prefix = f"[Worker {worker_id}] " if worker_id is not None else ""

    # Check HF_HUB_CACHE (primary) or HF_HOME (fallback)
    hf_cache = os.environ.get("HF_HUB_CACHE")
    if hf_cache:
        logger_instance.warning(f"{prefix}HF Hub Cache: {hf_cache}")
    else:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            logger_instance.warning(f"{prefix}HF Hub Cache: {hf_home} (via HF_HOME)")
        else:
            logger_instance.warning(f"{prefix}HF Hub Cache: Using HuggingFace default cache location")

    # Check HF_DATASETS_OFFLINE
    offline_mode = os.environ.get("HF_DATASETS_OFFLINE", "0")
    if offline_mode == "1":
        logger_instance.warning(f"{prefix}HF Datasets Offline Mode: ENABLED (will only use cached datasets)")
    else:
        logger_instance.warning(f"{prefix}HF Datasets Offline Mode: DISABLED (will download from Hub if needed)")


def get_system_max_map_count() -> Optional[int]:
    """
    Get the system's maximum number of memory map areas (vm.max_map_count).
    return None if not found!
    """
    try:
        with open("/proc/sys/vm/max_map_count", "r") as f:
            max_map_count = int(f.read().strip())
            return max_map_count
    except (FileNotFoundError, PermissionError, ValueError) as e:
        logger.debug(f"Could not read max_map_count: {e}")
        return None


def get_builder_split_info(
    dataset_name: str,
    config_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    dataset_load_method: str = "default",
    data_files: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Optional[Dict[str, Dict[str, int]]]:
    """
    Load dataset builder and extract split information without loading the dataset.

    Args:
        dataset_name: HF dataset name (e.g., "HuggingFaceM4/FineVision")
        config_name: Dataset configuration/subset name (e.g., "lrv_chart")
        cache_dir: Cache directory for dataset files
        dataset_load_method: How to load the dataset ("default" or "disk_load")
        data_files: Explicit data file path(s) or pattern(s) as a comma-separated string.
        data_dir: Local directory containing the dataset files (passed to builder as data_dir).

    Returns:
        Dictionary mapping split names to their info, or None if split info
        is not available (e.g. for webdataset from local tar files where the
        builder cannot determine splits without preparing the data first).
    """
    config_info = f" (config: {config_name})" if config_name else ""
    logger.info(f"Loading builder for: {dataset_name}{config_info}")

    if dataset_load_method == "disk_load":
        # For datasets saved with save_to_disk, load_dataset_builder doesn't work.
        # Use load_from_disk to get the dataset info directly.
        logger.info(f"Using disk_load method to get split info from: {dataset_name}")
        ds = load_from_disk(dataset_name)
        info = ds.info
    else:
        if cache_dir is None:
            logger.warning("No explicit cache_dir provided. " "Will use default: ~/.cache/huggingface/datasets")

        # Parse data_files string into the format load_dataset_builder expects
        parsed_data_files = None
        if data_files:
            parts = [p.strip() for p in data_files.split(",")]
            parsed_data_files = parts if len(parts) > 1 else parts[0]

        builder = load_dataset_builder(
            dataset_name, name=config_name, cache_dir=cache_dir, data_files=parsed_data_files, data_dir=data_dir
        )
        info = builder.info

    logger.info(f"=== Builder Split Info ===")

    if info.splits is None:
        logger.warning(
            "Builder has no split metadata available. This is expected for datasets "
            "loaded from local files (e.g. webdataset tar files) that have not been "
            "prepared yet. Dataset size will be determined after loading."
        )
        return None

    split_info = {}
    for split_name, split_data in info.splits.items():
        # Extract num_examples, num_shards, and shard_lengths
        num_examples = split_data.num_examples
        # Use `or []` to handle both missing attribute AND None value
        shard_lengths = getattr(split_data, "shard_lengths", None) or []
        num_shards = len(shard_lengths) if shard_lengths else None

        split_info[split_name] = {
            "num_examples": num_examples,
            "num_shards": num_shards,
            "shard_lengths": list(shard_lengths),  # Include full list for shard overlap calculation
        }

        logger.info(
            f"  Split '{split_name}': {num_examples:,} examples"
            + (f", {num_shards} shard(s)" if num_shards is not None else "")
        )

    return split_info


def check_memory_mapping_limits(
    split_info: Dict[str, Dict[str, int]], split: str, dataset_streamed: bool = False
) -> None:
    """
    Check if system memory mapping limits are sufficient for the dataset.
    Accounts for split slicing to accurately estimate required memory maps.
    If limits are insufficient and streaming is not enabled, abort tokenization.

    Args:
        split_info: Dictionary with split information from get_builder_split_info()
        split: The split being loaded (may include slice notation like "train[:100]" or "train[:50%]")
        dataset_streamed: Whether dataset streaming is enabled

    Raises:
        RuntimeError: If memory mapping limits are insufficient and streaming is not enabled
    """
    # Parse split with percentage support
    base_split, start, end, start_is_pct, end_is_pct = _parse_split_slice(split)

    # Get split metadata
    if base_split not in split_info:
        logger.warning(f"Split '{base_split}' not found in dataset info. Skipping memory mapping check.")
        return

    split_metadata = split_info[base_split]
    num_shards = split_metadata.get("num_shards")

    if num_shards is None:
        logger.debug(f"Number of shards not available for split '{base_split}'. Skipping memory mapping check.")
        return

    # Calculate actual shards needed if slicing is used
    shards_needed = num_shards  # Default: all shards

    if start is not None or end is not None:
        shard_lengths = split_metadata.get("shard_lengths")
        num_examples = split_metadata.get("num_examples")

        if shard_lengths and num_examples:
            # Convert percentages to absolute indices
            abs_start = _convert_percentage_to_absolute(start, start_is_pct, num_examples)
            abs_end = _convert_percentage_to_absolute(end, end_is_pct, num_examples)

            abs_start = abs_start if abs_start is not None else 0
            abs_end = abs_end if abs_end is not None else num_examples

            # Calculate which shards overlap with slice range
            first_shard, last_shard, shards_needed = _calculate_shard_overlap(abs_start, abs_end, shard_lengths)

            logger.info(
                f"Split slice '{split}' requires shards [{first_shard}, {last_shard}] "
                f"({shards_needed} out of {num_shards} total shards)"
            )
        else:
            logger.warning(
                f"Cannot calculate exact shard count for slice '{split}'. "
                f"Using total shard count ({num_shards}) for memory check."
            )

    # Get system max_map_count
    max_map_count = get_system_max_map_count()

    if max_map_count is None:
        logger.warning(
            f"⚠️  WARNING: Could not determine system's max memory mappings (vm.max_map_count). "
            f"Split '{base_split}' requires {shards_needed} memory map(s)."
        )
        return

    # Validate limits
    if max_map_count < shards_needed:
        if dataset_streamed:
            # Streaming mode is enabled - memory mapping not used, so this is OK
            logger.info(
                f"⚠️  System max memory mappings ({max_map_count:,}) < required shards ({shards_needed}).\n"
                f"✓ Streaming ENABLED - will work correctly without memory mapping."
            )
        else:
            # Streaming mode is NOT enabled - this will cause problems
            error_msg = (
                f"❌ ERROR: System max memory mappings ({max_map_count:,}) < required shards ({shards_needed}).\n"
                f"This will cause memory mapping errors during dataset loading.\n\n"
                f"SOLUTION: Enable streaming (--dataset-streamed) or increase limit:\n"
                f"  sudo sysctl -w vm.max_map_count={shards_needed * 2}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    else:
        logger.info(f"✓ System max memory mappings ({max_map_count:,}) sufficient " f"for {shards_needed} shard(s)")


def _parse_split_slice(split: str) -> Tuple[str, Optional[int], Optional[int], bool, bool]:
    """
    Parse split string with optional slice notation including percentages.

    Args:
        split: Split string (e.g., "train", "train[:100]", "train[:50%]", "train[25%:75%]")

    Returns:
        Tuple of (base_split, start, end, start_is_pct, end_is_pct)

    Examples:
        "train" -> ("train", None, None, False, False)
        "train[:100]" -> ("train", None, 100, False, False)
        "train[:50%]" -> ("train", None, 50, False, True)
        "train[25%:75%]" -> ("train", 25, 75, True, True)
        "train[100:50%]" -> ("train", 100, 50, False, True)
    """
    # Updated regex to capture percentage symbols
    match = re.match(r"^(\w+)(?:\[(\d*)(%)?:(\d*)(%)?])?$", split)
    if not match:
        # No slice notation or invalid format
        return split, None, None, False, False

    base_split = match.group(1)
    start_str = match.group(2)
    start_pct = match.group(3) == "%"
    end_str = match.group(4)
    end_pct = match.group(5) == "%"

    # Convert to int if present, otherwise None
    start = int(start_str) if start_str else None
    end = int(end_str) if end_str else None

    return base_split, start, end, start_pct, end_pct


def _convert_percentage_to_absolute(
    value: Optional[int],
    is_percentage: bool,
    total_size: int,
) -> Optional[int]:
    """
    Convert slice boundary from percentage to absolute index.
    Uses HuggingFace's "closest" rounding (round to nearest integer).

    Args:
        value: The slice boundary value
        is_percentage: Whether value is a percentage
        total_size: Total number of samples in split

    Returns:
        Absolute index, or None if value is None

    Examples:
        >>> _convert_percentage_to_absolute(50, True, 1000)
        500
        >>> _convert_percentage_to_absolute(100, False, 1000)
        100
        >>> _convert_percentage_to_absolute(None, False, 1000)
        None
    """
    if value is None:
        return None

    if not is_percentage:
        return value

    # Convert percentage to absolute with rounding
    return round(value * total_size / 100.0)


def _calculate_shard_overlap(
    slice_start: int,
    slice_end: int,
    shard_lengths: list,
) -> Tuple[int, int, int]:
    """
    Calculate which shards overlap with a sample index range.

    Args:
        slice_start: Starting sample index (inclusive)
        slice_end: Ending sample index (exclusive)
        shard_lengths: List of per-shard sample counts

    Returns:
        (first_shard_idx, last_shard_idx, num_shards_needed)

    Example:
        >>> _calculate_shard_overlap(500, 1500, [1000, 1000, 1000])
        (0, 1, 2)  # needs shards 0 and 1
    """
    if not shard_lengths:
        return 0, 0, 0

    total_samples = sum(shard_lengths)

    # Clamp slice to valid range
    slice_start = max(0, min(slice_start, total_samples))
    slice_end = max(slice_start, min(slice_end, total_samples))

    if slice_start >= slice_end:
        return 0, 0, 0

    # Build cumulative offsets: [0, len[0], len[0]+len[1], ...]
    cumulative = [0]
    for length in shard_lengths:
        cumulative.append(cumulative[-1] + length)

    # Find first shard containing slice_start
    first_shard = 0
    for i in range(len(shard_lengths)):
        if cumulative[i + 1] > slice_start:
            first_shard = i
            break

    # Find last shard containing data before slice_end
    last_shard = len(shard_lengths) - 1
    for i in range(len(shard_lengths)):
        if cumulative[i + 1] >= slice_end:
            last_shard = i
            break

    num_shards = last_shard - first_shard + 1
    return first_shard, last_shard, num_shards


def _load_with_default_method(
    dataset_name: str,
    config_name: Optional[str],
    actual_split: str,
    cache_dir: Optional[str],
    num_proc: Optional[int],
    streaming: bool,
    data_files: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Dataset:
    """Load dataset using load_dataset()."""
    if streaming and num_proc is not None:
        logger.warning(
            "Number of processes is set to None as streaming with multiple processes "
            "is not implemented in HuggingFace datasets"
        )
        num_proc = None

    # Parse data_files: comma-separated string -> list (or single string)
    parsed_data_files = None
    if data_files:
        parts = [p.strip() for p in data_files.split(",")]
        parsed_data_files = parts if len(parts) > 1 else parts[0]

    return load_dataset(
        dataset_name,
        name=config_name,
        split=actual_split,
        cache_dir=cache_dir,
        num_proc=num_proc,
        streaming=streaming,
        data_files=parsed_data_files,
        data_dir=data_dir,
    )


def _load_with_builder_method(
    dataset_name: str,
    config_name: Optional[str],
    actual_split: str,
    cache_dir: Optional[str],
    num_proc: Optional[int],
    streaming: bool,
    data_files: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Dataset:
    """Load dataset using builder.as_dataset() or builder.as_streaming_dataset()."""
    # Warn about ignored parameters
    if data_files is not None:
        logger.warning(
            f"data_files parameter ('{data_files}') is ignored when using 'builder_load' method. "
            f"load_dataset_builder does not support data_files."
        )
    if num_proc is not None:
        logger.warning(
            f"num_proc parameter ({num_proc}) is ignored when using 'builder_load' method. "
            f"Dataset is already prepared."
        )

    if cache_dir is None:
        logger.warning(
            "Using 'builder_load' without explicit cache_dir. " "Will use default: ~/.cache/huggingface/datasets"
        )

    builder = load_dataset_builder(dataset_name, name=config_name, cache_dir=cache_dir, data_dir=data_dir)

    try:
        # Use appropriate method based on streaming mode
        if streaming:
            dataset = builder.as_streaming_dataset(split=actual_split)
        else:
            dataset = builder.as_dataset(split=actual_split)

        return dataset

    except FileNotFoundError as e:
        error_msg = (
            f"Dataset '{dataset_name}' is not prepared. "
            f"When using 'builder_load', run download_and_prepare() first."
            f"Dataset not prepared. Use method='default' or prepare dataset first."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg) from e


def _load_with_disk_method(
    dataset_name: str,  # This is actually a path for disk_load
    actual_split: str,
    streaming: bool,
) -> Dataset:
    """Load dataset using load_from_disk().

    Used for datasets saved with dataset.save_to_disk() (Arrow format).
    The dataset_name parameter is treated as a local path.

    Args:
        dataset_name: Local path to the saved dataset directory
        actual_split: Split name (used only if load_from_disk returns a DatasetDict)
        streaming: If True, a warning is logged (not supported with load_from_disk)
    """
    from datasets import DatasetDict, load_from_disk

    if streaming:
        logger.warning("Streaming mode is not supported with 'disk_load' method. Ignoring streaming=True.")

    dataset = load_from_disk(dataset_name)

    # load_from_disk may return a DatasetDict — extract the requested split
    if isinstance(dataset, DatasetDict):
        # Strip slice notation from split to get base name
        base_split = actual_split.split("[")[0] if "[" in actual_split else actual_split
        if base_split not in dataset:
            available = list(dataset.keys())
            raise ValueError(
                f"Split '{base_split}' not found in dataset at '{dataset_name}'. " f"Available splits: {available}"
            )
        dataset = dataset[base_split]
        logger.info(f"Extracted split '{base_split}' from DatasetDict")

    return dataset


def _apply_streaming_slice(
    dataset,
    start: Optional[int],
    end: Optional[int],
) -> Dataset:
    """Apply slice to streaming dataset using skip/take."""
    if start is not None and start > 0:
        logger.info(f"[STREAMING] Skipping first {start} samples")
        dataset = dataset.skip(start)

    if end is not None:
        count = end - (start or 0)
        logger.info(f"[STREAMING] Taking {count} samples")
        dataset = dataset.take(count)
    elif start is not None:
        logger.info(f"[STREAMING] Taking all samples after skipping {start}")

    return dataset


def load_hf_dataset(
    dataset_name: str,
    config_name: Optional[str] = None,
    split: str = "train",
    cache_dir: Optional[str] = None,
    num_proc: Optional[int] = None,
    method: str = "default",
    streaming: bool = False,
    data_files: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Dataset:
    """
    Load a HuggingFace dataset using specified method ("default", "builder_load", or "disk_load")

    Both "default" and "builder_load" methods support streaming mode.

    Args:
        dataset_name: HF dataset name (e.g., "HuggingFaceM4/FineVision"), or a local path when using "disk_load"
        config_name: Dataset configuration/subset name (e.g., "lrv_chart"). Ignored for "disk_load".
        split: Dataset split to load (e.g., "train", "test", "validation", "train[:100]")
        cache_dir: Cache directory for dataset files. Ignored for "disk_load".
        num_proc: Number of processes (only used with "default" method)
        method: Loading method - "default", "builder_load", or "disk_load"
        streaming: Whether to use streaming mode (not supported with "disk_load")
        data_files: Explicit data file path(s) or pattern(s) as a comma-separated string.
            Only used with "default" and "builder_load" methods (warning logged for builder_load).
        data_dir: Local directory containing the dataset files (passed to builder as data_dir).

    Returns:
        Dataset object

    Raises:
        ValueError: If method is not "default", "builder_load", or "disk_load"
        FileNotFoundError: If "builder_load" is used but dataset is not prepared
    """
    # Validate method parameter
    if method not in ["default", "builder_load", "disk_load"]:
        raise ValueError(f"Invalid method: {method}. Must be 'default', 'builder_load', or 'disk_load'")

    # Parse split once for all cases (with percentage support)
    base_split, start, end, start_is_pct, end_is_pct = _parse_split_slice(split)
    has_slice = start is not None or end is not None

    # Determine actual split to load
    if streaming and has_slice:
        actual_split = base_split  # Load base, apply slice later
    else:
        actual_split = split  # Load with slice notation

    config_info = f" (config: {config_name})" if config_name else ""

    # Load dataset using appropriate method
    if method == "default":
        logger.info(f"[MODE: default] Loading dataset using load_dataset(): {dataset_name}{config_info}/{actual_split}")
        dataset = _load_with_default_method(
            dataset_name=dataset_name,
            config_name=config_name,
            actual_split=actual_split,
            cache_dir=cache_dir,
            num_proc=num_proc,
            streaming=streaming,
            data_files=data_files,
            data_dir=data_dir,
        )
    elif method == "builder_load":
        logger.info(
            f"[MODE: builder_load] Loading dataset using builder method: {dataset_name}{config_info}/{actual_split}"
        )
        dataset = _load_with_builder_method(
            dataset_name=dataset_name,
            config_name=config_name,
            actual_split=actual_split,
            cache_dir=cache_dir,
            num_proc=num_proc,
            streaming=streaming,
            data_files=data_files,
            data_dir=data_dir,
        )
    else:  # disk_load
        logger.info(f"[MODE: disk_load] Loading dataset using load_from_disk(): {dataset_name}")
        dataset = _load_with_disk_method(
            dataset_name=dataset_name,
            actual_split=actual_split,
            streaming=streaming,
        )
        # Apply slice manually since load_from_disk doesn't support slice notation
        if has_slice:
            num_examples = len(dataset)
            abs_start = _convert_percentage_to_absolute(start, start_is_pct, num_examples)
            abs_end = _convert_percentage_to_absolute(end, end_is_pct, num_examples)
            abs_start = abs_start if abs_start is not None else 0
            abs_end = abs_end if abs_end is not None else num_examples
            logger.info(f"Applying slice [{abs_start}:{abs_end}] to disk-loaded dataset ({num_examples} total)")
            dataset = dataset.select(range(abs_start, abs_end))

    # Apply slicing for streaming datasets (unified for both methods)
    if streaming and has_slice:
        # Convert percentages to absolute indices for streaming
        if start_is_pct or end_is_pct:
            # Need to load builder info to get total size
            split_info = get_builder_split_info(
                dataset_name=dataset_name,
                config_name=config_name,
                cache_dir=cache_dir,
                data_files=data_files,
                data_dir=data_dir,
            )

            if split_info is None or base_split not in split_info:
                raise ValueError(
                    f"Cannot convert percentage slice for streaming: "
                    f"split info not available for '{base_split}'. "
                    f"Use absolute indices instead of percentages."
                )

            num_examples = split_info[base_split]["num_examples"]
            abs_start = _convert_percentage_to_absolute(start, start_is_pct, num_examples)
            abs_end = _convert_percentage_to_absolute(end, end_is_pct, num_examples)
        else:
            abs_start = start
            abs_end = end

        dataset = _apply_streaming_slice(dataset, abs_start, abs_end)

    return dataset
