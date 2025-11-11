#!/usr/bin/env python3
"""
Check if a Hugging Face dataset is properly cached and report its size.
"""

import os
import sys
import argparse
from pathlib import Path
from datasets import load_dataset


def get_directory_size(directory):
    """
    Returns:
        Total size in bytes
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        print(f"Warning: Error calculating size: {e}")
    return total_size


def format_size(size_bytes):
    """
    Format bytes into human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.23 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def check_dataset_cache(dataset_name, cache_dir, config_name=None, split=None):
    """
    Check if a Hugging Face dataset is cached and report its size.

    Args:
        dataset_name: Name of the dataset (e.g., "squad", "imdb")
        cache_dir: Path to the cache directory
        config_name: Optional dataset configuration name
        split: Optional specific split to check

    Returns:
        Dictionary with cache status and size information
    """
    result = {
        "dataset_name": dataset_name,
        "cache_dir": cache_dir,
        "is_cached": False,
        "cache_exists": False,
        "cache_size_bytes": 0,
        "cache_size_formatted": "0 B",
        "error": None
    }

    # Check if cache directory exists
    if not os.path.exists(cache_dir):
        result["error"] = f"Cache directory does not exist: {cache_dir}"
        return result

    result["cache_exists"] = True

    # Try to load the dataset without downloading
    try:
        print(f"Checking if '{dataset_name}' is cached...")

        load_params = {
            "path": dataset_name,
            "cache_dir": cache_dir,
            "download_mode": "reuse_cache_if_exists"
        }

        if config_name:
            load_params["name"] = config_name
        if split:
            load_params["split"] = split

        dataset = load_dataset(**load_params)

        result["is_cached"] = True
        print(f"✓ Dataset '{dataset_name}' is properly cached!")

        # Print dataset info
        if isinstance(dataset, dict):
            print(f"\nDataset splits: {list(dataset.keys())}")
            for split_name, split_data in dataset.items():
                print(f"  - {split_name}: {len(split_data)} examples")
        else:
            print(f"\nDataset examples: {len(dataset)}")

    except Exception as e:
        result["error"] = str(e)
        print(f"✗ Dataset '{dataset_name}' is NOT properly cached or error occurred")
        print(f"Error: {e}")

    # Calculate cache size
    try:
        print(f"\nCalculating cache size for: {cache_dir}")
        size_bytes = get_directory_size(cache_dir)
        result["cache_size_bytes"] = size_bytes
        result["cache_size_formatted"] = format_size(size_bytes)
        print(f"Total cache size: {result['cache_size_formatted']}")
    except Exception as e:
        result["error"] = f"Error calculating cache size: {e}"
        print(f"Warning: {result['error']}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Check if a Hugging Face dataset is properly cached and report its size",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_dataset_cache.py squad /path/to/cache
  python check_dataset_cache.py imdb /path/to/cache
  python check_dataset_cache.py glue /path/to/cache --config mrpc
  python check_dataset_cache.py squad /path/to/cache --split train
        """
    )

    parser.add_argument(
        "dataset_name",
        help="Name of the Hugging Face dataset (e.g., 'squad', 'imdb')"
    )

    parser.add_argument(
        "cache_dir",
        help="Path to the cache directory", default="/capstor/store/cscs/swissai/infra01/vision-datasets/hf_cache"
    )

    parser.add_argument(
        "--config",
        help="Dataset configuration name (if applicable)",
        default=None
    )

    parser.add_argument(
        "--split",
        help="Specific split to check (e.g., 'train', 'test')",
        default=None
    )

    args = parser.parse_args()

    # Run the check
    print("=" * 60)
    print("Hugging Face Dataset Cache Checker")
    print("=" * 60)

    result = check_dataset_cache(
        args.dataset_name,
        args.cache_dir,
        args.config,
        args.split
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Dataset: {result['dataset_name']}")
    print(f"Cache Directory: {result['cache_dir']}")
    print(f"Cache Directory Exists: {result['cache_exists']}")
    print(f"Dataset Cached: {result['is_cached']}")
    print(f"Cache Size: {result['cache_size_formatted']}")

    if result['error']:
        print(f"Error: {result['error']}")
        sys.exit(1)
    elif result['is_cached']:
        print("\n✓ Dataset is properly cached and ready to use!")
        sys.exit(0)
    else:
        print("\n✗ Dataset is not cached. You may need to download it.")
        sys.exit(1)


if __name__ == "__main__":
    main()