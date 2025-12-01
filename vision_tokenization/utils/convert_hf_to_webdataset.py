#!/usr/bin/env python3
"""
Convert HuggingFace FineVision dataset to WebDataset format.
Optimized for large-scale processing with progress tracking.

Example commands:
    # Convert OCR-VQA dataset
    python vision_tokenization/utils/convert_hf_to_webdataset.py \
        --dataset "HuggingFaceM4/FineVision" \
        --config "allava_laion" \
        --output-dir "/capstor/store/cscs/swissai/infra01/vision-datasets/FineVision/wds" \
        --shard-size 10000 \
        --num-workers 128 \
        --verify
"""

import argparse
import io
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import webdataset as wds
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def convert_sample_batch(batch: List[Tuple[int, Dict[str, Any]]]) -> List[Optional[Dict[str, bytes]]]:
    """
    Convert a batch of samples in a single process.

    Args:
        batch: List of (idx, sample) tuples

    Returns:
        List of converted samples or None for failures
    """
    return [convert_sample(sample, idx) for idx, sample in batch]


def convert_sample(sample: Dict[str, Any], idx: int) -> Optional[Dict[str, bytes]]:
    """
    Convert a single HF dataset sample to WebDataset format.

    Args:
        sample: HuggingFace dataset sample with 'images' and 'texts' fields
        idx: Sample index for unique key

    Returns:
        Dict with __key__, png, and json fields for WebDataset
    """
    try:
        # FineVision uses 'images' (plural) field, which is a list
        images = sample.get("images")
        if not images or len(images) != 1:
            # Skip samples with no images or multiple images
            return None

        image = images[0]
        if not isinstance(image, Image.Image):
            return None

        # Save as PNG for lossless compression
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        width, height = image.size

        # Prepare metadata with texts (Q&A pairs) and image info
        metadata = {
            "texts": sample.get("texts", []),  # List of {"user": str, "assistant": str}
            "original_index": idx,  # Store original dataset index in metadata
            "width": width,
            "height": height,
            "mode": image.mode,
        }

        # Add any additional fields (except images)
        for key in sample.keys():
            if key not in ["images", "texts"]:
                # Handle non-serializable types
                try:
                    json.dumps(sample[key])  # Test if serializable
                    metadata[key] = sample[key]
                except (TypeError, ValueError):
                    metadata[key] = str(sample[key])

        # Return WebDataset format
        return {
            "__key__": f"{idx:08d}",
            "png": img_bytes.getvalue(),
            "json": json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
        }

    except Exception as e:
        print(f"Error converting sample {idx}: {e}")
        return None


def convert_dataset(
    dataset_path: str,
    output_dir: str,
    config: Optional[str] = None,
    split: str = "train",
    shard_size: int = 1000,
    num_workers: int = 4,
):
    """
    Convert HuggingFace dataset to WebDataset format.

    Args:
        dataset_path: HF dataset name
        output_dir: Output directory for WebDataset tar files
        config: Dataset configuration (e.g., 'ocrvqa')
        split: Dataset split to convert
        shard_size: Number of samples per tar shard
        num_workers: Number of parallel workers for loading
    """
    start_time = time.time()

    # Create output directory with config subfolder
    base_output_path = Path(output_dir)
    if config:
        output_path = base_output_path / config
    else:
        output_path = base_output_path
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset...")
    dataset = load_dataset(dataset_path, config, split=split, num_proc=16)

    total_samples = len(dataset)
    print(f"Dataset loaded: {total_samples} samples")

    # Check dataset structure
    if total_samples > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        if "texts" in sample and sample["texts"]:
            print(f"First Q&A pair: {sample['texts'][0]}")

    # Prepare output pattern
    output_pattern = str(output_path / f"{config or 'data'}-%06d.tar")

    # Convert to WebDataset with parallel processing
    print(f"\nConverting to WebDataset format...")
    print(f"Output pattern: {output_pattern}")
    print(f"Shard size: {shard_size} samples")
    print(f"Using {num_workers} parallel workers")

    successful = 0
    failed = 0

    # Process in batches for efficiency
    batch_size = min(100, shard_size // 10)  # Process 100 samples per worker batch

    def process_future_results(future, sink, pbar):
        """Helper to process a single future's results."""
        nonlocal successful, failed
        results = future.result()
        for wds_sample in results:
            if wds_sample is not None:
                # Use sequential counter for key to avoid shard fragmentation
                wds_sample["__key__"] = f"{successful:08d}"
                sink.write(wds_sample)
                successful += 1
            else:
                failed += 1
            pbar.update(1)

    with wds.ShardWriter(output_pattern, maxcount=shard_size) as sink:
        with tqdm(total=total_samples, desc="Converting samples") as pbar:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []

                # Submit all work in batches
                for i in range(0, total_samples, batch_size):
                    batch = [(idx, dataset[idx]) for idx in range(i, min(i + batch_size, total_samples))]
                    futures.append(executor.submit(convert_sample_batch, batch))

                    # Process completed futures when queue gets too large
                    if len(futures) >= num_workers * 2:
                        # Process one completed future
                        done_future = next(as_completed(futures[:num_workers]))
                        process_future_results(done_future, sink, pbar)
                        futures.remove(done_future)

                        # Update progress
                        pbar.set_postfix(
                            {
                                "shard": successful // shard_size,
                                "success": successful,
                                "failed": failed,
                                "pending": len(futures),
                            }
                        )

                # Process all remaining futures
                for future in as_completed(futures):
                    process_future_results(future, sink, pbar)

    # Print summary
    elapsed = time.time() - start_time
    num_shards = (successful + shard_size - 1) // shard_size

    print(f"\n{'='*50}")
    print(f"Conversion completed in {elapsed:.1f} seconds")
    print(f"Total samples: {total_samples}")
    print(f"Successfully converted: {successful}")
    print(f"Failed: {failed}")
    print(f"Created {num_shards} shards")
    print(f"Output directory: {output_path}")

    # List created files
    import glob

    tar_files = sorted(glob.glob(str(output_path / "*.tar")))
    if tar_files:
        print(f"\nCreated files:")
        for tar_file in tar_files[:5]:  # Show first 5
            size_mb = os.path.getsize(tar_file) / (1024 * 1024)
            print(f"  {Path(tar_file).name}: {size_mb:.1f} MB")
        if len(tar_files) > 5:
            print(f"  ... and {len(tar_files) - 5} more files")

    # Create info file
    info_file = output_path / "dataset_info.json"
    info = {
        "source_dataset": dataset_path,
        "config": config,
        "split": split,
        "total_samples": total_samples,
        "successful_samples": successful,
        "failed_samples": failed,
        "num_shards": num_shards,
        "shard_size": shard_size,
        "conversion_time": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(info_file, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nDataset info saved to: {info_file}")


def verify_webdataset(tar_path: str, num_samples: int = 5):
    """
    Verify the created WebDataset by reading a few samples.

    Args:
        tar_path: Path to a tar file or pattern
        num_samples: Number of samples to check
    """
    print(f"\nVerifying WebDataset: {tar_path}")

    dataset = wds.WebDataset(tar_path)

    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        print(f"\nSample {i}:")
        print(f"  Keys: {sample.keys()}")

        # Check PNG image
        if "png" in sample:
            img = Image.open(io.BytesIO(sample["png"]))
            print(f"  Image: {img.size} {img.mode}")

        # Check metadata
        if "json" in sample:
            metadata = json.loads(sample["json"])
            print(f"  Metadata keys: {metadata.keys()}")
            if "texts" in metadata and metadata["texts"]:
                print(f"  Q&A pairs: {len(metadata['texts'])}")
                print(f"  First Q: {metadata['texts'][0].get('user', '')[:100]}...")


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace dataset to WebDataset format")

    # Input options
    parser.add_argument("--dataset", type=str, default="HuggingFaceM4/FineVision", help="HuggingFace dataset name")
    parser.add_argument(
        "--config", type=str, default="ocrvqa", help="Dataset configuration (e.g., ocrvqa, textcaps, vqav2)"
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to convert")

    # Output options
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for WebDataset files")
    parser.add_argument("--shard-size", type=int, default=1000, help="Number of samples per shard")

    # Processing options
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(64, mp.cpu_count()),
        help="Number of parallel workers (default: min(64, cpu_count))",
    )
    parser.add_argument("--verify", action="store_true", help="Verify the created WebDataset")

    args = parser.parse_args()

    # Convert dataset
    convert_dataset(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        config=args.config,
        split=args.split,
        shard_size=args.shard_size,
        num_workers=args.num_workers,
    )

    # Verify if requested
    if args.verify:
        import glob

        tar_files = glob.glob(os.path.join(args.output_dir, "*.tar"))
        if tar_files:
            verify_webdataset(tar_files[0])


if __name__ == "__main__":
    main()
