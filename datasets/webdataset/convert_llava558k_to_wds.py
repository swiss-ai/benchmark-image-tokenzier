#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import signal
import tempfile
import tarfile
from multiprocessing import Pool, cpu_count
from pathlib import Path
from PIL import Image
from io import BytesIO
import webdataset as wds
from tqdm import tqdm
import sys


def load_annotations(json_path):
    """Load and parse the LLaVA 558k annotations."""
    print(f"Loading annotations from {json_path}")
    annotations = []
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    print(f"Loaded {len(annotations)} annotations")
    return annotations


def validate_tar_file(tar_path):
    """Validate that a tar file is properly formatted and complete."""
    try:
        with tarfile.open(tar_path, 'r') as tar:
            tar.getmembers()
        return True
    except (tarfile.ReadError, EOFError, OSError):
        return False


def process_chunk(args):
    """Process a chunk of annotations in parallel."""
    chunk_data, chunk_idx, output_dir, image_base_dir, samples_per_shard = args
    
    output_pattern = os.path.join(output_dir, f"llava558k-{chunk_idx:06d}-%09d.tar")
    temp_output_pattern = os.path.join(output_dir, f"temp_llava558k-{chunk_idx:06d}-%09d.tar")
    
    samples_written = 0
    samples_with_errors = 0
    
    def signal_handler(signum, frame):
        print(f"Process {os.getpid()} received signal {signum}, cleaning up...")
        sys.exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        with wds.ShardWriter(temp_output_pattern, maxcount=samples_per_shard) as sink:
            for sample in tqdm(chunk_data, desc=f"Chunk {chunk_idx}", leave=False):
                try:
                    # Extract sample information
                    sample_id = sample["id"]
                    image_path = sample["image"]  # e.g., "00453/004539375.jpg"
                    conversations = sample["conversations"]
                    
                    # Construct full image path
                    full_image_path = os.path.join(image_base_dir, image_path)
                    
                    # Check if image exists
                    if not os.path.exists(full_image_path):
                        print(f"Warning: Image not found: {full_image_path}")
                        samples_with_errors += 1
                        continue
                    
                    # Load and process image
                    try:
                        with Image.open(full_image_path) as img:
                            # Convert to RGB if needed
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # Save as JPEG bytes
                            img_buffer = BytesIO()
                            img.save(img_buffer, format='JPEG', quality=95)
                            img_bytes = img_buffer.getvalue()
                    
                    except Exception as e:
                        print(f"Error processing image {full_image_path}: {e}")
                        samples_with_errors += 1
                        continue
                    
                    # Create the sample for webdataset
                    sample_data = {
                        "__key__": sample_id,
                        "jpg": img_bytes,  # Image data
                        "json": {
                            "id": sample_id,
                            "image": image_path,
                            "conversations": conversations
                        }
                    }
                    
                    # Write sample to shard
                    sink.write(sample_data)
                    samples_written += 1
                    
                except Exception as e:
                    print(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
                    samples_with_errors += 1
                    continue
        
        # After successful writing, atomically move temp files to final location
        temp_files = [f for f in os.listdir(output_dir) if f.startswith(f"temp_llava558k-{chunk_idx:06d}")]
        for temp_file in temp_files:
            temp_path = os.path.join(output_dir, temp_file)
            final_path = os.path.join(output_dir, temp_file.replace("temp_", ""))
            
            # Validate the tar file before moving
            if validate_tar_file(temp_path):
                shutil.move(temp_path, final_path)
                print(f"Successfully created and validated: {final_path}")
            else:
                print(f"Warning: Invalid tar file detected, removing: {temp_path}")
                os.remove(temp_path)
                samples_written = 0  # Mark as failed
    
    except Exception as e:
        print(f"Critical error in chunk {chunk_idx}: {e}")
        # Clean up any temp files on error
        temp_files = [f for f in os.listdir(output_dir) if f.startswith(f"temp_llava558k-{chunk_idx:06d}")]
        for temp_file in temp_files:
            temp_path = os.path.join(output_dir, temp_file)
            if os.path.exists(temp_path):
                os.remove(temp_path)
        samples_written = 0
    
    return chunk_idx, samples_written, samples_with_errors


def main():
    parser = argparse.ArgumentParser(description="Convert LLaVA 558k dataset to WebDataset format")
    parser.add_argument(
        "--input_json",
        type=str,
        default="/capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json",
        help="Path to the LLaVA 558k annotations JSON file"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain",
        help="Path to the directory containing image folders (00000, 00001, etc.)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain/webdataset_output",
        help="Output directory for WebDataset tar files"
    )
    parser.add_argument(
        "--samples_per_shard",
        type=int,
        default=1000,
        help="Number of samples per shard"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(32, cpu_count()),
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10000,
        help="Number of samples per chunk for parallel processing"
    )
    
    args = parser.parse_args()
    
    # Validate input paths
    if not os.path.exists(args.input_json):
        print(f"Error: Input JSON file not found: {args.input_json}")
        sys.exit(1)
    
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load annotations
    annotations = load_annotations(args.input_json)
    
    # Split annotations into chunks for parallel processing
    chunk_size = args.chunk_size
    chunks = []
    for i in range(0, len(annotations), chunk_size):
        chunk = annotations[i:i + chunk_size]
        chunks.append((chunk, i // chunk_size, args.output_dir, args.image_dir, args.samples_per_shard))
    
    print(f"Processing {len(annotations)} samples in {len(chunks)} chunks using {args.num_workers} workers")
    print(f"Samples per shard: {args.samples_per_shard}")
    print(f"Output directory: {args.output_dir}")
    
    # Process chunks in parallel
    total_samples_written = 0
    total_samples_with_errors = 0
    
    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_chunk, chunks),
            total=len(chunks),
            desc="Processing chunks"
        ))
    
    # Collect results
    for chunk_idx, samples_written, samples_with_errors in results:
        total_samples_written += samples_written
        total_samples_with_errors += samples_with_errors
        print(f"Chunk {chunk_idx}: {samples_written} samples written, {samples_with_errors} errors")
    
    print(f"\nConversion completed!")
    print(f"Total samples processed: {len(annotations)}")
    print(f"Total samples written: {total_samples_written}")
    print(f"Total samples with errors: {total_samples_with_errors}")
    print(f"Success rate: {total_samples_written / len(annotations) * 100:.1f}%")
    
    # List output files
    output_files = sorted([f for f in os.listdir(args.output_dir) if f.endswith('.tar')])
    print(f"Generated {len(output_files)} shard files:")
    for i, filename in enumerate(output_files[:10]):  # Show first 10
        filepath = os.path.join(args.output_dir, filename)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  {filename} ({size_mb:.1f} MB)")
    
    if len(output_files) > 10:
        print(f"  ... and {len(output_files) - 10} more files")


if __name__ == "__main__":
    main()