#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import signal
import tarfile
from multiprocessing import Pool, cpu_count, Manager
from pathlib import Path
from PIL import Image
from io import BytesIO
import webdataset as wds
from tqdm import tqdm
import sys
from collections import defaultdict
import threading


def load_coco_annotations(annotations_dir, split='train2017'):
    """Load and parse COCO annotations for a given split."""
    print(f"Loading COCO annotations for {split}")
    
    # Define annotation file paths
    instances_file = os.path.join(annotations_dir, f"instances_{split}.json")
    captions_file = os.path.join(annotations_dir, f"captions_{split}.json")
    keypoints_file = os.path.join(annotations_dir, f"person_keypoints_{split}.json")
    
    # Load instances (object detection and segmentation)
    instances_data = None
    if os.path.exists(instances_file):
        print(f"Loading instances from {instances_file}")
        with open(instances_file, 'r') as f:
            instances_data = json.load(f)
    
    # Load captions
    captions_data = None
    if os.path.exists(captions_file):
        print(f"Loading captions from {captions_file}")
        with open(captions_file, 'r') as f:
            captions_data = json.load(f)
    
    # Load keypoints
    keypoints_data = None
    if os.path.exists(keypoints_file):
        print(f"Loading keypoints from {keypoints_file}")
        with open(keypoints_file, 'r') as f:
            keypoints_data = json.load(f)
    
    # Create a mapping from image_id to image info
    image_id_to_info = {}
    if instances_data:
        for img in instances_data['images']:
            image_id_to_info[img['id']] = img
    
    # Create categories mapping
    categories = {}
    if instances_data and 'categories' in instances_data:
        for cat in instances_data['categories']:
            categories[cat['id']] = cat
    
    # Group annotations by image_id
    image_annotations = defaultdict(lambda: {
        'instances': [],
        'captions': [],
        'keypoints': [],
        'categories': categories
    })
    
    # Process instance annotations
    if instances_data and 'annotations' in instances_data:
        print(f"Processing {len(instances_data['annotations'])} instance annotations")
        for ann in instances_data['annotations']:
            image_id = ann['image_id']
            image_annotations[image_id]['instances'].append(ann)
    
    # Process caption annotations
    if captions_data and 'annotations' in captions_data:
        print(f"Processing {len(captions_data['annotations'])} caption annotations")
        for ann in captions_data['annotations']:
            image_id = ann['image_id']
            image_annotations[image_id]['captions'].append(ann)
    
    # Process keypoint annotations
    if keypoints_data and 'annotations' in keypoints_data:
        print(f"Processing {len(keypoints_data['annotations'])} keypoint annotations")
        for ann in keypoints_data['annotations']:
            image_id = ann['image_id']
            image_annotations[image_id]['keypoints'].append(ann)
    
    # Create final samples list
    samples = []
    for image_id, img_info in image_id_to_info.items():
        sample = {
            'image_id': image_id,
            'file_name': img_info['file_name'],
            'height': img_info['height'],
            'width': img_info['width'],
            'coco_url': img_info.get('coco_url', ''),
            'flickr_url': img_info.get('flickr_url', ''),
            'date_captured': img_info.get('date_captured', ''),
            'license': img_info.get('license', 0),
            'annotations': image_annotations[image_id]
        }
        samples.append(sample)
    
    print(f"Loaded {len(samples)} image samples")
    return samples


def validate_tar_file(tar_path):
    """Validate that a tar file is properly formatted and complete."""
    try:
        with tarfile.open(tar_path, 'r') as tar:
            tar.getmembers()
        return True
    except (tarfile.ReadError, EOFError, OSError):
        return False


class SequentialShardWriter:
    """Custom shard writer that uses sequential numbering across all chunks."""
    def __init__(self, output_dir, split, samples_per_shard, shard_counter, shard_lock):
        self.output_dir = output_dir
        self.split = split
        self.samples_per_shard = samples_per_shard
        self.shard_counter = shard_counter
        self.shard_lock = shard_lock
        self.current_shard_samples = []
        self.current_writer = None
        self.current_shard_path = None
        
    def _get_next_shard_number(self):
        """Get the next shard number in a thread-safe manner."""
        with self.shard_lock:
            shard_num = self.shard_counter.value
            self.shard_counter.value += 1
            return shard_num
    
    def _start_new_shard(self):
        """Start a new shard file."""
        if self.current_writer:
            self._finish_current_shard()
        
        shard_num = self._get_next_shard_number()
        self.current_shard_path = os.path.join(
            self.output_dir, 
            f"temp_coco_{self.split}-{shard_num:06d}.tar"
        )
        self.current_writer = wds.TarWriter(self.current_shard_path)
        self.current_shard_samples = []
    
    def _finish_current_shard(self):
        """Finish the current shard and move to final location."""
        if self.current_writer:
            self.current_writer.close()
            
            # Move temp file to final location
            final_path = self.current_shard_path.replace("temp_", "")
            if validate_tar_file(self.current_shard_path):
                shutil.move(self.current_shard_path, final_path)
                print(f"Successfully created: {final_path} ({len(self.current_shard_samples)} samples)")
            else:
                print(f"Warning: Invalid tar file, removing: {self.current_shard_path}")
                os.remove(self.current_shard_path)
            
            self.current_writer = None
            self.current_shard_samples = []
    
    def write(self, sample):
        """Write a sample to the current shard."""
        if not self.current_writer or len(self.current_shard_samples) >= self.samples_per_shard:
            self._start_new_shard()
        
        self.current_writer.write(sample)
        self.current_shard_samples.append(sample['__key__'])
    
    def close(self):
        """Close any open shard."""
        if self.current_writer:
            self._finish_current_shard()


def process_chunk(args):
    """Process a chunk of COCO samples."""
    chunk_data, chunk_idx, output_dir, image_dir, samples_per_shard, split, shard_counter, shard_lock, jpeg_quality = args
    
    samples_written = 0
    samples_with_errors = 0
    
    def signal_handler(signum, frame):
        print(f"Process {os.getpid()} received signal {signum}, cleaning up...")
        sys.exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Create sequential shard writer
        writer = SequentialShardWriter(output_dir, split, samples_per_shard, shard_counter, shard_lock)
        
        for sample in tqdm(chunk_data, desc=f"Chunk {chunk_idx}", leave=False):
            try:
                # Extract sample information
                image_id = sample['image_id']
                file_name = sample['file_name']
                
                # Construct full image path
                full_image_path = os.path.join(image_dir, file_name)
                
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
                        
                        # Save as JPEG bytes with configurable quality
                        img_buffer = BytesIO()
                        if jpeg_quality == 100:
                            # Use PNG for lossless compression
                            img.save(img_buffer, format='PNG')
                            img_bytes = img_buffer.getvalue()
                            file_ext = 'png'
                        else:
                            # Use JPEG with specified quality
                            img.save(img_buffer, format='JPEG', quality=jpeg_quality)
                            img_bytes = img_buffer.getvalue()
                            file_ext = 'jpg'
                
                except Exception as e:
                    print(f"Error processing image {full_image_path}: {e}")
                    samples_with_errors += 1
                    file_ext = 'jpg'  # Default to jpg if error
                    continue
                
                # Create the sample for webdataset
                sample_data = {
                    "__key__": f"{image_id:012d}",  # Zero-pad to 12 digits
                    file_ext: img_bytes,  # Image data (jpg or png)
                    "json": {
                        "image_id": image_id,
                        "file_name": file_name,
                        "height": sample['height'],
                        "width": sample['width'],
                        "coco_url": sample.get('coco_url', ''),
                        "flickr_url": sample.get('flickr_url', ''),
                        "date_captured": sample.get('date_captured', ''),
                        "license": sample.get('license', 0),
                        "instances": sample['annotations']['instances'],
                        "captions": sample['annotations']['captions'],
                        "keypoints": sample['annotations']['keypoints'],
                        "categories": sample['annotations']['categories']
                    }
                }
                
                # Write sample to shard
                writer.write(sample_data)
                samples_written += 1
                
            except Exception as e:
                print(f"Error processing sample {sample.get('image_id', 'unknown')}: {e}")
                samples_with_errors += 1
                continue
        
        # Close the writer to finish any remaining shard
        writer.close()
    
    except Exception as e:
        print(f"Critical error in chunk {chunk_idx}: {e}")
        samples_written = 0
    
    return chunk_idx, samples_written, samples_with_errors


def main():
    parser = argparse.ArgumentParser(description="Convert COCO dataset to WebDataset format with sequential shard numbering")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/capstor/store/cscs/swissai/infra01/vision-datasets/coco2017",
        help="Root directory of COCO dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train2017",
        choices=["train2017", "val2017"],
        help="Dataset split to convert"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./coco_webdataset",
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
        default=5000,
        help="Number of samples per chunk for parallel processing"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG compression quality (1-100, 100 uses PNG for lossless)"
    )
    
    args = parser.parse_args()
    
    # Construct paths
    annotations_dir = os.path.join(args.data_root, "annotations")
    image_dir = os.path.join(args.data_root, args.split)
    
    # Validate input paths
    if not os.path.exists(annotations_dir):
        print(f"Error: Annotations directory not found: {annotations_dir}")
        sys.exit(1)
    
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found: {image_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load COCO annotations
    samples = load_coco_annotations(annotations_dir, args.split)
    
    # Sort samples by image_id for consistent ordering
    samples.sort(key=lambda x: x['image_id'])
    
    # Limit samples if max_samples is specified (for testing)
    if args.max_samples is not None and args.max_samples < len(samples):
        samples = samples[:args.max_samples]
        print(f"Limiting to first {args.max_samples} samples for testing")
    
    # Create a shared counter for sequential shard numbering
    manager = Manager()
    shard_counter = manager.Value('i', 0)
    shard_lock = manager.Lock()
    
    # Split samples into chunks for parallel processing
    chunk_size = args.chunk_size
    chunks = []
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i + chunk_size]
        chunks.append((chunk, i // chunk_size, args.output_dir, image_dir, 
                      args.samples_per_shard, args.split, shard_counter, shard_lock, args.jpeg_quality))
    
    print(f"\n{'='*60}")
    print(f"Converting COCO {args.split} to WebDataset format")
    print(f"{'='*60}")
    print(f"Data root: {args.data_root}")
    print(f"Image directory: {image_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Total samples: {len(samples)}")
    print(f"Samples per shard: {args.samples_per_shard}")
    print(f"Expected number of shards: ~{len(samples) // args.samples_per_shard + (1 if len(samples) % args.samples_per_shard else 0)}")
    print(f"Processing in {len(chunks)} chunks using {args.num_workers} workers")
    print(f"JPEG quality: {args.jpeg_quality}{'% (lossless PNG)' if args.jpeg_quality == 100 else '%'}")
    print(f"{'='*60}\n")
    
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
    
    print(f"\n{'='*60}")
    print(f"Conversion completed!")
    print(f"{'='*60}")
    print(f"Total samples in dataset: {len(samples)}")
    print(f"Total samples written: {total_samples_written}")
    print(f"Total samples with errors: {total_samples_with_errors}")
    if len(samples) > 0:
        print(f"Success rate: {total_samples_written / len(samples) * 100:.1f}%")
    
    # List output files
    output_files = sorted([f for f in os.listdir(args.output_dir) if f.endswith('.tar') and f.startswith(f'coco_{args.split}')])
    print(f"\nGenerated {len(output_files)} shard files:")
    
    total_size = 0
    for i, filename in enumerate(output_files[:10]):  # Show first 10
        filepath = os.path.join(args.output_dir, filename)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        total_size += os.path.getsize(filepath)
        print(f"  {filename} ({size_mb:.1f} MB)")
    
    if len(output_files) > 10:
        print(f"  ... and {len(output_files) - 10} more files")
        # Calculate total size for all files
        for filename in output_files[10:]:
            filepath = os.path.join(args.output_dir, filename)
            total_size += os.path.getsize(filepath)
    
    print(f"\nTotal size: {total_size / (1024 * 1024 * 1024):.2f} GB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()