#!/usr/bin/env python3

import argparse
import json
import os
import webdataset as wds
from tqdm import tqdm
from pathlib import Path
import glob
import io
from collections import defaultdict


def extract_permissive_licensed_data(input_dir, output_dir, samples_per_shard=1000):
    """Extract Attribution and Attribution-ShareAlike licensed images from COCO WebDataset."""
    
    # Permissive license IDs (from COCO documentation)
    # 4: Attribution
    # 5: Attribution-ShareAlike
    PERMISSIVE_LICENSES = {4, 5}
    
    # License names for logging
    license_names = {
        0: 'Unknown',
        1: 'Attribution-NonCommercial-ShareAlike',
        2: 'Attribution-NonCommercial',
        3: 'Attribution-NonCommercial-NoDerivs',
        4: 'Attribution',
        5: 'Attribution-ShareAlike',
        6: 'Attribution-NoDerivs',
        7: 'No known copyright restrictions',
        8: 'United States Government Work'
    }
    
    # Get all input tar files
    if Path(input_dir).is_dir():
        tar_files = sorted(glob.glob(f"{input_dir}/coco_*.tar"))
        if not tar_files:
            print(f"No COCO WebDataset tar files found in {input_dir}")
            return
        print(f"Found {len(tar_files)} input shard files")
    else:
        tar_files = [input_dir]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset reader (no decoding needed)
    dataset = wds.WebDataset(tar_files, shardshuffle=False)
    
    # Statistics
    stats = {
        'total_processed': 0,
        'total_extracted': 0,
        'license_counts': defaultdict(int),
        'skipped_licenses': defaultdict(int)
    }
    
    # Create shard writer for permissive licensed data
    output_pattern = os.path.join(output_dir, "coco_train2017_permissive-%06d.tar")
    
    current_shard_num = 0
    current_shard_samples = []
    current_writer = None
    
    def start_new_shard():
        nonlocal current_shard_num, current_writer, current_shard_samples
        if current_writer:
            current_writer.close()
            print(f"  Completed shard {current_shard_num-1}: {len(current_shard_samples)} samples")
        
        shard_path = output_pattern % current_shard_num
        current_writer = wds.TarWriter(shard_path)
        current_shard_samples = []
        current_shard_num += 1
        return current_writer
    
    def finish_current_shard():
        nonlocal current_writer
        if current_writer:
            current_writer.close()
            print(f"  Completed shard {current_shard_num-1}: {len(current_shard_samples)} samples")
            current_writer = None
    
    print("\nProcessing samples...")
    print(f"Extracting images with licenses: {', '.join([license_names[lid] for lid in PERMISSIVE_LICENSES])}")
    print("-" * 60)
    
    # Start first shard
    writer = start_new_shard()
    
    for sample in tqdm(dataset, desc="Filtering samples"):
        stats['total_processed'] += 1
        
        # Parse JSON metadata
        json_data = sample.get("json", sample.get("json".encode(), b"{}"))
        if isinstance(json_data, bytes):
            metadata = json.loads(json_data)
        else:
            metadata = json_data
        
        # Check license
        license_id = metadata.get('license', 0)
        license_name = license_names.get(license_id, f'Unknown License {license_id}')
        
        if license_id in PERMISSIVE_LICENSES:
            # This is a permissive licensed image - include it
            stats['license_counts'][license_name] += 1
            stats['total_extracted'] += 1
            
            # Get image data (could be jpg or png)
            img_data = None
            img_key = None
            for key in ['jpg', 'png', 'jpeg']:
                if key in sample:
                    img_data = sample[key]
                    img_key = key
                    break
                # Try encoded versions
                key_encoded = key.encode()
                if key_encoded in sample:
                    img_data = sample[key_encoded]
                    img_key = key
                    break
            
            if img_data:
                # Write sample to shard
                sample_to_write = {
                    "__key__": sample["__key__"],
                    img_key: img_data,
                    "json": json_data if isinstance(json_data, bytes) else json.dumps(metadata).encode('utf-8')
                }
                
                writer.write(sample_to_write)
                current_shard_samples.append(sample["__key__"])
                
                # Check if we need a new shard
                if len(current_shard_samples) >= samples_per_shard:
                    writer = start_new_shard()
            else:
                print(f"Warning: No image data found for sample {sample.get('__key__', 'unknown')}")
        else:
            # Skip non-permissive licenses
            stats['skipped_licenses'][license_name] += 1
    
    # Finish the last shard
    finish_current_shard()
    
    # Print statistics
    print("\n" + "="*60)
    print("Extraction Complete!")
    print("="*60)
    print(f"\n📊 **Processing Summary**")
    print(f"  Total samples processed: {stats['total_processed']:,}")
    print(f"  Samples extracted: {stats['total_extracted']:,}")
    print(f"  Extraction rate: {stats['total_extracted']/stats['total_processed']*100:.1f}%")
    print(f"  Output shards created: {current_shard_num}")
    
    print(f"\n✅ **Extracted Licenses**")
    for license_name, count in sorted(stats['license_counts'].items()):
        print(f"  {license_name}: {count:,} images")
    
    print(f"\n❌ **Skipped Licenses**")
    for license_name, count in sorted(stats['skipped_licenses'].items(), key=lambda x: -x[1])[:10]:
        print(f"  {license_name}: {count:,} images")
    
    # List output files
    output_files = sorted(glob.glob(f"{output_dir}/coco_train2017_permissive-*.tar"))
    if output_files:
        print(f"\n📁 **Output Files**")
        total_size = 0
        for i, filepath in enumerate(output_files[:10]):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            total_size += os.path.getsize(filepath)
            print(f"  {os.path.basename(filepath)} ({size_mb:.1f} MB)")
        
        if len(output_files) > 10:
            print(f"  ... and {len(output_files) - 10} more files")
            for filepath in output_files[10:]:
                total_size += os.path.getsize(filepath)
        
        print(f"\n  Total size: {total_size / (1024 * 1024 * 1024):.2f} GB")
    
    print("="*60)
    
    # Save extraction statistics
    stats_file = os.path.join(output_dir, "extraction_stats.json")
    with open(stats_file, 'w') as f:
        json.dump({
            'total_processed': stats['total_processed'],
            'total_extracted': stats['total_extracted'],
            'extraction_rate': stats['total_extracted']/stats['total_processed'] if stats['total_processed'] > 0 else 0,
            'output_shards': current_shard_num,
            'samples_per_shard': samples_per_shard,
            'extracted_licenses': dict(stats['license_counts']),
            'skipped_licenses': dict(stats['skipped_licenses'])
        }, f, indent=2)
    print(f"\nStatistics saved to: {stats_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Extract permissive licensed COCO images to new WebDataset")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/capstor/store/cscs/swissai/infra01/vision-datasets/coco2017/webdataset_output",
        help="Input directory containing COCO WebDataset tar files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/capstor/store/cscs/swissai/infra01/vision-datasets/coco2017/webdataset_permissive",
        help="Output directory for filtered WebDataset tar files"
    )
    parser.add_argument(
        "--samples_per_shard",
        type=int,
        default=1000,
        help="Number of samples per output shard"
    )
    
    args = parser.parse_args()
    
    # Run extraction
    extract_permissive_licensed_data(
        args.input_dir,
        args.output_dir,
        args.samples_per_shard
    )


if __name__ == "__main__":
    main()