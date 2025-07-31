#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any
import webdataset as wds
from PIL import Image
from io import BytesIO
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm
import pickle


class DistributedLLaVAWebDataset:
    """
    Distributed WebDataset loader for LLaVA 558K dataset.
    Supports distributed training and efficient data loading.
    """
    
    def __init__(
        self, 
        dataset_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        buffer_size: int = 1000,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last: bool = True
    ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.drop_last = drop_last
        
        # Distributed setup
        self.distributed = dist.is_initialized()
        if self.distributed:
            self.world_size = dist.get_world_size()
            print("world_size: ", self.world_size)
            self.rank = dist.get_rank()
        else:
            self.world_size = world_size or 1
            self.rank = rank or 0
        
        # Find all tar files
        if os.path.isdir(dataset_path):
            self.tar_files = sorted([
                os.path.join(dataset_path, f) 
                for f in os.listdir(dataset_path) 
                if f.endswith('.tar')
            ])
        else:
            self.tar_files = [dataset_path]
        
        if self.rank == 0:
            print(f"Found {len(self.tar_files)} shard files")
            print(f"Distributed setup: rank {self.rank}/{self.world_size}")
    
    def create_dataset(
        self,
        decode_images: bool = True,
        image_format: str = "PIL",  # "PIL" or "torch"
        return_metadata: bool = False,
        filter_fn: Optional[callable] = None
    ):
        """
        Create a WebDataset for distributed loading.
        
        Args:
            decode_images: Whether to decode images
            image_format: "PIL" for PIL Images, "torch" for tensors
            return_metadata: Whether to include image metadata (width, height, etc.)
            filter_fn: Optional function to filter samples
            
        Returns:
            WebDataset ready for DataLoader
        """
        # Create WebDataset with proper distributed sharding
        # Use WebDataset's built-in distributed splitting instead of manual sharding
        dataset = wds.WebDataset(
            self.tar_files,
            shardshuffle=1000 if self.shuffle else False,
            nodesplitter=wds.split_by_node if self.distributed else None,
            workersplitter=wds.split_by_worker
        )
        
        # Decode images if requested
        if decode_images:
            if image_format == "torch":
                dataset = dataset.decode("torchrgb8")  # Decode to torch tensors
            else:
                dataset = dataset.decode("pilrgb")     # Decode to PIL images
        
        # Process samples
        def process_sample(sample):
            try:
                key = sample["__key__"]
                image_data = sample["jpg"]
                json_data = sample["json"]
                
                # Parse JSON if string
                if isinstance(json_data, str):
                    json_data = json.loads(json_data)
                
                # Get image metadata if requested
                metadata = {}
                if return_metadata or not decode_images:
                    if decode_images:
                        # Image already decoded
                        if hasattr(image_data, 'size'):
                            width, height = image_data.size
                        else:  # torch tensor
                            height, width = image_data.shape[-2:]
                    else:
                        # Need to peek at image without full decode
                        if isinstance(image_data, bytes):
                            temp_img = Image.open(BytesIO(image_data))
                            width, height = temp_img.size
                        else:
                            width, height = image_data.size
                    
                    metadata = {
                        "width": width,
                        "height": height,
                        "resolution": width * height,
                        "aspect_ratio": width / height
                    }
                
                result = {
                    "key": key,
                    "image": image_data,
                    "json": json_data,
                }
                
                if return_metadata:
                    result["metadata"] = metadata
                
                return result
                
            except Exception as e:
                if self.rank == 0:
                    print(f"Error processing sample: {e}")
                return None
        
        # Apply processing
        dataset = dataset.map(process_sample, handler=wds.ignore_and_continue)
        
        # Filter None values
        dataset = dataset.select(lambda x: x is not None)
        
        # Apply custom filter if provided
        if filter_fn:
            dataset = dataset.select(filter_fn)
        
        # Shuffle if requested
        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size)
        
        return dataset
    
    def create_dataloader(self, dataset):
        """Create a PyTorch DataLoader from the WebDataset."""
        
        def collate_fn(samples):
            """Collate function for batched samples."""
            if not samples:
                return {}
            
            batch = {
                "keys": [s["key"] for s in samples],
                "images": [s["image"] for s in samples], 
                "jsons": [s["json"] for s in samples]
            }
            
            # Add metadata if present
            if "metadata" in samples[0]:
                batch["metadata"] = [s["metadata"] for s in samples]
            
            return batch
        
        # Create batched dataset
        dataset = dataset.batched(self.batch_size, collation_fn=collate_fn)
        
        return DataLoader(
            dataset,
            batch_size=None,  # Batching handled by WebDataset
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            # drop_last not compatible with batch_size=None
        )
    
    def analyze_resolution_distribution(
        self,
        max_samples: Optional[int] = None,
        save_results: bool = True,
        output_file: str = "resolution_analysis_distributed.pkl"
    ) -> Dict:
        """
        Analyze resolution distribution using distributed loading.
        
        Args:
            max_samples: Maximum samples to analyze per process
            save_results: Whether to save results
            output_file: Output file path
            
        Returns:
            Analysis results dictionary
        """
        if self.rank == 0:
            print("🔍 Starting distributed resolution analysis...")
        
        # Create dataset for analysis (don't decode images, just get metadata)
        dataset = self.create_dataset(
            decode_images=False,
            return_metadata=True
        )
        dataloader = self.create_dataloader(dataset)
        
        # Data collection
        resolutions = []
        widths = []
        heights = []
        aspect_ratios = []
        samples_processed = 0
        
        # Progress tracking (only on rank 0)
        if self.rank == 0:
            pbar = tqdm(desc=f"Analyzing (rank {self.rank})", unit="batches")
        
        try:
            for batch in dataloader:
                if max_samples and samples_processed >= max_samples:
                    break
                
                batch_metadata = batch["metadata"]
                
                for meta in batch_metadata:
                    resolutions.append(meta["resolution"])
                    widths.append(meta["width"])
                    heights.append(meta["height"])
                    aspect_ratios.append(meta["aspect_ratio"])
                    samples_processed += 1
                    
                    if max_samples and samples_processed >= max_samples:
                        break
                
                if self.rank == 0:
                    pbar.update(1)
                    pbar.set_postfix({"samples": samples_processed})
                
        except KeyboardInterrupt:
            if self.rank == 0:
                print(f"\\n⚠️ Analysis interrupted after {samples_processed} samples")
        
        if self.rank == 0:
            pbar.close()
        
        # Convert to numpy arrays
        resolutions = np.array(resolutions)
        widths = np.array(widths)
        heights = np.array(heights)
        aspect_ratios = np.array(aspect_ratios)
        
        # Gather results from all processes if distributed
        if self.distributed:
            # Gather data from all processes
            all_resolutions = self._gather_array(resolutions)
            all_widths = self._gather_array(widths)
            all_heights = self._gather_array(heights)
            all_aspect_ratios = self._gather_array(aspect_ratios)
        else:
            all_resolutions = resolutions
            all_widths = widths
            all_heights = heights
            all_aspect_ratios = aspect_ratios
        
        # Only rank 0 computes final statistics and saves
        if self.rank == 0:
            total_samples = len(all_resolutions)
            
            if total_samples == 0:
                print("❌ No samples processed")
                return {}
            
            # Compute statistics
            results = {
                "total_samples": total_samples,
                "world_size": self.world_size,
                "resolutions": {
                    "min": int(all_resolutions.min()),
                    "max": int(all_resolutions.max()),
                    "mean": float(all_resolutions.mean()),
                    "median": float(np.median(all_resolutions)),
                    "std": float(all_resolutions.std()),
                    "percentiles": {
                        "5": float(np.percentile(all_resolutions, 5)),
                        "25": float(np.percentile(all_resolutions, 25)),
                        "75": float(np.percentile(all_resolutions, 75)),
                        "95": float(np.percentile(all_resolutions, 95))
                    }
                },
                "widths": {
                    "min": int(all_widths.min()),
                    "max": int(all_widths.max()),
                    "mean": float(all_widths.mean()),
                    "std": float(all_widths.std())
                },
                "heights": {
                    "min": int(all_heights.min()),
                    "max": int(all_heights.max()),
                    "mean": float(all_heights.mean()),
                    "std": float(all_heights.std())
                },
                "aspect_ratios": {
                    "min": float(all_aspect_ratios.min()),
                    "max": float(all_aspect_ratios.max()),
                    "mean": float(all_aspect_ratios.mean()),
                    "std": float(all_aspect_ratios.std())
                },
                "common_resolutions": self._get_common_resolutions(all_widths, all_heights),
                "resolution_distribution": self._get_resolution_distribution(all_resolutions)
            }
            
            # Save results
            if save_results:
                with open(output_file, 'wb') as f:
                    pickle.dump(results, f)
                print(f"📊 Results saved to {output_file}")
            
            # Print summary
            self._print_summary(results)
            
            return results
        else:
            return {}
    
    def _gather_array(self, array: np.ndarray) -> np.ndarray:
        """Gather arrays from all processes."""
        # Convert to tensor for gathering
        tensor = torch.from_numpy(array)
        
        # Gather sizes first
        local_size = torch.tensor([len(array)], dtype=torch.int64)
        size_list = [torch.zeros_like(local_size) for _ in range(self.world_size)]
        dist.all_gather(size_list, local_size)
        
        # Pad tensor to max size for gathering
        max_size = max(size_list).item()
        if len(tensor) < max_size:
            padding = torch.zeros(max_size - len(tensor), dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding])
        
        # Gather tensors
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)
        
        # Concatenate valid portions
        result_parts = []
        for i, (t, size) in enumerate(zip(tensor_list, size_list)):
            if size.item() > 0:
                result_parts.append(t[:size.item()])
        
        if result_parts:
            return torch.cat(result_parts).numpy()
        else:
            return np.array([])
    
    def _get_common_resolutions(self, widths: np.ndarray, heights: np.ndarray) -> List[Tuple]:
        """Get most common width x height combinations."""
        resolution_pairs = list(zip(widths, heights))
        counter = Counter(resolution_pairs)
        return counter.most_common(20)
    
    def _get_resolution_distribution(self, resolutions: np.ndarray) -> Dict:
        """Get distribution of resolutions in buckets."""
        buckets = {
            "very_low": (0, 100000),
            "low": (100000, 500000),
            "medium": (500000, 1000000),
            "high": (1000000, 2000000),
            "very_high": (2000000, float('inf'))
        }
        
        distribution = {}
        for bucket_name, (min_res, max_res) in buckets.items():
            count = np.sum((resolutions >= min_res) & (resolutions < max_res))
            distribution[bucket_name] = {
                "count": int(count),
                "percentage": float(count / len(resolutions) * 100),
                "range": f"{min_res:,} - {max_res:,}" if max_res != float('inf') else f"> {min_res:,}"
            }
        
        return distribution
    
    def _print_summary(self, results: Dict):
        """Print comprehensive analysis results."""
        print("\\n" + "="*80)
        print("📊 COMPREHENSIVE DISTRIBUTED RESOLUTION ANALYSIS")
        print("="*80)
        
        # Basic info
        print(f"Total samples analyzed: {results['total_samples']:,}")
        print(f"Distributed across {results['world_size']} processes")
        
        # Resolution statistics
        res = results['resolutions']
        print(f"\\n🔍 RESOLUTION STATISTICS (pixels):")
        print(f"  Minimum: {res['min']:,}")
        print(f"  Maximum: {res['max']:,}")
        print(f"  Mean: {res['mean']:,.0f}")
        print(f"  Median: {res['median']:,.0f}")
        print(f"  Standard Deviation: {res['std']:,.0f}")
        print(f"\\n  📐 Percentiles:")
        for p, val in res['percentiles'].items():
            print(f"    {p}th percentile: {val:,.0f}")
        
        # Width statistics
        w = results['widths']
        print(f"\\n📏 WIDTH STATISTICS (pixels):")
        print(f"  Minimum: {w['min']:,}")
        print(f"  Maximum: {w['max']:,}")
        print(f"  Mean: {w['mean']:,.0f}")
        print(f"  Standard Deviation: {w['std']:,.0f}")
        
        # Height statistics
        h = results['heights']
        print(f"\\n📏 HEIGHT STATISTICS (pixels):")
        print(f"  Minimum: {h['min']:,}")
        print(f"  Maximum: {h['max']:,}")
        print(f"  Mean: {h['mean']:,.0f}")
        print(f"  Standard Deviation: {h['std']:,.0f}")
        
        # Aspect ratio statistics
        ar = results['aspect_ratios']
        print(f"\\n📐 ASPECT RATIO STATISTICS:")
        print(f"  Minimum: {ar['min']:.3f}")
        print(f"  Maximum: {ar['max']:.3f}")
        print(f"  Mean: {ar['mean']:.3f}")
        print(f"  Standard Deviation: {ar['std']:.3f}")
        
        # Resolution distribution
        print(f"\\n📊 RESOLUTION DISTRIBUTION:")
        for bucket, info in results['resolution_distribution'].items():
            print(f"  {bucket.replace('_', ' ').title():12s}: {info['count']:6,} ({info['percentage']:5.1f}%) - {info['range']} pixels")
        
        # All common resolutions (not just top 10)
        print(f"\\n🏆 ALL COMMON RESOLUTIONS (showing all with >0.1% occurrence):")
        print("    Rank  Resolution    Count      Percentage")
        print("    " + "-" * 45)
        total_samples = results['total_samples']
        for i, ((w, h), count) in enumerate(results['common_resolutions']):
            pct = count / total_samples * 100
            if pct >= 0.1:  # Show resolutions with at least 0.1%
                print(f"    {i+1:4d}. {w:4d}x{h:<4d}    {count:6,}    {pct:6.2f}%")
            elif i < 20:  # Always show top 20
                print(f"    {i+1:4d}. {w:4d}x{h:<4d}    {count:6,}    {pct:6.2f}%")
        
        # Additional insights
        print(f"\\n💡 INSIGHTS:")
        
        # Square vs non-square
        square_count = sum(count for (w, h), count in results['common_resolutions'] if w == h)
        square_pct = square_count / total_samples * 100
        print(f"  Square images (W=H): {square_count:,} ({square_pct:.1f}%)")
        
        # Landscape vs portrait
        landscape_count = sum(count for (w, h), count in results['common_resolutions'] if w > h)
        portrait_count = sum(count for (w, h), count in results['common_resolutions'] if w < h)
        landscape_pct = landscape_count / total_samples * 100
        portrait_pct = portrait_count / total_samples * 100
        print(f"  Landscape images (W>H): {landscape_count:,} ({landscape_pct:.1f}%)")
        print(f"  Portrait images (W<H): {portrait_count:,} ({portrait_pct:.1f}%)")
        
        # Common aspect ratios
        print(f"\\n📐 COMMON ASPECT RATIOS:")
        aspect_ratio_counts = {}
        for (w, h), count in results['common_resolutions']:
            ratio = round(w/h, 2)
            aspect_ratio_counts[ratio] = aspect_ratio_counts.get(ratio, 0) + count
        
        sorted_ratios = sorted(aspect_ratio_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (ratio, count) in enumerate(sorted_ratios[:15]):
            pct = count / total_samples * 100
            if ratio == 1.0:
                ratio_str = "1:1 (Square)"
            elif abs(ratio - 1.33) < 0.05:
                ratio_str = "4:3"
            elif abs(ratio - 1.78) < 0.05:
                ratio_str = "16:9"
            else:
                ratio_str = f"{ratio:.2f}:1"
            print(f"    {i+1:2d}. {ratio_str:15s}: {count:6,} ({pct:5.1f}%)")
        
        # Size categories
        print(f"\\n📏 SIZE CATEGORIES:")
        very_small = sum(1 for (w, h), count in results['common_resolutions'] 
                        for _ in range(count) if w * h < 150000)
        small = sum(1 for (w, h), count in results['common_resolutions'] 
                   for _ in range(count) if 150000 <= w * h < 300000)
        medium = sum(1 for (w, h), count in results['common_resolutions'] 
                    for _ in range(count) if 300000 <= w * h < 500000)
        large = sum(1 for (w, h), count in results['common_resolutions'] 
                   for _ in range(count) if w * h >= 500000)
        
        print(f"  Very Small (<150K pixels): {very_small:,} ({very_small/total_samples*100:.1f}%)")
        print(f"  Small (150K-300K pixels): {small:,} ({small/total_samples*100:.1f}%)")
        print(f"  Medium (300K-500K pixels): {medium:,} ({medium/total_samples*100:.1f}%)")
        print(f"  Large (>500K pixels): {large:,} ({large/total_samples*100:.1f}%)")
        
        print("\\n" + "="*80)


def test_distributed_loading(dataset_path: str, batch_size: int = 32, num_batches: int = 5):
    """Test basic distributed loading functionality."""
    loader = DistributedLLaVAWebDataset(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )
    
    # Create dataset and dataloader
    dataset = loader.create_dataset(decode_images=True, return_metadata=True)
    dataloader = loader.create_dataloader(dataset)
    
    print(f"\\n🧪 Testing distributed loading (rank {loader.rank})...")
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        print(f"Batch {i+1} (rank {loader.rank}):")
        print(f"  Batch size: {len(batch['keys'])}")
        print(f"  First image size: {batch['images'][0].size}")
        print(f"  Sample resolutions: {[m['resolution'] for m in batch['metadata'][:3]]}")
        print(f"  Sample IDs: {batch['keys'][:3]}")
    
    print("✅ Distributed loading test completed!")


def main():
    parser = argparse.ArgumentParser(description="Distributed LLaVA WebDataset Loader")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain/webdataset_output_filtered",
        help="Path to WebDataset directory"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "analyze"],
        default="test",
        help="Mode: test loading or analyze resolutions"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for loading"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to analyze"
    )
    
    args = parser.parse_args()
    
    if args.mode == "test":
        test_distributed_loading(args.dataset_path, args.batch_size)
        
    elif args.mode == "analyze":
        loader = DistributedLLaVAWebDataset(
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False  # For consistent analysis
        )
        
        results = loader.analyze_resolution_distribution(
            max_samples=args.max_samples
        )


if __name__ == "__main__":
    main()