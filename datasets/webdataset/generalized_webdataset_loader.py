#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any, Union
import webdataset as wds
from PIL import Image
from io import BytesIO
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm
import pickle
from enum import Enum


class DatasetType(Enum):
    """Enum for supported dataset types."""
    LLAVA = "llava"
    COCO = "coco"
    GENERIC = "generic"


# Global functions for processing that can be pickled
def process_llava_sample(sample):
    """Process a LLaVA format sample with conversations."""
    try:
        key = sample["__key__"]
        image_data = sample["jpg"]
        json_data = sample["json"]
        
        # Parse JSON if it's bytes or string
        if isinstance(json_data, bytes):
            json_data = json.loads(json_data)
        elif isinstance(json_data, str):
            json_data = json.loads(json_data)
        
        # Extract conversations
        conversations = json_data.get("conversations", [])
        
        # Get human and assistant messages
        human_msg = ""
        assistant_msg = ""
        for conv in conversations:
            if conv.get("from") == "human":
                human_msg = conv.get("value", "")
            elif conv.get("from") == "gpt":
                assistant_msg = conv.get("value", "")
        
        return {
            "key": key,
            "image": image_data,
            "json": json_data,
            "conversations": conversations,
            "human_msg": human_msg,
            "assistant_msg": assistant_msg,
            "has_text": len(conversations) > 0
        }
        
    except Exception as e:
        print(f"Error processing LLaVA sample {sample.get('__key__', 'unknown')}: {e}")
        return None


def process_coco_sample(sample):
    """Process a COCO format sample with annotations."""
    try:
        key = sample["__key__"]
        image_data = sample["jpg"]
        json_data = sample["json"]
        
        # Parse JSON if it's bytes or string
        if isinstance(json_data, bytes):
            json_data = json.loads(json_data)
        elif isinstance(json_data, str):
            json_data = json.loads(json_data)
        
        # Extract COCO specific fields
        image_id = json_data.get("image_id", key)
        instances = json_data.get("instances", [])
        captions = json_data.get("captions", [])
        keypoints = json_data.get("keypoints", [])
        categories = json_data.get("categories", [])
        
        # Image metadata
        height = json_data.get("height", 0)
        width = json_data.get("width", 0)
        
        return {
            "key": key,
            "image": image_data,
            "json": json_data,
            "image_id": image_id,
            "instances": instances,
            "captions": captions,
            "keypoints": keypoints,
            "categories": categories,
            "height": height,
            "width": width,
            "has_annotations": len(instances) > 0,
            "has_captions": len(captions) > 0,
            "has_keypoints": len(keypoints) > 0
        }
        
    except Exception as e:
        print(f"Error processing COCO sample {sample.get('__key__', 'unknown')}: {e}")
        return None


def process_generic_sample(sample):
    """Process a generic WebDataset sample."""
    try:
        key = sample["__key__"]
        image_data = sample["jpg"]
        json_data = sample.get("json", {})
        
        # Parse JSON if it's bytes or string
        if isinstance(json_data, bytes):
            json_data = json.loads(json_data)
        elif isinstance(json_data, str):
            json_data = json.loads(json_data)
        
        return {
            "key": key,
            "image": image_data,
            "json": json_data
        }
        
    except Exception as e:
        print(f"Error processing generic sample {sample.get('__key__', 'unknown')}: {e}")
        return None


def is_not_none(x):
    """Filter function to remove None values."""
    return x is not None


def collate_llava(samples):
    """Collate function for LLaVA samples."""
    if not samples:
        return {}
    
    batch = {
        "keys": [s["key"] for s in samples],
        "images": [s["image"] for s in samples],
        "conversations": [s["conversations"] for s in samples],
        "human_msgs": [s["human_msg"] for s in samples],
        "assistant_msgs": [s["assistant_msg"] for s in samples],
        "has_text": [s["has_text"] for s in samples]
    }
    
    # Add metadata if present
    if "metadata" in samples[0]:
        batch["metadata"] = [s["metadata"] for s in samples]
    
    return batch


def collate_coco(samples):
    """Collate function for COCO samples."""
    if not samples:
        return {}
    
    batch = {
        "keys": [s["key"] for s in samples],
        "images": [s["image"] for s in samples],
        "image_ids": [s["image_id"] for s in samples],
        "instances": [s["instances"] for s in samples],
        "captions": [s["captions"] for s in samples],
        "has_annotations": [s["has_annotations"] for s in samples],
        "has_captions": [s["has_captions"] for s in samples]
    }
    
    # Add metadata if present
    if "metadata" in samples[0]:
        batch["metadata"] = [s["metadata"] for s in samples]
    
    # Add dimensions
    batch["heights"] = [s["height"] for s in samples]
    batch["widths"] = [s["width"] for s in samples]
    
    return batch


def collate_generic(samples):
    """Collate function for generic samples."""
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


class GeneralizedWebDatasetLoader:
    """
    Generalized WebDataset loader for multi-modal datasets.
    Supports LLaVA (with conversations), COCO (with annotations), and other formats.
    Optimized for distributed multi-GPU training.
    """
    
    def __init__(
        self, 
        dataset_path: str,
        dataset_type: Union[str, DatasetType] = DatasetType.GENERIC,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        buffer_size: int = 1000,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True
    ):
        """
        Initialize the generalized dataloader.
        
        Args:
            dataset_path: Path to WebDataset directory containing .tar files
            dataset_type: Type of dataset ('llava', 'coco', or 'generic')
            batch_size: Batch size per GPU
            num_workers: Number of worker processes for data loading
            shuffle: Whether to shuffle shards and samples
            buffer_size: Buffer size for shuffling
            world_size: Total number of GPUs (auto-detected if dist.is_initialized())
            rank: Current GPU rank (auto-detected if dist.is_initialized())
            drop_last: Whether to drop the last incomplete batch
            prefetch_factor: Number of batches to prefetch per worker
            persistent_workers: Keep workers alive between epochs
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers and num_workers > 0
        
        # Handle dataset type
        if isinstance(dataset_type, str):
            dataset_type = DatasetType(dataset_type.lower())
        self.dataset_type = dataset_type
        
        # Distributed setup
        self.distributed = dist.is_initialized()
        if self.distributed:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            if self.rank == 0:
                print(f"Distributed training: world_size={self.world_size}")
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
            # Single tar file
            self.tar_files = [dataset_path]
        
        if not self.tar_files:
            raise ValueError(f"No .tar files found in {dataset_path}")
        
        if self.rank == 0:
            print(f"Found {len(self.tar_files)} shard files")
            print(f"Dataset type: {self.dataset_type.value}")
            print(f"Distributed setup: rank {self.rank}/{self.world_size}")
    
    def create_dataset(
        self,
        decode_images: bool = True,
        image_format: str = "PIL",  # "PIL" or "torch"
        return_metadata: bool = False,
        filter_fn: Optional[callable] = None,
        transform: Optional[callable] = None
    ):
        """
        Create a WebDataset for distributed loading.
        
        Args:
            decode_images: Whether to decode images
            image_format: "PIL" for PIL Images, "torch" for tensors
            return_metadata: Whether to include image metadata
            filter_fn: Optional function to filter samples
            transform: Optional transform to apply to images
            
        Returns:
            WebDataset ready for DataLoader
        """
        # Create WebDataset with proper distributed sharding
        dataset = wds.WebDataset(
            self.tar_files,
            shardshuffle=1000 if self.shuffle else False,
            nodesplitter=wds.split_by_node if self.distributed else None,
            workersplitter=wds.split_by_worker
        )
        
        # Decode images if requested
        if decode_images:
            if image_format == "torch":
                dataset = dataset.decode("torchrgb8")
            else:
                dataset = dataset.decode("pilrgb")
        
        # Select processing function based on dataset type (using global functions)
        if self.dataset_type == DatasetType.LLAVA:
            dataset = dataset.map(process_llava_sample, handler=wds.ignore_and_continue)
        elif self.dataset_type == DatasetType.COCO:
            dataset = dataset.map(process_coco_sample, handler=wds.ignore_and_continue)
        else:
            dataset = dataset.map(process_generic_sample, handler=wds.ignore_and_continue)
        
        # Filter None values
        dataset = dataset.select(is_not_none)
        
        # Apply custom filter if provided
        if filter_fn:
            dataset = dataset.select(filter_fn)
        
        # Apply transforms if provided
        if transform and decode_images:
            def apply_transform(sample):
                sample["image"] = transform(sample["image"])
                return sample
            dataset = dataset.map(apply_transform)
        
        # Add metadata if requested
        if return_metadata and decode_images:
            def add_metadata(sample):
                img = sample["image"]
                if hasattr(img, 'size'):  # PIL Image
                    width, height = img.size
                elif hasattr(img, 'shape'):  # Torch tensor
                    if len(img.shape) == 3:
                        _, height, width = img.shape
                    else:
                        height, width = img.shape[-2:]
                else:
                    width, height = 0, 0
                
                sample["metadata"] = {
                    "width": width,
                    "height": height,
                    "resolution": width * height,
                    "aspect_ratio": width / height if height > 0 else 0
                }
                return sample
            dataset = dataset.map(add_metadata)
        
        # Shuffle if requested
        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size)
        
        return dataset
    
    def create_dataloader(self, dataset, collate_fn=None):
        """
        Create a PyTorch DataLoader from the WebDataset.
        
        Args:
            dataset: WebDataset instance
            collate_fn: Optional custom collate function
            
        Returns:
            DataLoader instance
        """
        
        if collate_fn is None:
            # Use global collate functions based on dataset type
            if self.dataset_type == DatasetType.LLAVA:
                collate_fn = collate_llava
            elif self.dataset_type == DatasetType.COCO:
                collate_fn = collate_coco
            else:
                collate_fn = collate_generic
        
        # Create batched dataset
        dataset = dataset.batched(self.batch_size, collation_fn=collate_fn)
        
        # Create DataLoader with optimized settings for multi-GPU
        return DataLoader(
            dataset,
            batch_size=None,  # Batching handled by WebDataset
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers
        )
    
    def get_dataset_info(self) -> Dict:
        """Get information about the dataset."""
        info = {
            "dataset_path": self.dataset_path,
            "dataset_type": self.dataset_type.value,
            "num_shards": len(self.tar_files),
            "batch_size": self.batch_size,
            "world_size": self.world_size,
            "rank": self.rank,
            "distributed": self.distributed,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers
        }
        
        # Estimate total samples (requires scanning)
        if self.rank == 0:
            print("Estimating dataset size...")
            sample_count = 0
            for tar_file in tqdm(self.tar_files[:5], desc="Sampling shards"):
                try:
                    ds = wds.WebDataset(tar_file, shardshuffle=False)
                    for _ in ds:
                        sample_count += 1
                except:
                    pass
            
            avg_samples_per_shard = sample_count / min(5, len(self.tar_files))
            estimated_total = int(avg_samples_per_shard * len(self.tar_files))
            info["estimated_samples"] = estimated_total
            info["samples_per_shard_avg"] = avg_samples_per_shard
        
        return info


def test_dataset_loading(
    dataset_path: str,
    dataset_type: str,
    batch_size: int = 32,
    num_batches: int = 5
):
    """Test loading functionality for a dataset."""
    
    print(f"\n{'='*60}")
    print(f"Testing {dataset_type.upper()} dataset loading")
    print(f"Dataset path: {dataset_path}")
    print(f"{'='*60}")
    
    # Create loader
    loader = GeneralizedWebDatasetLoader(
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )
    
    # Get dataset info
    info = loader.get_dataset_info()
    print("\nDataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Create dataset and dataloader
    dataset = loader.create_dataset(
        decode_images=True,
        image_format="PIL",
        return_metadata=True
    )
    dataloader = loader.create_dataloader(dataset)
    
    print(f"\n🧪 Testing batch loading...")
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        print(f"\nBatch {i+1}:")
        print(f"  Batch size: {len(batch['keys'])}")
        
        if batch['images']:
            first_img = batch['images'][0]
            if hasattr(first_img, 'size'):
                print(f"  First image size: {first_img.size}")
            
        print(f"  Keys: {batch['keys'][:3]}")
        
        if dataset_type == "llava":
            print(f"  Has text: {batch['has_text'][:3]}")
            if batch['human_msgs']:
                msg = batch['human_msgs'][0]
                print(f"  First human msg: {msg[:100]}..." if len(msg) > 100 else f"  First human msg: {msg}")
        elif dataset_type == "coco":
            print(f"  Image IDs: {batch['image_ids'][:3]}")
            print(f"  Has annotations: {batch['has_annotations'][:3]}")
            print(f"  Has captions: {batch['has_captions'][:3]}")
        
        if "metadata" in batch:
            print(f"  Resolutions: {[m['resolution'] for m in batch['metadata'][:3]]}")
    
    print(f"\n✅ {dataset_type.upper()} dataset loading test completed!")


def main():
    parser = argparse.ArgumentParser(description="Generalized WebDataset Loader")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to WebDataset directory"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["llava", "coco", "generic"],
        default="generic",
        help="Type of dataset"
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
        "--test",
        action="store_true",
        help="Run test loading"
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=5,
        help="Number of batches to test"
    )
    
    args = parser.parse_args()
    
    if args.test:
        test_dataset_loading(
            args.dataset_path,
            args.dataset_type,
            args.batch_size,
            args.num_batches
        )
    else:
        # Create loader
        loader = GeneralizedWebDatasetLoader(
            dataset_path=args.dataset_path,
            dataset_type=args.dataset_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Get and print dataset info
        info = loader.get_dataset_info()
        print("\nDataset Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()