#!/usr/bin/env python3
"""
Multi-GPU parallel CLIP embedding extraction from webdataset format.
Optimized for systems with multiple GPUs and high CPU count.
"""

import os
import json
import torch
import torch.multiprocessing as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse
import webdataset as wds
import open_clip
from typing import Dict, List, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
from queue import Queue
import threading

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logger(gpu_id: int = 0) -> logging.Logger:
    """Setup logger for each GPU process."""
    logger = logging.getLogger(f'GPU_{gpu_id}')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f'[GPU {gpu_id}] %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_shard_splits(num_shards: int, num_gpus: int) -> List[List[int]]:
    """
    Split shards across GPUs evenly.
    
    Args:
        num_shards: Total number of shards
        num_gpus: Number of GPUs to use
    
    Returns:
        List of shard indices for each GPU
    """
    shards_per_gpu = num_shards // num_gpus
    remainder = num_shards % num_gpus
    
    splits = []
    start = 0
    for i in range(num_gpus):
        # Add extra shard to first GPUs if there's a remainder
        count = shards_per_gpu + (1 if i < remainder else 0)
        splits.append(list(range(start, start + count)))
        start += count
    
    return splits


def process_batch_gpu(batch, model, preprocess, device="cuda"):
    """
    Process a batch of images on specified GPU.
    
    Args:
        batch: Batch from webdataset
        model: CLIP model
        preprocess: Preprocessing function
        device: Device to run on
    
    Returns:
        Tuple of (keys, embeddings)
    """
    # Handle both formats: list of tuples or tuple of lists
    if len(batch) == 3 and isinstance(batch[0], list):
        # Tuple of lists format (batched)
        keys, imgs, json_datas = batch
    else:
        # List of tuples format
        keys = [item[0] for item in batch]
        imgs = [item[1] for item in batch]
        json_datas = [item[2] for item in batch]
    
    # Preprocess images
    processed_imgs = []
    valid_keys = []
    
    for key, img in zip(keys, imgs):
        try:
            processed_img = preprocess(img)
            processed_imgs.append(processed_img)
            valid_keys.append(key)
        except Exception as e:
            print(f"Error processing image {key}: {e}")
            continue
    
    if not processed_imgs:
        return [], np.array([])
    
    # Stack images
    image_tensors = torch.stack(processed_imgs).to(device)
    
    # Extract features
    with torch.no_grad():
        image_features = model.encode_image(image_tensors)
        # Normalize features (required for LAION aesthetic model)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().numpy()
    
    return valid_keys, image_features


def process_shard_range(
    gpu_id: int,
    shard_indices: List[int],
    tar_files: List[Path],
    config: Dict,
    output_queue: mp.Queue,
    progress_queue: mp.Queue
):
    """
    Process a range of shards on a specific GPU.
    
    Args:
        gpu_id: GPU device ID
        shard_indices: List of shard indices to process
        tar_files: List of all tar files
        config: Configuration dictionary
        output_queue: Queue for output results
        progress_queue: Queue for progress updates
    """
    # Set GPU
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    logger = setup_logger(gpu_id)
    
    logger.info(f"Processing {len(shard_indices)} shards: {shard_indices[:5]}...")
    
    # Load CLIP model
    model_name = config['clip']['model_name']
    pretrained = config['clip']['pretrained']
    
    logger.info(f"Loading CLIP model {model_name} on GPU {gpu_id}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    
    # Get batch size per GPU (scale based on number of GPUs)
    base_batch_size = config['processing']['batch_size']
    batch_size = base_batch_size  # Keep same batch size per GPU for stability
    
    # Handle workers carefully - webdataset can have issues with multiprocessing
    base_workers = config['processing']['num_workers']
    if base_workers > 0:
        num_workers = max(1, base_workers // config['processing'].get('num_gpus', 4))
    else:
        num_workers = 0  # No workers if configured to 0
    
    # Process each assigned shard
    for shard_idx in shard_indices:
        tar_file = tar_files[shard_idx]
        logger.info(f"Processing shard {shard_idx}: {tar_file.name}")
        
        # Create webdataset with proper settings to avoid worker issues
        dataset = wds.WebDataset(str(tar_file), shardshuffle=False)
        dataset = dataset.decode("pil").to_tuple("__key__", "jpg", "json")
        dataset = dataset.batched(batch_size)
        
        # Create dataloader - use 0 workers to avoid webdataset issues
        # WebDataset already handles parallelism internally
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,  # Set to 0 to avoid worker issues
            pin_memory=True if device != "cpu" else False
        )
        
        # Process batches
        all_keys = []
        all_embeddings = []
        
        for batch in dataloader:
            keys, embeddings = process_batch_gpu(batch, model, preprocess, device)
            if len(keys) > 0:
                all_keys.extend(keys)
                all_embeddings.append(embeddings)
        
        if all_embeddings:
            # Concatenate embeddings
            shard_embeddings = np.vstack(all_embeddings)
            
            # Put results in queue
            output_queue.put({
                'shard_idx': shard_idx,
                'keys': all_keys,
                'embeddings': shard_embeddings
            })
            
            # Update progress
            progress_queue.put(1)
            logger.info(f"Completed shard {shard_idx}: {len(all_keys)} samples")
    
    logger.info(f"GPU {gpu_id} completed all assigned shards")


def save_embeddings_worker(save_queue: mp.Queue, output_path: Path, stop_event: threading.Event):
    """
    Worker thread to save embeddings as they're computed.
    
    Args:
        save_queue: Queue with embedding data to save
        output_path: Output directory path
        stop_event: Event to signal when to stop
    """
    while not stop_event.is_set() or not save_queue.empty():
        try:
            data = save_queue.get(timeout=1)
            if data is None:  # Poison pill
                break
            
            # Save embeddings
            shard_idx = data['shard_idx']
            output_file = output_path / f"embeddings_shard{shard_idx:04d}.npz"
            np.savez_compressed(
                output_file,
                keys=data['keys'],
                embeddings=data['embeddings']
            )
            print(f"Saved embeddings for shard {shard_idx} to {output_file}")
        except:
            continue


def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Setup paths
    dataset_path = Path(config['dataset']['input_path'])
    embeddings_path = Path(config['metadata']['embeddings_path'])
    embeddings_path.mkdir(parents=True, exist_ok=True)
    
    # Get tar files
    tar_files = sorted(dataset_path.glob("*.tar"))
    if args.max_shards:
        tar_files = tar_files[:args.max_shards]
    
    num_shards = len(tar_files)
    print(f"Found {num_shards} tar files to process")
    
    # Determine number of GPUs to use
    num_gpus = args.num_gpus if args.num_gpus else torch.cuda.device_count()
    num_gpus = min(num_gpus, num_shards)  # Don't use more GPUs than shards
    print(f"Using {num_gpus} GPUs for processing")
    
    # Update config with GPU count
    config['processing']['num_gpus'] = num_gpus
    
    # Split shards across GPUs
    shard_splits = get_shard_splits(num_shards, num_gpus)
    for gpu_id, shards in enumerate(shard_splits):
        print(f"GPU {gpu_id}: {len(shards)} shards")
    
    # Create multiprocessing queues
    output_queue = mp.Queue()
    progress_queue = mp.Queue()
    
    # Start save worker thread
    save_queue = output_queue
    stop_event = threading.Event()
    save_thread = threading.Thread(
        target=save_embeddings_worker,
        args=(save_queue, embeddings_path, stop_event)
    )
    save_thread.start()
    
    # Start GPU processes
    processes = []
    start_time = time.time()
    
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=process_shard_range,
            args=(
                gpu_id,
                shard_splits[gpu_id],
                tar_files,
                config,
                output_queue,
                progress_queue
            )
        )
        p.start()
        processes.append(p)
        print(f"Started process for GPU {gpu_id}")
    
    # Monitor progress
    total_processed = 0
    pbar = tqdm(total=num_shards, desc="Processing shards")
    
    while total_processed < num_shards:
        try:
            _ = progress_queue.get(timeout=1)
            total_processed += 1
            pbar.update(1)
        except:
            continue
    
    pbar.close()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Stop save worker
    stop_event.set()
    output_queue.put(None)  # Poison pill
    save_thread.join()
    
    # Calculate processing time
    total_time = time.time() - start_time
    print(f"\nProcessing completed in {total_time:.2f} seconds")
    print(f"Average time per shard: {total_time/num_shards:.2f} seconds")
    
    # Count total samples processed
    embedding_files = list(embeddings_path.glob("embeddings_shard*.npz"))
    total_samples = 0
    embedding_dim = None
    
    for emb_file in embedding_files:
        data = np.load(emb_file)
        total_samples += len(data['keys'])
        if embedding_dim is None:
            embedding_dim = data['embeddings'].shape[1]
    
    print(f"Total samples processed: {total_samples}")
    print(f"Throughput: {total_samples/total_time:.2f} samples/second")
    
    # Save metadata
    metadata = {
        'total_samples': total_samples,
        'num_shards': num_shards,
        'model_name': config['clip']['model_name'],
        'pretrained': config['clip']['pretrained'],
        'embedding_dim': embedding_dim,
        'num_gpus_used': num_gpus,
        'processing_time_seconds': total_time,
        'throughput_samples_per_second': total_samples/total_time
    }
    
    metadata_file = embeddings_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_file}")
    print(f"Embeddings saved to: {embeddings_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU CLIP embedding extraction")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Maximum number of shards to process (for testing)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)"
    )
    
    args = parser.parse_args()
    main(args)