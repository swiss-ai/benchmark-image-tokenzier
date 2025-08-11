#!/usr/bin/env python3
"""
Multi-GPU parallel aesthetic score computation using pre-computed CLIP embeddings.
Optimized for systems with multiple GPUs and high CPU count.
"""

import os
import json
import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse
from typing import Dict, List, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
import time
from utils.aesthetic_model import get_aesthetic_model

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


def get_embedding_splits(embedding_files: List[Path], num_gpus: int) -> List[List[Path]]:
    """
    Split embedding files across GPUs evenly.
    
    Args:
        embedding_files: List of embedding files
        num_gpus: Number of GPUs to use
    
    Returns:
        List of file paths for each GPU
    """
    files_per_gpu = len(embedding_files) // num_gpus
    remainder = len(embedding_files) % num_gpus
    
    splits = []
    start = 0
    for i in range(num_gpus):
        count = files_per_gpu + (1 if i < remainder else 0)
        splits.append(embedding_files[start:start + count])
        start += count
    
    return splits


def compute_aesthetic_scores_batch(
    embeddings: np.ndarray,
    aesthetic_model: torch.nn.Module,
    device: str = "cuda"
) -> np.ndarray:
    """
    Compute aesthetic scores for a batch of embeddings.
    
    Args:
        embeddings: Numpy array of CLIP embeddings
        aesthetic_model: LAION aesthetic predictor model
        device: Device to run on
    
    Returns:
        Array of aesthetic scores
    """
    # Convert to torch tensor
    embeddings_tensor = torch.from_numpy(embeddings).float().to(device)
    
    # Compute scores
    with torch.no_grad():
        scores = aesthetic_model(embeddings_tensor)
        scores = scores.squeeze().cpu().numpy()
    
    # Handle single sample case
    if scores.ndim == 0:
        scores = np.array([scores])
    
    return scores


def process_embedding_files_gpu(
    gpu_id: int,
    embedding_files: List[Path],
    clip_model_type: str,
    output_queue: mp.Queue,
    progress_queue: mp.Queue,
    batch_size: int = 1024
):
    """
    Process embedding files on a specific GPU.
    
    Args:
        gpu_id: GPU device ID
        embedding_files: List of embedding files to process
        clip_model_type: Type of CLIP model (vit_l_14 or vit_b_32)
        output_queue: Queue for output results
        progress_queue: Queue for progress updates
        batch_size: Batch size for score computation
    """
    # Set GPU
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    logger = setup_logger(gpu_id)
    
    logger.info(f"Processing {len(embedding_files)} embedding files")
    
    # Load aesthetic model
    logger.info(f"Loading aesthetic model for {clip_model_type} on GPU {gpu_id}")
    aesthetic_model = get_aesthetic_model(clip_model_type, device)
    
    # Process each file
    for emb_file in embedding_files:
        # Extract shard index from filename (handle both formats)
        filename = emb_file.name
        
        # Parse filename to get shard index
        # Format 1: embeddings_shard0000.npz
        # Format 2: embeddings_shard000000_batch000000.npz
        try:
            shard_part = filename.split('shard')[1].split('.')[0]
            if '_batch' in shard_part:
                shard_idx = int(shard_part.split('_batch')[0])
            else:
                shard_idx = int(shard_part)
        except (IndexError, ValueError) as e:
            logger.warning(f"Skipping file with unexpected format: {filename}")
            continue
        
        logger.info(f"Processing {filename}")
        
        # Load embeddings
        data = np.load(emb_file)
        keys = data['keys']
        embeddings = data['embeddings']
        
        # Process in batches for memory efficiency
        num_samples = len(keys)
        all_scores = []
        
        for i in range(0, num_samples, batch_size):
            batch_embeddings = embeddings[i:i+batch_size]
            batch_scores = compute_aesthetic_scores_batch(
                batch_embeddings, aesthetic_model, device
            )
            all_scores.extend(batch_scores)
        
        # Create records
        records = []
        for key, score in zip(keys, all_scores):
            record = {
                'sample_id': key,
                'shard_idx': shard_idx,
                'aesthetic_score': float(score),
                'embedding_file': filename
            }
            records.append(record)
        
        # Put results in queue
        output_queue.put({
            'shard_idx': shard_idx,
            'records': records
        })
        
        # Update progress
        progress_queue.put(1)
        logger.info(f"Completed {filename}: {len(records)} scores computed")
    
    logger.info(f"GPU {gpu_id} completed all assigned files")


def save_scores_worker(
    save_queue: mp.Queue,
    output_path: Path,
    partition_by_shard: bool,
    stop_event: mp.Event
):
    """
    Worker process to save scores as they're computed.
    
    Args:
        save_queue: Queue with score data to save
        output_path: Output directory path
        partition_by_shard: Whether to partition by shard
        stop_event: Event to signal when to stop
    """
    all_records = []
    
    while not stop_event.is_set() or not save_queue.empty():
        try:
            data = save_queue.get(timeout=1)
            if data is None:  # Poison pill
                break
            
            all_records.extend(data['records'])
            
            # Save periodically or when we have enough data
            if len(all_records) >= 100000:  # Save every 100k records
                df_batch = pd.DataFrame(all_records)
                
                if partition_by_shard:
                    # Save partitioned
                    aesthetic_dir = output_path / "aesthetic_scores"
                    aesthetic_dir.mkdir(exist_ok=True)
                    
                    table = pa.Table.from_pandas(df_batch)
                    pq.write_to_dataset(
                        table,
                        root_path=str(aesthetic_dir),
                        partition_cols=['shard_idx']
                    )
                else:
                    # Append to single file
                    output_file = output_path / "aesthetic_scores.parquet"
                    if output_file.exists():
                        existing_df = pd.read_parquet(output_file)
                        df_batch = pd.concat([existing_df, df_batch], ignore_index=True)
                    df_batch.to_parquet(output_file, engine='pyarrow', compression='snappy')
                
                all_records = []
                print(f"Saved batch of scores to {output_path}")
        except:
            continue
    
    # Save remaining records
    if all_records:
        df_final = pd.DataFrame(all_records)
        
        if partition_by_shard:
            aesthetic_dir = output_path / "aesthetic_scores"
            aesthetic_dir.mkdir(exist_ok=True)
            
            table = pa.Table.from_pandas(df_final)
            pq.write_to_dataset(
                table,
                root_path=str(aesthetic_dir),
                partition_cols=['shard_idx']
            )
        else:
            output_file = output_path / "aesthetic_scores.parquet"
            if output_file.exists():
                existing_df = pd.read_parquet(output_file)
                df_final = pd.concat([existing_df, df_final], ignore_index=True)
            df_final.to_parquet(output_file, engine='pyarrow', compression='snappy')
        
        print(f"Saved final batch of scores to {output_path}")


def generate_statistics(scores_path: Path, partition_by_shard: bool) -> Dict:
    """Generate statistics about the aesthetic scores."""
    # Load scores
    if partition_by_shard:
        aesthetic_dir = scores_path / "aesthetic_scores"
        df = pd.read_parquet(aesthetic_dir)
    else:
        output_file = scores_path / "aesthetic_scores.parquet"
        df = pd.read_parquet(output_file)
    
    stats = {
        'total_samples': len(df),
        'mean_score': float(df['aesthetic_score'].mean()),
        'std_score': float(df['aesthetic_score'].std()),
        'min_score': float(df['aesthetic_score'].min()),
        'max_score': float(df['aesthetic_score'].max()),
        'percentiles': {
            '10': float(df['aesthetic_score'].quantile(0.1)),
            '25': float(df['aesthetic_score'].quantile(0.25)),
            '50': float(df['aesthetic_score'].quantile(0.5)),
            '75': float(df['aesthetic_score'].quantile(0.75)),
            '90': float(df['aesthetic_score'].quantile(0.9)),
            '95': float(df['aesthetic_score'].quantile(0.95)),
            '99': float(df['aesthetic_score'].quantile(0.99))
        }
    }
    
    # Create sample sets
    top_samples = df.nlargest(100, 'aesthetic_score')
    top_samples.to_csv(scores_path / "top_100_aesthetic.csv", index=False)
    
    bottom_samples = df.nsmallest(100, 'aesthetic_score')
    bottom_samples.to_csv(scores_path / "bottom_100_aesthetic.csv", index=False)
    
    return stats


def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Setup paths
    embeddings_path = Path(config['metadata']['embeddings_path'])
    scores_path = Path(config['metadata']['scores_path'])
    scores_path.mkdir(parents=True, exist_ok=True)
    
    # Check if embeddings exist
    if not embeddings_path.exists():
        raise ValueError(f"Embeddings directory not found: {embeddings_path}")
    
    # Load embeddings metadata
    metadata_file = embeddings_path / "metadata.json"
    embeddings_metadata = None
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            embeddings_metadata = json.load(f)
        print(f"Embeddings info: {embeddings_metadata}")
    
    # Determine correct aesthetic model
    embedding_dim = embeddings_metadata.get('embedding_dim', 768) if embeddings_metadata else 768
    if embedding_dim == 768:
        clip_model_type = "vit_l_14"
    elif embedding_dim == 512:
        clip_model_type = "vit_b_32"
    else:
        clip_model_type = "vit_l_14" if "l_14" in config['clip']['model_name'].lower() else "vit_b_32"
    
    print(f"Using aesthetic model for {clip_model_type} (embedding_dim={embedding_dim})")
    
    # Get all embedding files
    embedding_files = sorted(embeddings_path.glob("embeddings_shard*.npz"))
    print(f"Found {len(embedding_files)} embedding files to process")
    
    # Determine number of GPUs to use
    num_gpus = args.num_gpus if args.num_gpus else torch.cuda.device_count()
    num_gpus = min(num_gpus, len(embedding_files))  # Don't use more GPUs than files
    print(f"Using {num_gpus} GPUs for processing")
    
    # Split files across GPUs
    file_splits = get_embedding_splits(embedding_files, num_gpus)
    for gpu_id, files in enumerate(file_splits):
        print(f"GPU {gpu_id}: {len(files)} files")
    
    # Create multiprocessing queues and events
    output_queue = mp.Queue()
    progress_queue = mp.Queue()
    stop_event = mp.Event()
    
    # Start save worker process
    save_process = mp.Process(
        target=save_scores_worker,
        args=(output_queue, scores_path, args.partition_by_shard, stop_event)
    )
    save_process.start()
    
    # Start GPU processes
    processes = []
    start_time = time.time()
    
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=process_embedding_files_gpu,
            args=(
                gpu_id,
                file_splits[gpu_id],
                clip_model_type,
                output_queue,
                progress_queue,
                config['processing'].get('score_batch_size', 1024)
            )
        )
        p.start()
        processes.append(p)
        print(f"Started process for GPU {gpu_id}")
    
    # Monitor progress
    total_processed = 0
    pbar = tqdm(total=len(embedding_files), desc="Processing embeddings")
    
    while total_processed < len(embedding_files):
        try:
            _ = progress_queue.get(timeout=1)
            total_processed += 1
            pbar.update(1)
        except:
            continue
    
    pbar.close()
    
    # Wait for all GPU processes to complete
    for p in processes:
        p.join()
    
    # Stop save worker
    output_queue.put(None)  # Poison pill
    stop_event.set()
    save_process.join()
    
    # Calculate processing time
    total_time = time.time() - start_time
    print(f"\nProcessing completed in {total_time:.2f} seconds")
    
    # Generate and save statistics
    print("\nGenerating statistics...")
    stats = generate_statistics(scores_path, args.partition_by_shard)
    
    print("\nAesthetic Score Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Mean score: {stats['mean_score']:.3f}")
    print(f"  Std deviation: {stats['std_score']:.3f}")
    print(f"  Min score: {stats['min_score']:.3f}")
    print(f"  Max score: {stats['max_score']:.3f}")
    print("\nPercentiles:")
    for p, v in stats['percentiles'].items():
        print(f"  {p}th percentile: {v:.3f}")
    
    # Add processing info to stats
    stats['processing_time_seconds'] = total_time
    stats['num_gpus_used'] = num_gpus
    stats['throughput_samples_per_second'] = stats['total_samples'] / total_time
    
    # Save statistics
    stats_file = scores_path / "aesthetic_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_file}")
    print(f"Throughput: {stats['throughput_samples_per_second']:.2f} samples/second")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU aesthetic score computation")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--partition-by-shard",
        action="store_true",
        help="Partition parquet files by shard index"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)"
    )
    
    args = parser.parse_args()
    main(args)