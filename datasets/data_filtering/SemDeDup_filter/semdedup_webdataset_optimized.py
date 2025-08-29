"""
Optimized SemDeDup Pipeline for WebDataset Format
High-performance multi-GPU pipeline with progress reporting and efficiency improvements
"""

import os
import sys
import json
import time
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
import io
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import webdataset as wds
from PIL import Image
import pandas as pd
import faiss
from tqdm import tqdm
import yaml
import argparse
from torch.nn.functional import normalize
from torch.cuda.amp import autocast

# Add SemDeDup modules to path
sys.path.append('/iopsstor/scratch/cscs/rqu/benchmark-image-tokenzier/datasets/data_filtering/SemDeDup_filter/SemDeDup')
from clustering.clustering import compute_centroids
from clustering.sort_clusters import assign_and_sort_clusters
from extract_dedup_data import extract_pruned_data
from semdedup import SemDeDupJob

# Setup logging with color support
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""
    
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',   # Green  
        'WARNING': '\033[33m', # Yellow
        'ERROR': '\033[31m',   # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'     # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
))
logger.addHandler(handler)


@dataclass
class OptimizedSemDeDupConfig:
    """Configuration for optimized SemDeDup pipeline"""
    # Dataset paths
    dataset_path: str = "/capstor/store/cscs/swissai/infra01/vision-datasets/coco2017/webdataset_permissive"
    output_path: str = "./semdedup_output"
    # output_path: str = "/capstor/store/cscs/swissai/infra01/vision-datasets/coco2017/semdedup_filtered"
    
    # Data loading control
    max_shards: Optional[int] = None  # Limit number of shards to process
    max_samples: Optional[int] = None  # Limit total samples
    
    # Model configuration
    clip_model: str = "ViT-L/14"
    batch_size: int = 256  # Increased for better GPU utilization
    num_workers: int = 16  # More workers for faster loading
    prefetch_factor: int = 4  # Prefetch batches
    
    # Clustering parameters
    num_clusters: int = 100
    kmeans_iterations: int = 50
    use_cosine_distance: bool = True
    
    # SemDeDup parameters  
    epsilon_values: List[float] = None
    which_to_keep: str = "hard"
    
    # GPU configuration
    use_multi_gpu: bool = True
    devices: List[int] = None
    use_mixed_precision: bool = True  # Use FP16 for speed
    
    # Testing/debugging options
    seed: int = 42
    verbose: bool = True
    
    # Processing configuration
    embedding_dim: int = 768
    dataset_size: int = 30371
    num_shards: int = 31
    save_intermediate: bool = True
    
    # Progress reporting
    report_interval: int = 10  # Report every N batches
    
    # Paths for intermediate results
    embeddings_path: str = None
    clusters_path: str = None
    semdedup_results_path: str = None
    
    def __post_init__(self):
        """Set default values after initialization"""
        if self.epsilon_values is None:
            self.epsilon_values = [0.001, 0.005, 0.01, 0.02, 0.05]
        
        if self.devices is None and torch.cuda.is_available():
            self.devices = list(range(torch.cuda.device_count()))
            logger.info(f"🚀 Detected {len(self.devices)} GPUs: {self.devices}")
        
        # Set intermediate paths
        output_base = Path(self.output_path)
        self.embeddings_path = str(output_base / "embeddings")
        self.clusters_path = str(output_base / "clusters")
        self.semdedup_results_path = str(output_base / "semdedup_results")


class OptimizedCLIPEmbedder:
    """Optimized CLIP model wrapper with mixed precision and efficient batching"""
    
    def __init__(self, model_name: str = "ViT-L/14", device: str = "cuda", use_fp16: bool = True):
        """Initialize CLIP model with optimizations"""
        import clip
        self.device = device
        self.use_fp16 = use_fp16
        
        logger.info(f"📊 Loading CLIP model {model_name} on {device}")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        
        # Convert to half precision if requested
        if use_fp16 and device != "cpu":
            self.model = self.model.half()
            logger.info("✅ Using FP16 mixed precision for faster inference")
    
    @torch.no_grad()
    def encode_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Encode a batch of images with optimizations"""
        # Preprocess images
        processed = torch.stack([self.preprocess(img) for img in images])
        processed = processed.to(self.device)
        
        # Use mixed precision if enabled
        if self.use_fp16 and self.device != "cpu":
            processed = processed.half()
        
        # Get embeddings
        with autocast(enabled=self.use_fp16 and self.device != "cpu"):
            embeddings = self.model.encode_image(processed)
        
        # Normalize and convert to numpy
        embeddings = normalize(embeddings.float(), dim=1)
        return embeddings.cpu().numpy()


def get_limited_tar_files(dataset_path: str, max_shards: Optional[int] = None) -> List[str]:
    """Get list of tar files with optional limit"""
    tar_files = sorted(Path(dataset_path).glob("*.tar"))
    total_shards = len(tar_files)
    
    if max_shards and max_shards < total_shards:
        tar_files = tar_files[:max_shards]
        logger.info(f"📁 Limited to first {max_shards} shards out of {total_shards} total")
    
    return [str(f) for f in tar_files]


def extract_embeddings_single_gpu_optimized(
    config: OptimizedSemDeDupConfig,
    device_id: int = 0,
    shard_indices: Optional[List[int]] = None,
    progress_position: int = 0
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """Optimized embedding extraction on a single GPU"""
    
    start_time = time.time()
    device = f"cuda:{device_id}"
    torch.cuda.set_device(device_id)
    
    # Initialize CLIP model with optimizations
    embedder = OptimizedCLIPEmbedder(config.clip_model, device, config.use_mixed_precision)
    
    # Select shards for this GPU
    all_tar_files = get_limited_tar_files(config.dataset_path, config.max_shards)
    
    if shard_indices is None:
        shard_indices = list(range(len(all_tar_files)))
    
    selected_files = [all_tar_files[i] for i in shard_indices if i < len(all_tar_files)]
    urls = [str(f) for f in selected_files]
    
    logger.info(f"🎯 GPU {device_id}: Processing {len(urls)} shards")
    
    # If no shards assigned, return empty results
    if not urls:
        logger.info(f"⚠️ GPU {device_id}: No shards assigned, skipping")
        return np.array([]), [], {
            'gpu_id': device_id,
            'samples_processed': 0,
            'elapsed_time': 0,
            'samples_per_second': 0,
            'peak_gpu_memory_gb': 0
        }
    
    # Create dataset with proper buffering
    dataset = (
        wds.WebDataset(urls, shardshuffle=False)
        .decode("pil")
        .to_tuple("jpg", "json", "__key__")
    )
    
    # Extract embeddings with progress tracking
    all_embeddings = []
    all_keys = []
    
    batch_images = []
    batch_keys = []
    samples_processed = 0
    batches_processed = 0
    last_report_time = time.time()
    
    # Create progress bar
    pbar = tqdm(
        total=config.max_samples if config.max_samples else None,
        desc=f"GPU {device_id}",
        position=progress_position,
        leave=True,
        unit="samples"
    )
    
    try:
        for sample in dataset:
            # Check if we've reached max_samples BEFORE processing
            if config.max_samples and samples_processed >= config.max_samples:
                break
                
            img, json_data, key = sample
            
            batch_images.append(img)
            batch_keys.append(key)
            samples_processed += 1
            
            if len(batch_images) >= config.batch_size:
                # Process batch
                embeddings = embedder.encode_batch(batch_images)
                all_embeddings.append(embeddings)
                all_keys.extend(batch_keys)
                
                batches_processed += 1
                pbar.update(len(batch_images))
                
                # Report progress
                if batches_processed % config.report_interval == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    samples_per_sec = samples_processed / elapsed
                    gpu_memory = torch.cuda.max_memory_allocated(device_id) / 1024**3
                    
                    logger.info(
                        f"📈 GPU {device_id}: {samples_processed} samples | "
                        f"{samples_per_sec:.1f} samples/sec | "
                        f"GPU Memory: {gpu_memory:.1f} GB"
                    )
                
                batch_images = []
                batch_keys = []
        
        # Process remaining samples
        if batch_images:
            embeddings = embedder.encode_batch(batch_images)
            all_embeddings.append(embeddings)
            all_keys.extend(batch_keys)
            pbar.update(len(batch_images))
    
    finally:
        pbar.close()
    
    # Concatenate all embeddings
    if all_embeddings:
        all_embeddings = np.vstack(all_embeddings)
    else:
        all_embeddings = np.array([])
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    stats = {
        'gpu_id': device_id,
        'samples_processed': len(all_embeddings),
        'elapsed_time': elapsed_time,
        'samples_per_second': len(all_embeddings) / elapsed_time if elapsed_time > 0 else 0,
        'peak_gpu_memory_gb': torch.cuda.max_memory_allocated(device_id) / 1024**3
    }
    
    logger.info(
        f"✅ GPU {device_id}: Completed! Extracted {len(all_embeddings)} embeddings in {elapsed_time:.1f}s "
        f"({stats['samples_per_second']:.1f} samples/sec)"
    )
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    return all_embeddings, all_keys, stats


def extract_embeddings_multi_gpu_optimized(config: OptimizedSemDeDupConfig) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """Optimized multi-GPU embedding extraction"""
    
    import torch.multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    # Set multiprocessing start method for CUDA
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    num_gpus = len(config.devices)
    logger.info(f"🚀 Starting multi-GPU extraction with {num_gpus} GPUs")
    
    # Get limited tar files
    all_tar_files = get_limited_tar_files(config.dataset_path, config.max_shards)
    num_shards_to_process = len(all_tar_files)
    
    # Distribute shards across GPUs
    shards_per_gpu = num_shards_to_process // num_gpus
    remainder = num_shards_to_process % num_gpus
    
    shard_assignments = []
    start_idx = 0
    
    for gpu_id in range(num_gpus):
        num_shards = shards_per_gpu + (1 if gpu_id < remainder else 0)
        end_idx = start_idx + num_shards
        shard_assignments.append(list(range(start_idx, end_idx)))
        start_idx = end_idx
    
    logger.info(f"📊 Shard distribution: {[len(s) for s in shard_assignments]} shards per GPU")
    
    # Adjust max_samples per GPU if specified
    samples_per_gpu = None
    if config.max_samples:
        samples_per_gpu = config.max_samples // num_gpus
        remainder_samples = config.max_samples % num_gpus
        logger.info(f"📊 Sample limit: ~{samples_per_gpu} samples per GPU (total: {config.max_samples})")
    
    # Run extraction in parallel
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        
        for i, (gpu_id, shard_indices) in enumerate(zip(config.devices, shard_assignments)):
            # Adjust config for this GPU
            gpu_config = OptimizedSemDeDupConfig(**asdict(config))
            
            if samples_per_gpu:
                gpu_config.max_samples = samples_per_gpu + (1 if i < remainder_samples else 0)
            
            future = executor.submit(
                extract_embeddings_single_gpu_optimized,
                gpu_config,
                gpu_id,
                shard_indices,
                gpu_id  # progress position
            )
            futures.append(future)
        
        # Collect results
        all_embeddings = []
        all_keys = []
        all_stats = []
        
        for future in as_completed(futures):
            embeddings, keys, stats = future.result()
            if len(embeddings) > 0:  # Only add non-empty embeddings
                all_embeddings.append(embeddings)
            all_keys.extend(keys)
            all_stats.append(stats)
    
    # Concatenate all embeddings
    if all_embeddings:
        all_embeddings = np.vstack(all_embeddings)
    else:
        all_embeddings = np.array([])
    
    # Calculate aggregate statistics
    elapsed_time = time.time() - start_time
    total_samples = len(all_embeddings)
    aggregate_stats = {
        'total_samples': total_samples,
        'total_elapsed_time': elapsed_time,
        'aggregate_samples_per_second': total_samples / elapsed_time,
        'gpu_stats': all_stats
    }
    
    logger.info("="*60)
    logger.info(f"🎉 Multi-GPU extraction complete!")
    logger.info(f"📊 Total: {total_samples} embeddings in {elapsed_time:.1f}s")
    logger.info(f"⚡ Speed: {aggregate_stats['aggregate_samples_per_second']:.1f} samples/sec")
    logger.info("="*60)
    
    return all_embeddings, all_keys, aggregate_stats


def run_optimized_semdedup_pipeline(config: OptimizedSemDeDupConfig):
    """Run optimized SemDeDup pipeline with progress reporting"""
    
    logger.info("="*70)
    logger.info("🚀 Starting Optimized SemDeDup Pipeline for WebDataset")
    logger.info("="*70)
    
    # Report configuration
    logger.info(f"📋 Configuration:")
    logger.info(f"   - Dataset: {config.dataset_path}")
    logger.info(f"   - Max shards: {config.max_shards if config.max_shards else 'All'}")
    logger.info(f"   - Max samples: {config.max_samples if config.max_samples else 'All'}")
    logger.info(f"   - Batch size: {config.batch_size}")
    logger.info(f"   - GPUs: {config.devices}")
    logger.info(f"   - Mixed precision: {config.use_mixed_precision}")
    logger.info(f"   - Clusters: {config.num_clusters}")
    logger.info(f"   - Epsilon values: {config.epsilon_values}")
    
    # Create output directories
    os.makedirs(config.output_path, exist_ok=True)
    
    # Step 1: Extract embeddings
    logger.info("\n" + "="*60)
    logger.info("📝 Step 1/5: Extracting embeddings...")
    logger.info("="*60)
    
    step1_start = time.time()
    
    embeddings_exist = os.path.exists(os.path.join(config.embeddings_path, "metadata.json"))
    
    if not embeddings_exist or not config.save_intermediate:
        if config.use_multi_gpu and len(config.devices) > 1:
            embeddings, keys, stats = extract_embeddings_multi_gpu_optimized(config)
        else:
            device_id = config.devices[0] if config.devices else 0
            embeddings, keys, stats = extract_embeddings_single_gpu_optimized(config, device_id)
        
        if config.save_intermediate:
            save_embeddings(embeddings, keys, config.embeddings_path)
    else:
        logger.info("📂 Loading existing embeddings...")
        embeddings, keys = load_embeddings(config.embeddings_path)
    
    step1_time = time.time() - step1_start
    
    # Update dataset size
    config.dataset_size = len(embeddings)
    logger.info(f"✅ Step 1 complete! Dataset size: {config.dataset_size} samples ({step1_time:.1f}s)")
    
    # Convert keys to paths format
    paths = np.array(keys, dtype='S256')
    paths_memmap_path = os.path.join(config.embeddings_path, "paths.npy")
    paths_memmap = np.memmap(
        paths_memmap_path,
        dtype='S256',
        mode='w+',
        shape=(config.dataset_size,)
    )
    paths_memmap[:] = paths
    del paths_memmap
    
    # Step 2: K-means clustering
    logger.info("\n" + "="*60)
    logger.info("📝 Step 2/5: Running K-means clustering...")
    logger.info("="*60)
    
    step2_start = time.time()
    
    compute_centroids(
        data=embeddings,
        ncentroids=config.num_clusters,
        niter=config.kmeans_iterations,
        seed=42,
        Kmeans_with_cos_dist=config.use_cosine_distance,
        save_folder=config.clusters_path,
        logger=logger,
        verbose=config.verbose
    )
    
    step2_time = time.time() - step2_start
    logger.info(f"✅ Step 2 complete! Clustering done ({step2_time:.1f}s)")
    
    # Step 3: Sort clusters
    logger.info("\n" + "="*60)
    logger.info("📝 Step 3/5: Sorting clusters...")
    logger.info("="*60)
    
    step3_start = time.time()
    
    sorted_clusters_path = os.path.join(config.clusters_path, "sorted_clusters")
    
    # Reload paths memmap
    paths_memmap = np.memmap(
        paths_memmap_path,
        dtype='S256',
        mode='r',
        shape=(config.dataset_size,)
    )
    
    assign_and_sort_clusters(
        data=embeddings,
        paths_list=paths_memmap,
        sim_metric='cosine' if config.use_cosine_distance else 'l2',
        keep_hard=(config.which_to_keep == 'hard'),
        kmeans_with_cos_dist=config.use_cosine_distance,
        save_folder=config.clusters_path,
        sorted_clusters_file_loc=sorted_clusters_path,
        cluster_ids=range(0, config.num_clusters),
        logger=logger
    )
    
    step3_time = time.time() - step3_start
    logger.info(f"✅ Step 3 complete! Clusters sorted ({step3_time:.1f}s)")
    
    # Step 4: Run SemDeDup
    logger.info("\n" + "="*60)
    logger.info("📝 Step 4/5: Running SemDeDup deduplication...")
    logger.info("="*60)
    
    step4_start = time.time()
    
    # Create args for SemDeDup
    class Args:
        def __init__(self, config):
            self.embs_memory_loc = os.path.join(config.embeddings_path, "embeddings.npy")
            self.dataset_size = config.dataset_size
            self.emd_size = config.embedding_dim
            self.sorted_clusters_path = sorted_clusters_path
            self.save_loc = config.semdedup_results_path
            self.eps_list = config.epsilon_values
            self.which_to_keep = config.which_to_keep
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.seed = 42
            self.num_clusters = config.num_clusters
            self.clusters_per_job = config.num_clusters
    
    args = Args(config)
    os.makedirs(config.semdedup_results_path, exist_ok=True)
    os.makedirs(os.path.join(config.semdedup_results_path, 'dataframes'), exist_ok=True)
    
    # Run SemDeDup
    job = SemDeDupJob(args, job_start_cluster=0)
    job._process_shard(0, config.num_clusters)
    
    step4_time = time.time() - step4_start
    logger.info(f"✅ Step 4 complete! Deduplication done ({step4_time:.1f}s)")
    
    # Step 5: Extract deduplicated data
    logger.info("\n" + "="*60)
    logger.info("📝 Step 5/5: Extracting deduplicated data...")
    logger.info("="*60)
    
    step5_start = time.time()
    
    results = {}
    
    for eps in config.epsilon_values:
        output_file = os.path.join(config.output_path, f'kept_samples_eps_{eps}.txt')
        
        extract_pruned_data(
            sorted_clusters_path=sorted_clusters_path,
            semdedup_pruning_tables_path=os.path.join(config.semdedup_results_path, 'dataframes'),
            eps=eps,
            num_clusters=config.num_clusters,
            output_txt_path=output_file,
            retreive_kept_samples=True
        )
        
        # Count kept samples
        with open(output_file, 'r') as f:
            kept_keys = [line.strip() for line in f.readlines()]
        
        kept_ratio = len(kept_keys) / config.dataset_size
        removed_count = config.dataset_size - len(kept_keys)
        
        results[eps] = {
            'kept_count': len(kept_keys),
            'removed_count': removed_count,
            'kept_ratio': kept_ratio,
            'removed_ratio': 1 - kept_ratio,
            'output_file': output_file
        }
        
        logger.info(
            f"📊 Epsilon {eps}: Kept {len(kept_keys)}/{config.dataset_size} "
            f"({100*kept_ratio:.1f}%), Removed {removed_count} ({100*(1-kept_ratio):.1f}%)"
        )
        
        # Create filtering metadata
        create_filter_metadata(kept_keys, keys, eps, config.output_path)
    
    step5_time = time.time() - step5_start
    logger.info(f"✅ Step 5 complete! Results extracted ({step5_time:.1f}s)")
    
    # Save summary results
    summary_path = os.path.join(config.output_path, "semdedup_summary.json")
    
    total_time = step1_time + step2_time + step3_time + step4_time + step5_time
    
    summary = {
        'configuration': asdict(config),
        'results': results,
        'timing': {
            'step1_embedding_extraction': step1_time,
            'step2_clustering': step2_time,
            'step3_sorting': step3_time,
            'step4_deduplication': step4_time,
            'step5_extraction': step5_time,
            'total_time': total_time
        },
        'performance': {
            'samples_processed': config.dataset_size,
            'overall_samples_per_second': config.dataset_size / total_time if total_time > 0 else 0
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Final report
    logger.info("\n" + "="*70)
    logger.info("🎉 SemDeDup Pipeline Complete!")
    logger.info("="*70)
    logger.info(f"📊 Performance Summary:")
    logger.info(f"   - Total samples: {config.dataset_size}")
    logger.info(f"   - Total time: {total_time:.1f}s ({str(timedelta(seconds=int(total_time)))})")
    logger.info(f"   - Overall speed: {summary['performance']['overall_samples_per_second']:.1f} samples/sec")
    logger.info(f"\n📊 Step Timing:")
    logger.info(f"   - Embedding extraction: {step1_time:.1f}s")
    logger.info(f"   - K-means clustering: {step2_time:.1f}s")
    logger.info(f"   - Cluster sorting: {step3_time:.1f}s")
    logger.info(f"   - Deduplication: {step4_time:.1f}s")
    logger.info(f"   - Result extraction: {step5_time:.1f}s")
    logger.info(f"\n📁 Results saved to: {config.output_path}")
    logger.info("="*70)


def save_embeddings(embeddings: np.ndarray, keys: List[str], output_path: str):
    """Save embeddings and keys to disk"""
    
    os.makedirs(output_path, exist_ok=True)
    
    # Save embeddings as memmap
    emb_path = os.path.join(output_path, "embeddings.npy")
    emb_memmap = np.memmap(
        emb_path,
        dtype='float32',
        mode='w+',
        shape=embeddings.shape
    )
    emb_memmap[:] = embeddings
    del emb_memmap
    
    # Save keys
    keys_path = os.path.join(output_path, "keys.pkl")
    with open(keys_path, 'wb') as f:
        pickle.dump(keys, f)
    
    # Save metadata
    metadata = {
        'num_samples': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'keys_file': 'keys.pkl',
        'embeddings_file': 'embeddings.npy'
    }
    
    metadata_path = os.path.join(output_path, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"💾 Saved embeddings to {output_path}")


def load_embeddings(input_path: str) -> Tuple[np.ndarray, List[str]]:
    """Load embeddings and keys from disk"""
    
    # Load metadata
    metadata_path = os.path.join(input_path, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load embeddings
    emb_path = os.path.join(input_path, metadata['embeddings_file'])
    embeddings = np.memmap(
        emb_path,
        dtype='float32',
        mode='r',
        shape=(metadata['num_samples'], metadata['embedding_dim'])
    )
    
    # Load keys
    keys_path = os.path.join(input_path, metadata['keys_file'])
    with open(keys_path, 'rb') as f:
        keys = pickle.load(f)
    
    logger.info(f"📂 Loaded {len(embeddings)} embeddings from {input_path}")
    
    return embeddings, keys


def create_filter_metadata(kept_keys: List[str], all_keys: List[str], epsilon: float, output_path: str):
    """Create filtering metadata for easy webdataset filtering"""
    
    # Create a set for fast lookup
    kept_set = set(kept_keys)
    
    # Create filtering DataFrame
    filter_data = []
    for key in all_keys:
        filter_data.append({
            'key': key,
            'keep': key in kept_set,
            'epsilon': epsilon
        })
    
    df = pd.DataFrame(filter_data)
    
    # Save as parquet for efficient loading
    filter_path = os.path.join(output_path, f'filter_eps_{epsilon}.parquet')
    df.to_parquet(filter_path, index=False)
    
    # Also save as JSON for easy inspection
    filter_json_path = os.path.join(output_path, f'filter_eps_{epsilon}.json')
    filter_metadata = {
        'epsilon': epsilon,
        'total_samples': len(all_keys),
        'kept_samples': len(kept_keys),
        'removed_samples': len(all_keys) - len(kept_keys),
        'kept_ratio': len(kept_keys) / len(all_keys),
        'kept_keys': kept_keys
    }
    
    with open(filter_json_path, 'w') as f:
        json.dump(filter_metadata, f, indent=2)
    
    logger.info(f"💾 Created filter metadata for epsilon {epsilon}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Optimized SemDeDup for WebDataset')
    
    # Dataset parameters
    parser.add_argument('--dataset-path', type=str, 
                       default='/capstor/store/cscs/swissai/infra01/vision-datasets/coco2017/webdataset_permissive',
                       help='Path to WebDataset directory')
    parser.add_argument('--output-path', type=str, default='./semdedup_output',
                       help='Output directory')
    parser.add_argument('--max-shards', type=int, default=None,
                       help='Maximum number of shards to process')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process')
    
    # Model parameters
    parser.add_argument('--clip-model', type=str, default='ViT-L/14',
                       help='CLIP model variant')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for embedding extraction')
    parser.add_argument('--num-workers', type=int, default=16,
                       help='Number of data loading workers')
    
    # Clustering parameters
    parser.add_argument('--num-clusters', type=int, default=100,
                       help='Number of clusters for K-means')
    parser.add_argument('--epsilon', type=float, nargs='+',
                       default=[0.001, 0.005, 0.01, 0.02, 0.05],
                       help='Epsilon values for deduplication')
    
    # GPU parameters
    parser.add_argument('--no-multi-gpu', action='store_true',
                       help='Disable multi-GPU processing')
    parser.add_argument('--no-mixed-precision', action='store_true',
                       help='Disable FP16 mixed precision')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Create configuration
    config = OptimizedSemDeDupConfig(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        max_shards=args.max_shards,
        max_samples=args.max_samples,
        clip_model=args.clip_model,
        batch_size=args.batch_size,
        num_clusters=args.num_clusters,
        epsilon_values=args.epsilon,
        num_workers=args.num_workers,
        use_multi_gpu=not args.no_multi_gpu,
        use_mixed_precision=not args.no_mixed_precision,
        seed=args.seed,
        verbose=args.verbose
    )
    
    # Run pipeline
    run_optimized_semdedup_pipeline(config)


if __name__ == "__main__":
    main()