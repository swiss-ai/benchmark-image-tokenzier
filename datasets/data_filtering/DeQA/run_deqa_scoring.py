#!/usr/bin/env python
"""
YAML-configured Multi-GPU DeQA-Score processing pipeline for webdataset format
"""

import os
import sys
import json
import time
import yaml
import torch
import torch.multiprocessing as mp
from queue import Empty
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import webdataset as wds
from PIL import Image
import io
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import traceback
from tqdm import tqdm
import logging
import warnings
import argparse
from datetime import datetime
warnings.filterwarnings('ignore')

# Add DeQA-Score to path
sys.path.append('/iopsstor/scratch/cscs/rqu/benchmark-image-tokenzier/datasets/data_filtering/DeQA/DeQA-Score')


@dataclass
class Config:
    """Configuration loaded from YAML"""
    # Model config
    model_name: str
    model_cache_dir: str
    device_map: str
    dtype: str
    
    # Dataset config
    input_dir: str
    output_dir: str
    shard_pattern: str
    max_shards: Optional[int]
    
    # Processing config
    num_gpus: int
    batch_size: int
    num_workers: int
    prefetch_factor: int
    save_interval: int
    
    # Output config
    output_format: str
    compression: str
    partitioning: str
    include_metadata: bool
    save_distributions: bool
    
    # Statistics config
    generate_stats: bool
    stats_file: str
    percentiles: List[int]
    quality_thresholds: Dict[str, float]
    
    # Logging config
    log_level: str
    log_format: str
    save_logs: bool
    log_dir: str
    
    # Runtime config
    seed: int
    deterministic: bool
    debug_mode: bool
    checkpoint_interval: int
    resume_from_checkpoint: bool
    checkpoint_dir: str
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        return cls(
            # Model
            model_name=cfg['model']['name'],
            model_cache_dir=cfg['model']['cache_dir'],
            device_map=cfg['model']['device_map'],
            dtype=cfg['model']['dtype'],
            
            # Dataset
            input_dir=cfg['dataset']['input_dir'],
            output_dir=cfg['dataset']['output_dir'],
            shard_pattern=cfg['dataset']['shard_pattern'],
            max_shards=cfg['dataset']['max_shards'],
            
            # Processing
            num_gpus=cfg['processing']['num_gpus'],
            batch_size=cfg['processing']['batch_size'],
            num_workers=cfg['processing']['num_workers'],
            prefetch_factor=cfg['processing']['prefetch_factor'],
            save_interval=cfg['processing']['save_interval'],
            
            # Output
            output_format=cfg['output']['format'],
            compression=cfg['output']['compression'],
            partitioning=cfg['output']['partitioning'],
            include_metadata=cfg['output']['include_metadata'],
            save_distributions=cfg['output']['save_distributions'],
            
            # Statistics
            generate_stats=cfg['statistics']['generate'],
            stats_file=cfg['statistics']['output_file'],
            percentiles=cfg['statistics']['percentiles'],
            quality_thresholds=cfg['statistics']['quality_thresholds'],
            
            # Logging
            log_level=cfg['logging']['level'],
            log_format=cfg['logging']['format'],
            save_logs=cfg['logging']['save_to_file'],
            log_dir=cfg['logging']['log_dir'],
            
            # Runtime
            seed=cfg['runtime']['seed'],
            deterministic=cfg['runtime']['deterministic'],
            debug_mode=cfg['runtime']['debug_mode'],
            checkpoint_interval=cfg['runtime']['checkpoint_interval'],
            resume_from_checkpoint=cfg['runtime']['resume_from_checkpoint'],
            checkpoint_dir=cfg['runtime']['checkpoint_dir']
        )


@dataclass
class SampleScore:
    """Data structure for a scored sample"""
    sample_id: str
    shard_idx: int
    deqa_score: float
    score_distribution: Dict[str, float]
    image_width: int
    image_height: int
    image_mode: str
    has_conversations: bool
    num_conversations: int
    processing_time: float
    timestamp: str


def setup_logging(config: Config, name: str = "Main") -> logging.Logger:
    """Setup logging based on configuration"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.log_level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(config.log_format))
    logger.addHandler(console_handler)
    
    # File handler if requested
    if config.save_logs:
        os.makedirs(config.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            os.path.join(config.log_dir, f"deqa_scoring_{timestamp}.log")
        )
        file_handler.setFormatter(logging.Formatter(config.log_format))
        logger.addHandler(file_handler)
    
    return logger


def initialize_scorer_on_gpu(gpu_id: int, config: Config):
    """Initialize scorer on a specific GPU"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['HF_HOME'] = config.model_cache_dir
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(config.model_cache_dir, 'hub')
    torch.cuda.set_device(0)
    
    from src.model.builder import load_pretrained_model
    from src.constants import IMAGE_TOKEN_INDEX
    from src.mm_utils import tokenizer_image_token
    
    # Load model - use 'deqa' as model_name to trigger correct loading path
    model_name = 'deqa'  # This triggers the correct loading logic in builder.py
    tokenizer, model, image_processor, _ = load_pretrained_model(
        config.model_name,  # The actual model path/name from HuggingFace
        None,
        model_name,  # Use 'deqa' to trigger correct loading logic
        device_map="auto" if config.device_map == "auto" else {"": "cuda:0"},
        device="cuda:0"
    )
    
    # Setup scorer components - ensure model is in eval mode
    model.eval()
    prompt = "USER: How would you rate the quality of this image?\n<|image|>\nASSISTANT: The quality of the image is"
    preferential_ids = [id_[1] for id_ in tokenizer(["excellent","good","fair","poor","bad"])["input_ids"]]
    
    # Get the actual device the model is on
    model_device = next(model.parameters()).device
    weight_tensor = torch.Tensor([5.,4.,3.,2.,1.]).half().to(model_device)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model_device)
    
    return {
        'tokenizer': tokenizer,
        'model': model,
        'image_processor': image_processor,
        'preferential_ids': preferential_ids,
        'weight_tensor': weight_tensor,
        'input_ids': input_ids,
        'device': model_device
    }


def expand2square(pil_img, background_color):
    """Expand image to square"""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_batch_on_gpu(batch: List[Dict], scorer_components: Dict, config: Config) -> List[SampleScore]:
    """Process a batch of samples"""
    results = []
    
    model = scorer_components['model']
    image_processor = scorer_components['image_processor']
    preferential_ids = scorer_components['preferential_ids']
    weight_tensor = scorer_components['weight_tensor']
    input_ids = scorer_components['input_ids']
    device = scorer_components['device']
    
    # Ensure model is in eval mode
    model.eval()
    
    images = []
    metadata = []
    
    for sample in batch:
        try:
            img_bytes = sample.get('jpg') or sample.get('png')
            if img_bytes:
                img = Image.open(io.BytesIO(img_bytes))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
                
                meta = {
                    'sample_id': sample['__key__'],
                    'image_size': img.size,
                    'image_mode': img.mode
                }
                
                if 'json' in sample and config.include_metadata:
                    try:
                        json_data = json.loads(sample['json'])
                        meta['has_conversations'] = 'conversations' in json_data
                        meta['num_conversations'] = len(json_data.get('conversations', []))
                    except:
                        meta['has_conversations'] = False
                        meta['num_conversations'] = 0
                else:
                    meta['has_conversations'] = False
                    meta['num_conversations'] = 0
                
                metadata.append(meta)
        except Exception as e:
            if config.debug_mode:
                print(f"Error processing sample {sample.get('__key__', 'unknown')}: {e}")
    
    if not images:
        return results
    
    try:
        start_time = time.time()
        
        with torch.inference_mode():
            background_color = tuple(int(x*255) for x in image_processor.image_mean)
            expanded_images = [expand2square(img, background_color) for img in images]
            
            # Process images to tensors with proper dtype
            image_tensors = image_processor.preprocess(
                expanded_images,
                return_tensors="pt"
            )["pixel_values"]
            
            # Convert to half precision if model is in half precision
            if next(model.parameters()).dtype == torch.float16:
                image_tensors = image_tensors.half()
            
            image_tensors = image_tensors.to(device)
            
            output_logits = model(
                input_ids=input_ids.repeat(image_tensors.shape[0], 1),
                images=image_tensors
            )["logits"][:, -1, preferential_ids]
            
            distributions = torch.softmax(output_logits, -1)
            scores = distributions @ weight_tensor
        
        processing_time = time.time() - start_time
        
        for i, (meta, score, dist) in enumerate(zip(metadata, scores, distributions)):
            score_dist = {
                'excellent': float(dist[0].item()),
                'good': float(dist[1].item()),
                'fair': float(dist[2].item()),
                'poor': float(dist[3].item()),
                'bad': float(dist[4].item())
            } if config.save_distributions else {}
            
            result = SampleScore(
                sample_id=meta['sample_id'],
                shard_idx=-1,
                deqa_score=float(score.item()),
                score_distribution=score_dist,
                image_width=meta['image_size'][0],
                image_height=meta['image_size'][1],
                image_mode=meta['image_mode'],
                has_conversations=meta.get('has_conversations', False),
                num_conversations=meta.get('num_conversations', 0),
                processing_time=processing_time / len(metadata),
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            results.append(result)
    
    except Exception as e:
        if config.debug_mode:
            print(f"Error scoring batch: {e}")
            traceback.print_exc()
    
    return results


def gpu_worker_process(gpu_id: int, shard_queue: mp.Queue, result_queue: mp.Queue,
                       config: Config, worker_status: mp.Array):
    """GPU worker process"""
    try:
        logger = setup_logging(config, f'GPU_{gpu_id}')
        logger.info("Initializing model...")
        
        scorer_components = initialize_scorer_on_gpu(gpu_id, config)
        
        logger.info("Model initialized successfully")
        worker_status[gpu_id] = 1
        
        while True:
            try:
                shard_info = shard_queue.get(timeout=5)
                
                if shard_info is None:
                    break
                
                shard_path, shard_idx = shard_info
                worker_status[gpu_id] = 2
                
                logger.info(f"Processing shard {shard_idx}: {os.path.basename(shard_path)}")
                
                results = []
                dataset = wds.WebDataset(shard_path, shardshuffle=False)
                
                batch = []
                total_samples = 0
                
                for sample in dataset:
                    batch.append(sample)
                    
                    if len(batch) >= config.batch_size:
                        batch_results = process_batch_on_gpu(batch, scorer_components, config)
                        for r in batch_results:
                            r.shard_idx = shard_idx
                        results.extend(batch_results)
                        total_samples += len(batch)
                        
                        if total_samples % 100 == 0:
                            logger.debug(f"Shard {shard_idx}: Processed {total_samples} samples")
                        
                        batch = []
                
                if batch:
                    batch_results = process_batch_on_gpu(batch, scorer_components, config)
                    for r in batch_results:
                        r.shard_idx = shard_idx
                    results.extend(batch_results)
                    total_samples += len(batch)
                
                logger.info(f"Completed shard {shard_idx}: {total_samples} samples, {len(results)} scored")
                
                result_queue.put((shard_idx, results))
                worker_status[gpu_id] = 1
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing shard: {e}")
                if config.debug_mode:
                    traceback.print_exc()
        
        worker_status[gpu_id] = 0
        logger.info("Worker finished")
        
    except Exception as e:
        print(f"GPU {gpu_id} worker failed: {e}")
        traceback.print_exc()
        worker_status[gpu_id] = -1


def result_writer_process(result_queue: mp.Queue, config: Config, total_shards: int):
    """Writer process for saving results"""
    os.makedirs(config.output_dir, exist_ok=True)
    logger = setup_logging(config, 'ResultWriter')
    
    processed_shards = 0
    all_scores = []
    checkpoint_data = {}
    
    # Load checkpoint if resuming
    checkpoint_file = os.path.join(config.checkpoint_dir, 'writer_checkpoint.json')
    if config.resume_from_checkpoint and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        processed_shards = checkpoint_data.get('processed_shards', 0)
        all_scores = checkpoint_data.get('all_scores', [])
        logger.info(f"Resumed from checkpoint: {processed_shards} shards already processed")
    
    while processed_shards < total_shards:
        try:
            shard_idx, results = result_queue.get(timeout=60)
            
            if results:
                records = []
                for r in results:
                    record = {
                        'sample_id': r.sample_id,
                        'shard_idx': r.shard_idx,
                        'deqa_score': r.deqa_score,
                        'image_width': r.image_width,
                        'image_height': r.image_height,
                        'image_mode': r.image_mode,
                        'processing_time': r.processing_time,
                        'timestamp': r.timestamp
                    }
                    
                    if config.save_distributions and r.score_distribution:
                        record.update({
                            'score_excellent': r.score_distribution.get('excellent', 0),
                            'score_good': r.score_distribution.get('good', 0),
                            'score_fair': r.score_distribution.get('fair', 0),
                            'score_poor': r.score_distribution.get('poor', 0),
                            'score_bad': r.score_distribution.get('bad', 0)
                        })
                    
                    if config.include_metadata:
                        record.update({
                            'has_conversations': r.has_conversations,
                            'num_conversations': r.num_conversations
                        })
                    
                    records.append(record)
                    all_scores.append(r.deqa_score)
                
                df = pd.DataFrame(records)
                
                # Save to partitioned Parquet
                shard_dir = os.path.join(config.output_dir, f'{config.partitioning}={shard_idx}')
                os.makedirs(shard_dir, exist_ok=True)
                
                parquet_path = os.path.join(shard_dir, f'scores_{shard_idx:06d}.{config.output_format}')
                df.to_parquet(parquet_path, compression=config.compression, index=False)
                
                logger.info(f"Saved {len(results)} scores for shard {shard_idx}")
            
            processed_shards += 1
            logger.info(f"Progress: {processed_shards}/{total_shards} shards completed")
            
            # Save checkpoint
            if processed_shards % config.checkpoint_interval == 0:
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                checkpoint_data = {
                    'processed_shards': processed_shards,
                    'all_scores': all_scores,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)
                logger.debug(f"Checkpoint saved at shard {processed_shards}")
            
        except Empty:
            logger.warning("No results received in 60 seconds")
        except Exception as e:
            logger.error(f"Error writing results: {e}")
            if config.debug_mode:
                traceback.print_exc()
    
    # Generate statistics
    if config.generate_stats and all_scores:
        generate_statistics(all_scores, config)
    
    logger.info("Result writer finished")


def generate_statistics(scores: List[float], config: Config):
    """Generate statistics JSON file"""
    scores_array = np.array(scores)
    
    # Calculate quality distribution based on thresholds
    quality_counts = {
        'excellent': int(np.sum(scores_array >= config.quality_thresholds['excellent'])),
        'good': int(np.sum((scores_array >= config.quality_thresholds['good']) & 
                          (scores_array < config.quality_thresholds['excellent']))),
        'fair': int(np.sum((scores_array >= config.quality_thresholds['fair']) & 
                          (scores_array < config.quality_thresholds['good']))),
        'poor': int(np.sum((scores_array >= config.quality_thresholds['poor']) & 
                          (scores_array < config.quality_thresholds['fair']))),
        'bad': int(np.sum(scores_array < config.quality_thresholds['poor']))
    }
    
    quality_percentages = {
        k: float(v / len(scores) * 100) for k, v in quality_counts.items()
    }
    
    statistics = {
        'total_samples': len(scores),
        'mean': float(np.mean(scores_array)),
        'std': float(np.std(scores_array)),
        'min': float(np.min(scores_array)),
        'max': float(np.max(scores_array)),
        'median': float(np.median(scores_array)),
        'percentiles': {
            str(p): float(np.percentile(scores_array, p)) 
            for p in config.percentiles
        },
        'quality_distribution': quality_counts,
        'quality_percentages': quality_percentages,
        'quality_thresholds': config.quality_thresholds,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    stats_path = os.path.join(config.output_dir, config.stats_file)
    with open(stats_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    logger = logging.getLogger('Statistics')
    logger.info(f"Statistics saved to {stats_path}")
    logger.info(f"Mean score: {statistics['mean']:.3f} ± {statistics['std']:.3f}")
    logger.info(f"Score range: [{statistics['min']:.3f}, {statistics['max']:.3f}]")
    logger.info("Quality distribution:")
    for quality, percentage in quality_percentages.items():
        logger.info(f"  {quality}: {percentage:.1f}%")


def main(config_path: str):
    """Main processing pipeline"""
    # Load configuration
    config = Config.from_yaml(config_path)
    
    # Set random seed if deterministic
    if config.deterministic:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
    
    # Setup logging
    logger = setup_logging(config, 'Main')
    
    logger.info(f"Loaded configuration from: {config_path}")
    logger.info(f"Input directory: {config.input_dir}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Using {config.num_gpus} GPUs with batch size {config.batch_size}")
    
    # Get list of shards
    shard_files = sorted(Path(config.input_dir).glob(config.shard_pattern))
    
    if config.max_shards:
        shard_files = shard_files[:config.max_shards]
    
    logger.info(f"Found {len(shard_files)} shards to process")
    
    # Setup multiprocessing
    mp.set_start_method('spawn', force=True)
    
    shard_queue = mp.Queue()
    result_queue = mp.Queue()
    worker_status = mp.Array('i', [0] * config.num_gpus)
    
    # Start GPU workers with staggered initialization to avoid model loading conflicts
    workers = []
    for gpu_id in range(config.num_gpus):
        p = mp.Process(
            target=gpu_worker_process,
            args=(gpu_id, shard_queue, result_queue, config, worker_status)
        )
        p.start()
        workers.append(p)
        logger.info(f"Started worker for GPU {gpu_id}")
        # Stagger worker starts to avoid simultaneous model loading
        if gpu_id < config.num_gpus - 1:
            time.sleep(5)
    
    # Wait for workers to initialize
    logger.info("Waiting for GPU workers to initialize...")
    timeout = 300  # Increased timeout to 5 minutes
    start_time = time.time()
    while time.time() - start_time < timeout:
        ready_count = sum(1 for s in worker_status if s == 1)
        failed_count = sum(1 for s in worker_status if s == -1)
        
        if failed_count > 0:
            logger.error(f"{failed_count} workers failed to initialize")
            break
        
        if ready_count == config.num_gpus:
            logger.info(f"All {config.num_gpus} GPU workers ready")
            break
        
        time.sleep(2)
    
    ready_workers = sum(1 for s in worker_status if s == 1)
    if ready_workers == 0:
        logger.error("No workers initialized successfully")
        for w in workers:
            w.terminate()
        return
    
    logger.info(f"{ready_workers}/{config.num_gpus} GPU workers ready")
    
    # Start result writer
    writer = mp.Process(
        target=result_writer_process,
        args=(result_queue, config, len(shard_files))
    )
    writer.start()
    
    # Queue all shards
    for idx, shard_path in enumerate(shard_files):
        shard_queue.put((str(shard_path), idx))
    
    # Add poison pills
    for _ in range(config.num_gpus):
        shard_queue.put(None)
    
    # Monitor progress
    logger.info("Processing started...")
    start_time = time.time()
    
    # Wait for workers to finish
    for w in workers:
        w.join()
    
    # Wait for writer to finish
    writer.join()
    
    # Report results
    elapsed_time = time.time() - start_time
    logger.info(f"Processing completed in {elapsed_time/60:.2f} minutes")
    logger.info(f"Results saved to {config.output_dir}")
    
    # Print final statistics if available
    stats_file = os.path.join(config.output_dir, config.stats_file)
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        logger.info("Final Statistics:")
        logger.info(f"  Total samples: {stats['total_samples']}")
        logger.info(f"  Mean score: {stats['mean']:.3f} ± {stats['std']:.3f}")
        logger.info(f"  Score range: [{stats['min']:.3f}, {stats['max']:.3f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeQA Score WebDataset Processing with YAML Config')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    main(args.config)