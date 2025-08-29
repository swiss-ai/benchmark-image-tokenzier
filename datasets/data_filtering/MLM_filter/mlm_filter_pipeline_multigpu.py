#!/usr/bin/env python3
"""
Fixed Multi-GPU MLM Filter Pipeline - Matches Original Implementation
Addresses all critical and moderate issues for consistent model outputs
"""

import os
import sys
import yaml
import json
import logging
import argparse
import time
import io
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
from multiprocessing import Pool, Queue
import multiprocessing as mp
import warnings
import traceback
from dataclasses import dataclass

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as tmp
import webdataset as wds
from PIL import Image

# Suppress expected warnings
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint.*were not used")
warnings.filterwarnings("ignore", message=".*This IS expected if you are initializing.*")

# Set up environment
def setup_environment():
    """Set up environment variables for CUDA"""
    if 'LD_LIBRARY_PATH' in os.environ:
        del os.environ['LD_LIBRARY_PATH']
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/compat/lib.real:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64'
    
    # Fix LLaVA imports
    llava_init = '/users/rqu/miniconda3/envs/mlm_filter/lib/python3.10/site-packages/llava/__init__.py'
    model_init = '/users/rqu/miniconda3/envs/mlm_filter/lib/python3.10/site-packages/llava/model/__init__.py'
    
    if os.path.exists(llava_init):
        with open(llava_init, 'w') as f:
            f.write('# LLaVA package\n')
    
    if os.path.exists(model_init):
        with open(model_init, 'w') as f:
            f.write('# LLaVA model package\n')

setup_environment()

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

# Use exact template from original
LLAVA_INSTRUCTION_TEMPLATE = """Text Caption: {caption}

{criteria} A higher score indicates higher level of {aspect}. Ensure that your scoring is nuanced and uses the entire range from 0 to 100, reflecting the subtle differences. The score should be given as an integer, with each number between 0 and 100 considered as a potential score, avoiding the tendency to round to multiples of 5 or 10. Please first output a single line containing the value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias."""

criteria_image_text_matching = "Please evaluate if the provided text caption accurately represents the main features and objects of the image. The caption doesn't need to detail every aspect of the image, but it should capture its primary theme. Rate the overall quality of the text caption's match to the image on a scale of 1-100, considering the criteria mentioned."

criteria_object_detail_fulfillment = "Please evaluate the text caption to determine if it provides detailed descriptions of objects that align with the image. Specifically, assess if the caption sufficiently describes the color, size, position, shape, material, etc., of the objects. Afterward, rate the caption's overall accuracy in capturing object details from the image on a scale of 1-100, based on the criteria provided."

criteria_caption_text_quality = "Please evaluate the text caption based on the following criteria: Grammatical Correctness, Diversity of Vocabulary (e.g., the range and uniqueness of words used), Fluency (e.g., smoothness and natural flow of sentences), Readability, Length, and Structure. Assign an overall quality score on a scale of 1-100."

criteria_semantic_understanding = """Evaluate the given text caption in relation to its corresponding image. Your goal is to determine if the text caption provides additional semantic information that isn't readily apparent just from the image itself.

For example:

1. If the image mentions "a man" but the caption elaborates he is a "homeless man" or a "businessman," then the caption is enriching the semantic context.
2. If the caption introduces concepts like the mathematical tangent function, which require in-depth knowledge to deduce, it is imparting external semantics.
3. Captions revealing specific location addresses, festival details, or other nuanced data not easy to infer from the image also provide external semantic information.
4. Directly identifying specific entities in the image such as buildings, people, bird species, animal breeds, car models, engines, etc., in the caption introduces additional insights.
5. Should the image act as a contextual backdrop and the caption describes elements not explicitly showcased in the image, it has semantic depth.
6. Lastly, if the caption depicts relationships between the subjects in the image, which need commonsense knowledge to understand, it should be considered semantically rich.

Please assess and determine the extent of semantic enrichment the caption provides over the image. Rate the text caption's semantic depth on a scale from 1 to 100."""

ALL_METRICS = {
    "image_text_matching": criteria_image_text_matching,
    "object_detail_fulfillment": criteria_object_detail_fulfillment,
    "caption_text_quality": criteria_caption_text_quality,
    "semantic_understanding": criteria_semantic_understanding,
}

@dataclass
class DataCollatorForImagePreprocessing:
    """DataCollator matching the original implementation exactly"""
    def __init__(self, tokenizer, image_processor, mm_use_im_start_end, criteria, task, conv_mode, max_len):
        self.image_processor = image_processor
        self.mm_use_im_start_end = mm_use_im_start_end
        self.tokenizer = tokenizer
        self.conv_mode = conv_mode
        self.criteria = criteria
        self.task = task
        self.max_len = max_len
        self.whitespace_id = self.tokenizer(" ")["input_ids"][-1]

    def format_text(self, text: str):
        """Format text exactly as in original"""
        text = LLAVA_INSTRUCTION_TEMPLATE.format(
            caption=text, 
            criteria=self.criteria, 
            aspect=self.task.replace("_", " ")
        )
        if self.mm_use_im_start_end:
            text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text
        else:
            text = DEFAULT_IMAGE_TOKEN + "\n" + text

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()

        return text

    def pad_sequence(self, sequence, padding_value=0):
        """Pad sequence with flip logic for left padding - EXACTLY as original"""
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in sequence]
        else:
            input_ids = sequence
            
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
            
        return input_ids

    def __call__(self, batch: Tuple[List, List]) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """Process batch exactly as in original"""
        images, json_data = batch
        
        # Extract captions from JSON data
        captions = []
        infos = []
        for j in json_data:
            if 'conversations' in j and len(j['conversations']) > 1:
                caption = j['conversations'][1]['value']
            else:
                caption = j.get('caption', '')
            captions.append(caption)
            infos.append({
                'sample_id': j.get('id', 'unknown'),
                'caption': caption,
                '__key__': j.get('__key__', j.get('id', 'unknown'))
            })
        
        # Format prompts
        prompts = [self.format_text(caption) for caption in captions]
        
        # Tokenize with truncation at max_len
        batch_input_ids = [
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")[:self.max_len]
            for prompt in prompts
        ]
        
        # Pad sequences using the flip logic
        batch_input_ids = self.pad_sequence(batch_input_ids)
        
        # Add whitespace token exactly as original
        batch_input_ids = torch.cat(
            (batch_input_ids, torch.tensor([[self.whitespace_id]]).repeat(batch_input_ids.shape[0], 1)), 
            dim=1
        )
        
        # Process images with direct image_processor call (NOT process_images)
        batch_image_tensor = self.image_processor(images, return_tensors="pt")["pixel_values"]
        
        return (batch_image_tensor, batch_input_ids, infos)


def setup_logging(config):
    """Set up logging configuration"""
    log_level = getattr(logging, config['logging']['level'], logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger('MLM_Pipeline_Fixed')
    logger.setLevel(log_level)
    logger.addHandler(console_handler)
    
    if 'log_file' in config['logging']:
        file_handler = logging.FileHandler(config['logging']['log_file'])
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_model_on_gpu(model_name, device_id):
    """Load model on specific GPU - matching original's approach"""
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')
    
    disable_torch_init()
    
    model_path = model_name
    model_name_parsed = get_model_name_from_path(model_path)
    
    # IMPORTANT: The original uses 'device' parameter, but for multi-GPU we need device_map
    # to avoid device mismatch errors. The working multi-GPU implementation uses device_map.
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name_parsed, device_map=device
    )
    
    # Set padding side exactly as original
    model.config.tokenizer_padding_side = tokenizer.padding_side = "left"
    
    model.eval()
    return tokenizer, model, image_processor, device, context_len


def worker_process(gpu_id, input_queue, output_queue, model_name, conv_mode, max_new_tokens, max_len, logger_queue):
    """Worker process matching original's processing logic"""
    try:
        # Set up GPU
        torch.cuda.set_device(gpu_id)
        
        # Load model once
        tokenizer, model, image_processor, device, context_len = load_model_on_gpu(model_name, gpu_id)
        
        # Get mm_use_im_start_end from model config
        mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
        
        logger_queue.put(('info', f'Worker {gpu_id}: Model loaded successfully'))
        
        while True:
            batch_data = input_queue.get()
            if batch_data is None:  # Sentinel value to stop
                break
            
            try:
                results = []
                
                # batch_data is a tuple of (images_list, json_list) from batched webdataset
                if isinstance(batch_data, tuple) and len(batch_data) == 2:
                    images_list, json_list = batch_data
                else:
                    logger_queue.put(('error', f'Worker {gpu_id}: Unexpected batch format: {type(batch_data)}'))
                    continue
                
                # Ensure images are RGB (decode("pilrgb") should handle this, but make sure)
                images_rgb = []
                for img in images_list:
                    if not isinstance(img, Image.Image):
                        if isinstance(img, bytes):
                            img = Image.open(io.BytesIO(img))
                    img = img.convert('RGB')  # Force RGB like original
                    images_rgb.append(img)
                
                # Process each metric separately like original
                for metric_name, criteria in ALL_METRICS.items():
                    # Create collator for this metric
                    collator = DataCollatorForImagePreprocessing(
                        tokenizer, image_processor, mm_use_im_start_end, 
                        criteria, metric_name, conv_mode, max_len
                    )
                    
                    # Use collator to process batch - it will extract captions from JSON
                    batch_image_tensor, batch_input_ids, batch_infos = collator((images_rgb, json_list))
                    
                    # Move to GPU
                    batch_image_tensor = batch_image_tensor.to(device, dtype=torch.float16)
                    batch_input_ids = batch_input_ids.to(device)
                    
                    # Generate - matching original exactly
                    with torch.inference_mode():
                        output_ids = model.generate(
                            batch_input_ids,
                            images=batch_image_tensor,
                            do_sample=False,
                            max_new_tokens=max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.pad_token_id
                            # NO eos_token_id, NO attention_mask - match original
                        )
                    
                    # Decode FULL output like original, not just generated tokens
                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    
                    # Store outputs exactly as original
                    for i, output in enumerate(outputs):
                        if metric_name == "image_text_matching":
                            results.append({
                                'sample_id': batch_infos[i]['sample_id'],
                                'caption': batch_infos[i]['caption'],
                                '__key__': batch_infos[i]['__key__'],
                                f'{metric_name}_score': output  # Store full output like original
                            })
                        else:
                            # Update existing result
                            for r in results:
                                if r['sample_id'] == batch_infos[i]['sample_id']:
                                    r[f'{metric_name}_score'] = output
                                    break
                
                output_queue.put(('result', results))
                
            except Exception as e:
                logger_queue.put(('error', f'Worker {gpu_id}: Error processing batch: {e}\n{traceback.format_exc()}'))
                output_queue.put(('error', str(e)))
    
    except Exception as e:
        logger_queue.put(('error', f'Worker {gpu_id}: Fatal error: {e}\n{traceback.format_exc()}'))


def logger_process(logger_queue, logger):
    """Process for handling logging from workers"""
    while True:
        log_item = logger_queue.get()
        if log_item is None:
            break
        
        level, message = log_item
        if level == 'info':
            logger.info(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'error':
            logger.error(message)


def main():
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    
    parser = argparse.ArgumentParser(description='Fixed Multi-GPU MLM Filter Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--num_gpus', type=int, default=4,
                       help='Number of GPUs to use')
    parser.add_argument('--test_mode', action='store_true',
                       help='Run in test mode with small sample')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up logging
    logger = setup_logging(config)
    
    logger.info("="*60)
    logger.info("Fixed Multi-GPU MLM Filter Pipeline")
    logger.info("="*60)
    logger.info(f"Using {args.num_gpus} GPUs")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Input: {config['data']['input_dir']}")
    logger.info(f"Output: {config['data']['output_dir']}")
    logger.info(f"Model: {config['model']['name']}")
    
    # Determine conv_mode like original
    model_name = get_model_name_from_path(config['model']['name'])
    if 'qwen' in model_name.lower():
        conv_mode = "qwen_instruct"
        max_new_tokens = 2
    elif 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
        max_new_tokens = 2
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
        max_new_tokens = 2
    elif "llama-3" in model_name.lower():
        conv_mode = "llama_3"
        max_new_tokens = 1
    else:
        conv_mode = "llava_v0"
        max_new_tokens = 2
    
    logger.info(f"Conv mode: {conv_mode}")
    logger.info(f"Max new tokens: {max_new_tokens}")
    
    # Create output directory
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Create queues
    input_queues = [mp.Queue(maxsize=2) for _ in range(args.num_gpus)]
    output_queue = mp.Queue()
    logger_queue = mp.Queue()
    
    # Start logger process
    logger_proc = mp.Process(target=logger_process, args=(logger_queue, logger))
    logger_proc.start()
    
    # Start worker processes
    processes = []
    max_len = config['processing'].get('max_len', 2040)
    
    for gpu_id in range(args.num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, input_queues[gpu_id], output_queue, 
                  config['model']['name'], conv_mode, max_new_tokens, max_len, logger_queue)
        )
        p.start()
        processes.append(p)
        logger.info(f"Started worker process on GPU {gpu_id}")
    
    # Load dataset with pilrgb decode like original
    input_dir = Path(config['data']['input_dir'])
    tar_files = sorted(input_dir.glob(config['data'].get('tar_pattern', '*.tar')))
    
    if args.test_mode:
        tar_files = tar_files[:1]  # Use only first tar for testing
        logger.info("TEST MODE: Processing only first tar file")
    elif config['data'].get('max_tars'):
        tar_files = tar_files[:config['data']['max_tars']]
    
    logger.info(f"Found {len(tar_files)} tar files to process")
    
    # Create WebDataset pipeline matching original more closely
    urls = [str(tf) for tf in tar_files]
    
    # Build pipeline similar to original
    pipeline = [
        wds.SimpleShardList(urls),
        # No split_by_worker in multi-GPU context (handled by queue distribution)
        wds.tarfile_to_samples(),
        wds.decode("pilrgb"),  # CRITICAL: Use pilrgb like original, not pil
        wds.rename(image="jpg;png", json="json"),  # Adapt for JSON data
        wds.to_tuple("image", "json"),
    ]
    
    # Apply sampling if configured (must be before batching)
    if not args.test_mode and config['sampling'].get('enabled', False):
        sample_rate = config['sampling'].get('sample_rate', 1.0)
        if sample_rate < 1.0:
            # Add select to pipeline for random sampling
            pipeline.append(wds.select(lambda _: random.random() < sample_rate))
            logger.info(f"Sampling enabled at {sample_rate*100:.1f}%")
    
    # Add batching after sampling
    pipeline.append(wds.batched(config['processing']['batch_size'], partial=True))  # partial=True like original
    
    dataset = wds.DataPipeline(*pipeline)
    
    # Process batches
    all_results = []
    start_time = time.time()
    batch_count = 0
    gpu_idx = 0
    error_count = 0
    total_processed = 0
    pending_results = []
    
    logger.info("Starting multi-GPU processing...")
    
    # Process dataset
    for batch in tqdm(dataset, desc="Distributing batches"):
        # batch is already a list of tuples from batched()
        batch_data = batch
        
        # Send to next GPU in round-robin
        input_queues[gpu_idx].put(batch_data)
        pending_results.append(gpu_idx)
        gpu_idx = (gpu_idx + 1) % args.num_gpus
        batch_count += 1
        
        # Collect results if queues are full
        if len(pending_results) >= args.num_gpus * 2:
            result_type, result_data = output_queue.get()
            if result_type == 'result':
                all_results.extend(result_data)
                total_processed += len(result_data)
            else:
                error_count += 1
                logger.error(f"Error from worker: {result_data}")
            
            pending_results.pop(0)
            
            # Log progress
            elapsed = time.time() - start_time
            speed = total_processed / elapsed if elapsed > 0 else 0
            logger.info(f"Processed {total_processed} samples, {speed:.1f} samples/s, {error_count} errors")
            
            # Save intermediate results
            if config['output'].get('save_intermediate', False):
                if batch_count % config['output'].get('intermediate_frequency', 100) == 0:
                    df = pd.DataFrame(all_results)
                    intermediate_path = output_dir / f'intermediate_{batch_count}.parquet'
                    df.to_parquet(intermediate_path, compression=config['output'].get('compression', 'snappy'))
                    logger.info(f"Saved intermediate results ({len(df)} samples)")
        
        # Break early in test mode
        if args.test_mode and batch_count >= 3:
            logger.info("TEST MODE: Processed 3 batches, stopping")
            break
    
    # Collect remaining results
    for _ in pending_results:
        result_type, result_data = output_queue.get()
        if result_type == 'result':
            all_results.extend(result_data)
            total_processed += len(result_data)
        else:
            error_count += 1
            logger.error(f"Error from worker: {result_data}")
    
    # Stop worker processes
    for queue in input_queues:
        queue.put(None)
    
    for p in processes:
        p.join()
    
    # Stop logger process
    logger_queue.put(None)
    logger_proc.join()
    
    # Save and analyze results
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Post-process scores (extract from full text if needed)
        for metric in ALL_METRICS.keys():
            col_name = f"{metric}_score"
            if col_name in df.columns:
                # Extract numeric score from text
                def extract_score(text):
                    if pd.isna(text) or text == '':
                        return -1
                    # Try to extract first number from text
                    text_str = str(text).strip()
                    # Split and find first token that looks like a number
                    for token in text_str.split():
                        try:
                            score = int(token)
                            if 0 <= score <= 100:
                                return score
                        except ValueError:
                            continue
                    return -1
                
                df[col_name] = df[col_name].apply(extract_score)
        
        # Save final results
        final_path = output_dir / 'mlm_scores_final.parquet'
        df.to_parquet(final_path, compression=config['output'].get('compression', 'snappy'))
        
        total_time = time.time() - start_time
        
        logger.info("\n" + "="*60)
        logger.info("PROCESSING COMPLETE")
        logger.info("="*60)
        logger.info(f"Total samples processed: {len(df)}")
        logger.info(f"Total errors: {error_count}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average speed: {len(df)/total_time:.2f} samples/second")
        logger.info(f"Results saved to: {final_path}")
        
        # Show score distribution
        logger.info("\nScore distributions:")
        for metric in ALL_METRICS.keys():
            col_name = f"{metric}_score"
            if col_name in df.columns:
                valid_scores = df[df[col_name] != -1][col_name]
                if len(valid_scores) > 0:
                    logger.info(f"  {metric}:")
                    logger.info(f"    Valid: {len(valid_scores)}/{len(df)} ({100*len(valid_scores)/len(df):.1f}%)")
                    logger.info(f"    Mean: {valid_scores.mean():.1f}, Std: {valid_scores.std():.1f}")
                    logger.info(f"    Min: {valid_scores.min()}, Max: {valid_scores.max()}")
                    unique_vals = sorted(valid_scores.unique())
                    logger.info(f"    Unique values: {unique_vals[:20]}")
        
        if not args.test_mode:
            # Estimate for full dataset
            full_dataset_size = 558000
            estimated_time = full_dataset_size / (len(df)/total_time)
            logger.info(f"\nEstimated time for full dataset ({full_dataset_size:,} samples):")
            logger.info(f"  {estimated_time/3600:.1f} hours ({estimated_time/3600/24:.1f} days)")
            logger.info(f"  Using {args.num_gpus} GPUs at {len(df)/total_time:.1f} samples/second")
    else:
        logger.error("No results collected!")


if __name__ == "__main__":
    main()