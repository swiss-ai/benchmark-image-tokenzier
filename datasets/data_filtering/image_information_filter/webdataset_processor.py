"""
WebDataset processor for computing image information metrics on large-scale datasets.
Efficiently processes WebDataset tar files in parallel and saves results to sharded parquet files.
"""

import os
import json
import time
import io
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import warnings
import traceback

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import webdataset as wds
import cv2
from PIL import Image
from tqdm import tqdm

from image_metrics import ImageMetrics, ColorSpace


@dataclass
class WebDatasetSample:
    """Container for WebDataset sample data"""
    key: str
    image: Optional[np.ndarray]
    json_data: Dict[str, Any]
    text: Optional[str]
    shard_idx: int
    error: Optional[str] = None


@dataclass
class ProcessedSample:
    """Container for processed sample with metrics"""
    # Original metadata
    key: str
    shard_idx: int
    image_id: str
    image_path: str
    caption: str
    url: str
    
    # Original image properties
    original_width: Optional[int]
    original_height: Optional[int]
    width: Optional[int]
    height: Optional[int]
    sha256: Optional[str]
    
    # Computed metrics
    luminance_entropy: Optional[float] = None
    spatial_information: Optional[float] = None
    edge_density: Optional[float] = None
    variance_of_laplacian: Optional[float] = None
    brenner_focus: Optional[float] = None
    
    # Processing metadata
    processing_time: Optional[float] = None
    processing_error: Optional[str] = None
    metrics_computed: bool = False


class WebDatasetProcessor:
    """
    Processor for computing image metrics on WebDataset format data.
    Supports parallel processing and sharded output.
    """
    
    def __init__(self, 
                 dataset_path: str,
                 output_path: str,
                 reference_size: Optional[Tuple[int, int]] = (512, 512),
                 color_space: str = 'bt709',
                 num_workers: int = 4,
                 batch_size: int = 100,
                 metrics_to_compute: Optional[List[str]] = None,
                 resolution_independent: bool = False):
        """
        Initialize WebDataset processor.
        
        Args:
            dataset_path: Path to WebDataset directory containing tar files
            output_path: Path for output parquet files
            reference_size: Target size for normalization (width, height), None for native resolution
            color_space: Color space for conversion ('bt601' or 'bt709')
            num_workers: Number of parallel workers
            batch_size: Batch size for processing
            metrics_to_compute: List of metrics to compute (default: all)
            resolution_independent: If True, compute metrics at native resolution with normalization
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.reference_size = reference_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        
        # Setup metrics calculator
        cs = ColorSpace.BT709 if color_space == 'bt709' else ColorSpace.BT601
        self.metrics_calculator = ImageMetrics(
            reference_size=reference_size,
            color_space=cs,
            use_gpu=False,
            resolution_independent=resolution_independent
        )
        
        # Metrics to compute
        if metrics_to_compute is None:
            self.metrics_to_compute = ['luminance_entropy', 'spatial_information', 
                                      'edge_density', 'variance_of_laplacian']
        else:
            self.metrics_to_compute = metrics_to_compute
        
        # Get list of tar files
        self.tar_files = sorted(self.dataset_path.glob("*.tar"))
        self.num_shards = len(self.tar_files)
        
        print(f"Found {self.num_shards} tar shards in {self.dataset_path}")
    
    def decode_sample(self, sample: Dict) -> WebDatasetSample:
        """
        Decode a WebDataset sample.
        
        Args:
            sample: Raw WebDataset sample dict
            
        Returns:
            Decoded WebDatasetSample
        """
        key = sample['__key__']
        
        # Parse JSON metadata
        json_data = {}
        if 'json' in sample:
            try:
                json_bytes = sample['json']
                if isinstance(json_bytes, bytes):
                    json_str = json_bytes.decode('utf-8')
                else:
                    json_str = json_bytes
                json_data = json.loads(json_str)
            except Exception as e:
                warnings.warn(f"Failed to parse JSON for {key}: {e}")
        
        # Decode image
        image = None
        if 'jpg' in sample or 'png' in sample:
            try:
                img_key = 'jpg' if 'jpg' in sample else 'png'
                img_bytes = sample[img_key]
                
                # Use PIL for robust image decoding
                img_pil = Image.open(io.BytesIO(img_bytes))
                # Convert to RGB if necessary
                if img_pil.mode != 'RGB':
                    img_pil = img_pil.convert('RGB')
                image = np.array(img_pil)
                
            except Exception as e:
                warnings.warn(f"Failed to decode image for {key}: {e}")
                return WebDatasetSample(
                    key=key,
                    image=None,
                    json_data=json_data,
                    text=sample.get('txt', None),
                    shard_idx=-1,
                    error=str(e)
                )
        
        # Get text if available
        text = None
        if 'txt' in sample:
            text_bytes = sample['txt']
            if isinstance(text_bytes, bytes):
                text = text_bytes.decode('utf-8')
            else:
                text = text_bytes
        
        return WebDatasetSample(
            key=key,
            image=image,
            json_data=json_data,
            text=text,
            shard_idx=-1
        )
    
    def compute_metrics_for_sample(self, sample: WebDatasetSample) -> ProcessedSample:
        """
        Compute metrics for a single sample.
        
        Args:
            sample: WebDatasetSample to process
            
        Returns:
            ProcessedSample with computed metrics
        """
        start_time = time.time()
        
        # Extract metadata from JSON
        json_data = sample.json_data
        processed = ProcessedSample(
            key=sample.key,
            shard_idx=sample.shard_idx,
            image_id=json_data.get('image_id', ''),
            image_path=json_data.get('image_path', ''),
            caption=json_data.get('caption', sample.text or ''),
            url=json_data.get('url', ''),
            original_width=json_data.get('original_width'),
            original_height=json_data.get('original_height'),
            width=json_data.get('width'),
            height=json_data.get('height'),
            sha256=json_data.get('sha256')
        )
        
        # Check if we have a valid image
        if sample.image is None:
            processed.processing_error = sample.error or "No image data"
            processed.processing_time = time.time() - start_time
            return processed
        
        # Compute metrics
        try:
            # Convert PIL image to numpy array for processing
            import numpy as np
            img_array = np.array(sample.image)
            
            # Use normalized computation if resolution_independent is enabled
            if hasattr(self.metrics_calculator, 'resolution_independent') and self.metrics_calculator.resolution_independent:
                results = self.metrics_calculator.compute_all_metrics_normalized(img_array)
                for metric_name in self.metrics_to_compute:
                    if metric_name in results:
                        # Results are already normalized values (floats)
                        setattr(processed, metric_name, results[metric_name])
            else:
                # Original metric computation
                if 'luminance_entropy' in self.metrics_to_compute:
                    result = self.metrics_calculator.luminance_entropy(sample.image)
                    processed.luminance_entropy = result.value
                
                if 'spatial_information' in self.metrics_to_compute:
                    result = self.metrics_calculator.spatial_information(sample.image)
                    processed.spatial_information = result.value
                
                if 'edge_density' in self.metrics_to_compute:
                    result = self.metrics_calculator.edge_density(sample.image)
                    processed.edge_density = result.value
                
                if 'variance_of_laplacian' in self.metrics_to_compute:
                    result = self.metrics_calculator.variance_of_laplacian(sample.image)
                    processed.variance_of_laplacian = result.value
                
                if 'brenner_focus' in self.metrics_to_compute:
                    result = self.metrics_calculator.brenner_focus(sample.image)
                    processed.brenner_focus = result.value
            
            processed.metrics_computed = True
            
        except Exception as e:
            processed.processing_error = f"Metric computation error: {str(e)}"
            warnings.warn(f"Failed to compute metrics for {sample.key}: {e}")
        
        processed.processing_time = time.time() - start_time
        return processed
    
    def process_shard(self, shard_idx: int, max_samples: Optional[int] = None) -> List[ProcessedSample]:
        """
        Process a single tar shard.
        
        Args:
            shard_idx: Index of shard to process
            max_samples: Maximum number of samples to process (for testing)
            
        Returns:
            List of ProcessedSample objects
        """
        if shard_idx >= len(self.tar_files):
            raise ValueError(f"Shard index {shard_idx} out of range (max: {len(self.tar_files)-1})")
        
        tar_path = self.tar_files[shard_idx]
        print(f"Processing shard {shard_idx}: {tar_path.name}")
        
        # Create WebDataset pipeline
        dataset = wds.WebDataset(str(tar_path), shardshuffle=False)
        
        results = []
        count = 0
        
        try:
            for sample in dataset:
                if max_samples and count >= max_samples:
                    break
                
                # Get key from sample
                key = sample.get('__key__', '')
                
                # Get image (check both jpg and png)
                img_array = None
                if 'jpg' in sample:
                    img_bytes = sample['jpg']
                    try:
                        img_pil = Image.open(io.BytesIO(img_bytes))
                        if img_pil.mode != 'RGB':
                            img_pil = img_pil.convert('RGB')
                        img_array = np.array(img_pil)
                    except Exception as e:
                        print(f"Failed to decode jpg for {key}: {e}")
                elif 'png' in sample:
                    img_bytes = sample['png']
                    try:
                        img_pil = Image.open(io.BytesIO(img_bytes))
                        if img_pil.mode != 'RGB':
                            img_pil = img_pil.convert('RGB')
                        img_array = np.array(img_pil)
                    except Exception as e:
                        print(f"Failed to decode png for {key}: {e}")
                
                # Get JSON metadata
                json_data = {}
                if 'json' in sample:
                    try:
                        json_bytes = sample['json']
                        if isinstance(json_bytes, bytes):
                            json_str = json_bytes.decode('utf-8')
                        else:
                            json_str = json_bytes
                        json_data = json.loads(json_str)
                    except Exception as e:
                        print(f"Failed to parse JSON for {key}: {e}")
                
                # Get text
                text = None
                if 'txt' in sample:
                    txt_bytes = sample['txt']
                    if isinstance(txt_bytes, bytes):
                        text = txt_bytes.decode('utf-8')
                    else:
                        text = txt_bytes
                
                # Create WebDatasetSample
                if img_array is not None:
                    wds_sample = WebDatasetSample(
                        key=key,
                        image=img_array,
                        json_data=json_data,
                        text=text,
                        shard_idx=shard_idx
                    )
                else:
                    wds_sample = WebDatasetSample(
                        key=key,
                        image=None,
                        json_data=json_data,
                        text=text,
                        shard_idx=shard_idx,
                        error="No valid image found"
                    )
                
                # Process sample
                processed = self.compute_metrics_for_sample(wds_sample)
                results.append(processed)
                count += 1
                
                if count % 100 == 0:
                    print(f"  Processed {count} samples from shard {shard_idx}")
                    
        except Exception as e:
            print(f"Error processing shard {shard_idx}: {e}")
            traceback.print_exc()
        
        print(f"Completed shard {shard_idx}: processed {len(results)} samples")
        return results
    
    def process_shards_parallel(self, 
                               shard_indices: Optional[List[int]] = None,
                               max_samples_per_shard: Optional[int] = None) -> pd.DataFrame:
        """
        Process multiple shards in parallel.
        
        Args:
            shard_indices: List of shard indices to process (default: all)
            max_samples_per_shard: Maximum samples per shard (for testing)
            
        Returns:
            DataFrame with all processed samples
        """
        if shard_indices is None:
            shard_indices = list(range(self.num_shards))
        
        print(f"Processing {len(shard_indices)} shards with {self.num_workers} workers")
        
        all_results = []
        
        # Process shards in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_shard = {
                executor.submit(self.process_shard, idx, max_samples_per_shard): idx
                for idx in shard_indices
            }
            
            # Collect results
            with tqdm(total=len(shard_indices), desc="Processing shards") as pbar:
                for future in as_completed(future_to_shard):
                    shard_idx = future_to_shard[future]
                    try:
                        shard_results = future.result()
                        all_results.extend(shard_results)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Failed to process shard {shard_idx}: {e}")
                        traceback.print_exc()
                        pbar.update(1)
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in all_results])
        print(f"Total samples processed: {len(df)}")
        
        return df
    
    def save_results_to_parquet(self, 
                               df: pd.DataFrame, 
                               preserve_sharding: bool = True) -> None:
        """
        Save results to parquet files, optionally preserving original sharding.
        
        Args:
            df: DataFrame with processed results
            preserve_sharding: If True, save separate parquet per shard
        """
        if preserve_sharding:
            # Group by shard and save separately
            for shard_idx, shard_df in df.groupby('shard_idx'):
                output_file = self.output_path / f"{shard_idx:05d}_metrics.parquet"
                shard_df.to_parquet(output_file, index=False, compression='snappy')
                print(f"Saved {len(shard_df)} samples to {output_file}")
        else:
            # Save as single file
            output_file = self.output_path / "all_metrics.parquet"
            df.to_parquet(output_file, index=False, compression='snappy')
            print(f"Saved {len(df)} samples to {output_file}")
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the processed dataset.
        
        Args:
            df: DataFrame with processed results
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_samples': int(len(df)),
            'successfully_processed': int(df['metrics_computed'].sum()),
            'failed_processing': int((~df['metrics_computed']).sum()),
            'processing_rate': float(df['metrics_computed'].mean()),
            'avg_processing_time': float(df['processing_time'].mean()),
            'total_processing_time': float(df['processing_time'].sum())
        }
        
        # Add metric statistics for successfully processed samples
        success_df = df[df['metrics_computed']]
        
        for metric in self.metrics_to_compute:
            if metric in success_df.columns:
                summary[f'{metric}_mean'] = float(success_df[metric].mean())
                summary[f'{metric}_std'] = float(success_df[metric].std())
                summary[f'{metric}_min'] = float(success_df[metric].min())
                summary[f'{metric}_max'] = float(success_df[metric].max())
                summary[f'{metric}_median'] = float(success_df[metric].median())
        
        # Resolution statistics
        if 'width' in df.columns and 'height' in df.columns:
            valid_res = df[df['width'].notna() & df['height'].notna()]
            if len(valid_res) > 0:
                summary['avg_width'] = float(valid_res['width'].mean())
                summary['avg_height'] = float(valid_res['height'].mean())
                summary['unique_resolutions'] = len(valid_res[['width', 'height']].drop_duplicates())
        
        return summary