"""
Analyze and visualize image metrics distribution from WebDataset processing results.
Extract samples by percentiles and create visualization with metrics overlay.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
from PIL import Image, ImageDraw, ImageFont
import webdataset as wds
import warnings
from tqdm import tqdm
from scipy import stats
import io


class MetricsAnalyzer:
    """Analyzer for image metrics with visualization and sample extraction capabilities."""
    
    def __init__(self, 
                 metrics_path: str,
                 dataset_path: str,
                 output_dir: str = "./analysis_output",
                 use_metrics: Optional[List[str]] = None,
                 normalization_method: str = 'auto'):
        """
        Initialize metrics analyzer.
        
        Args:
            metrics_path: Path to directory containing metric parquet files
            dataset_path: Path to WebDataset tar files
            output_dir: Directory for output visualizations and samples
            use_metrics: List of metrics to use for average score calculation
            normalization_method: 'auto', 'percentile_rank', 'percentile_clip', or 'minmax'
        """
        self.metrics_path = Path(metrics_path)
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_metrics = use_metrics
        self.normalization_method = normalization_method
        
        # Load metrics data
        self.df = self.load_metrics()
        
        # Compute average score for sorting
        self.compute_average_scores(use_metrics=self.use_metrics, 
                                   normalization_method=normalization_method)
        
    def load_metrics(self) -> pd.DataFrame:
        """Load all metrics parquet files."""
        parquet_files = sorted(self.metrics_path.glob("*_metrics.parquet"))
        
        if not parquet_files:
            raise ValueError(f"No parquet files found in {self.metrics_path}")
        
        dfs = []
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(combined_df)} samples from {len(parquet_files)} files")
        
        return combined_df
    
    def compute_average_scores(self, use_metrics=None, normalization_method='auto'):
        """Compute average score across selected metrics for each sample.
        
        Args:
            use_metrics: List of metric names to include in average. 
                        If None, uses all available metrics.
                        Options: 'luminance_entropy', 'spatial_information',
                                'edge_density', 'variance_of_laplacian', 'brenner_focus'
            normalization_method: Method for normalizing metrics
                        'auto': Automatically choose based on number of metrics
                        'percentile_rank': Percentile ranking (best for single metrics)
                        'percentile_clip': 5-95 percentile with clipping
                        'minmax': Min-max normalization
        """
        all_metrics = ['luminance_entropy', 'spatial_information', 
                      'edge_density', 'variance_of_laplacian', 'brenner_focus']
        
        if use_metrics is None:
            metric_cols = all_metrics
        else:
            metric_cols = [m for m in use_metrics if m in all_metrics]
            print(f"Using metrics for average score: {metric_cols}")
        
        # Determine normalization method
        if normalization_method == 'auto':
            # Auto-select based on number of metrics
            if len(metric_cols) == 1:
                method = 'percentile_rank'
            else:
                method = 'percentile_clip'
        else:
            method = normalization_method
        
        print(f"Normalization method: {method}")
        
        # Normalize each metric to 0-1 range for fair averaging
        for col in metric_cols:
            if col in self.df.columns:
                values = self.df[col].values
                
                if method == 'percentile_rank':
                    # Percentile ranking for uniform distribution
                    from scipy import stats
                    normalized = stats.rankdata(values, method='average') / len(values)
                    self.df[f'{col}_normalized'] = normalized
                    
                elif method == 'percentile_clip':
                    # Robust scaling with 5-95 percentile clipping
                    p5 = self.df[col].quantile(0.05)
                    p95 = self.df[col].quantile(0.95)
                    if p95 > p5:
                        self.df[f'{col}_normalized'] = (self.df[col] - p5) / (p95 - p5)
                        self.df[f'{col}_normalized'] = self.df[f'{col}_normalized'].clip(0, 1)
                    else:
                        self.df[f'{col}_normalized'] = 0.5
                        
                elif method == 'minmax':
                    # Min-max normalization
                    min_val = values.min()
                    max_val = values.max()
                    if max_val > min_val:
                        self.df[f'{col}_normalized'] = (values - min_val) / (max_val - min_val)
                    else:
                        self.df[f'{col}_normalized'] = 0.5
                else:
                    raise ValueError(f"Unknown normalization method: {method}")
        
        # Compute average normalized score
        norm_cols = [f'{col}_normalized' for col in metric_cols if col in self.df.columns]
        self.df['avg_score'] = self.df[norm_cols].mean(axis=1)
        
        print(f"Average score computed from {len(norm_cols)} metrics")
        print(f"Average score range: [{self.df['avg_score'].min():.3f}, {self.df['avg_score'].max():.3f}]")
    
    def analyze_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of metrics."""
        analysis = {
            'total_samples': len(self.df),
            'metrics': {},
            'percentiles': {}
        }
        
        metric_cols = ['luminance_entropy', 'spatial_information', 
                      'edge_density', 'variance_of_laplacian', 'brenner_focus', 'avg_score']
        
        for col in metric_cols:
            if col not in self.df.columns:
                continue
            
            values = self.df[col].dropna()
            
            analysis['metrics'][col] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'median': float(values.median()),
                'skewness': float(stats.skew(values)),
                'kurtosis': float(stats.kurtosis(values))
            }
            
            # Compute percentiles
            percentiles = {}
            for p in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
                percentiles[f'p{p}'] = float(values.quantile(p/100))
            analysis['percentiles'][col] = percentiles
        
        return analysis
    
    def plot_distributions(self, save_path: Optional[str] = None):
        """Create distribution plots for all metrics."""
        metric_cols = ['luminance_entropy', 'spatial_information', 
                      'edge_density', 'variance_of_laplacian', 'brenner_focus', 'avg_score']
        
        # Filter to only existing columns
        metric_cols = [col for col in metric_cols if col in self.df.columns]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(metric_cols):
            ax = axes[idx]
            values = self.df[col].dropna()
            
            # Histogram with KDE
            ax.hist(values, bins=50, alpha=0.7, density=True, edgecolor='black')
            values.plot.kde(ax=ax, color='red', linewidth=2, label='KDE')
            
            # Add percentile lines
            for p in [25, 50, 75]:
                val = values.quantile(p/100)
                ax.axvline(val, color='green', linestyle='--', alpha=0.5)
                ax.text(val, ax.get_ylim()[1]*0.9, f'P{p}', rotation=90)
            
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide extra subplot
        if len(metric_cols) < len(axes):
            for idx in range(len(metric_cols), len(axes)):
                axes[idx].set_visible(False)
        
        plt.suptitle('Metrics Distribution Analysis', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, save_path: Optional[str] = None):
        """Plot correlation matrix of metrics."""
        metric_cols = ['luminance_entropy', 'spatial_information', 
                      'edge_density', 'variance_of_laplacian', 'brenner_focus']
        metric_cols = [col for col in metric_cols if col in self.df.columns]
        
        if len(metric_cols) < 2:
            print("Not enough metrics for correlation analysis")
            return
        
        # Compute correlation matrix
        corr = self.df[metric_cols].corr()
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Metrics Correlation Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Correlation matrix saved to {save_path}")
        
        plt.show()
    
    def extract_percentile_samples(self, 
                                  n_percentiles: int = 10,
                                  samples_per_percentile: int = 5,
                                  percentile_ranges: Optional[List[float]] = None) -> Dict[str, pd.DataFrame]:
        """
        Extract samples from different percentile ranges.
        
        Args:
            n_percentiles: Number of percentile bins (e.g., 10 for deciles)
            samples_per_percentile: Number of samples to extract per percentile
            percentile_ranges: Custom percentile boundaries (e.g., [0, 10, 25, 50, 75, 90, 100])
            
        Returns:
            Dictionary mapping percentile ranges to DataFrames of samples
        """
        # Sort by average score
        sorted_df = self.df.sort_values('avg_score')
        
        # Create percentile bins
        percentile_samples = {}
        n = len(sorted_df)
        
        if percentile_ranges:
            # Use custom percentile ranges
            percentile_ranges = sorted(percentile_ranges)
            for i in range(len(percentile_ranges) - 1):
                p_start = percentile_ranges[i]
                p_end = percentile_ranges[i + 1]
                
                # Calculate indices
                start_idx = int(p_start * n / 100)
                end_idx = int(p_end * n / 100)
                
                # Get samples from this range
                range_df = sorted_df.iloc[start_idx:end_idx]
                
                if len(range_df) == 0:
                    continue
                
                # Sample randomly from this range
                sample_size = min(samples_per_percentile, len(range_df))
                samples = range_df.sample(n=sample_size, random_state=42)
                
                # Calculate score range for this percentile
                min_score = range_df['avg_score'].min()
                max_score = range_df['avg_score'].max()
                
                percentile_key = f"p{int(p_start):02d}-{int(p_end):02d}"
                score_range_key = f"{percentile_key}_score_{min_score:.3f}-{max_score:.3f}"
                
                percentile_samples[score_range_key] = samples
                
                print(f"Percentile {percentile_key}: {len(samples)} samples, "
                      f"score range [{min_score:.3f}, {max_score:.3f}]")
        else:
            # Use equal bins
            for i in range(n_percentiles):
                # Calculate range
                start_idx = int(i * n / n_percentiles)
                end_idx = int((i + 1) * n / n_percentiles)
                
                # Get samples from this range
                range_df = sorted_df.iloc[start_idx:end_idx]
                
                # Sample randomly from this range
                sample_size = min(samples_per_percentile, len(range_df))
                samples = range_df.sample(n=sample_size, random_state=42)
                
                # Calculate score range for this percentile
                min_score = range_df['avg_score'].min()
                max_score = range_df['avg_score'].max()
                
                percentile_key = f"p{i*100//n_percentiles:02d}-{(i+1)*100//n_percentiles:02d}"
                score_range_key = f"{percentile_key}_score_{min_score:.3f}-{max_score:.3f}"
                
                percentile_samples[score_range_key] = samples
                
                print(f"Percentile {percentile_key}: {len(samples)} samples, "
                      f"score range [{min_score:.3f}, {max_score:.3f}]")
        
        return percentile_samples
    
    def load_image_from_webdataset(self, key: str, shard_idx: int) -> Optional[np.ndarray]:
        """Load a specific image from WebDataset tar file."""
        tar_path = self.dataset_path / f"{shard_idx:05d}.tar"
        
        if not tar_path.exists():
            print(f"Tar file not found: {tar_path}")
            return None
        
        # Open WebDataset and search for the key
        dataset = wds.WebDataset(str(tar_path), shardshuffle=False)
        
        for sample in dataset:
            if sample.get('__key__') == key:
                # Extract image
                if 'jpg' in sample:
                    img_bytes = sample['jpg']
                elif 'png' in sample:
                    img_bytes = sample['png']
                else:
                    continue
                
                # Decode image
                img = Image.open(io.BytesIO(img_bytes))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return np.array(img)
        
        return None
    
    def create_image_with_metrics(self, 
                                 image: np.ndarray, 
                                 metrics: Dict[str, float],
                                 title: str = "",
                                 used_metrics: Optional[List[str]] = None) -> np.ndarray:
        """
        Create visualization with image and metrics overlay.
        
        Args:
            image: Input image
            metrics: Dictionary of metric values
            title: Optional title for the image
            used_metrics: List of metrics used in scoring (if None, shows all)
            
        Returns:
            Image with metrics overlay
        """
        # Convert to PIL for easier text overlay
        if image.shape[0] > 512 or image.shape[1] > 512:
            # Resize if too large
            scale = min(512/image.shape[0], 512/image.shape[1])
            new_h = int(image.shape[0] * scale)
            new_w = int(image.shape[1] * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        img_pil = Image.fromarray(image)
        
        # Calculate text height based on metrics being displayed
        num_metrics = len(used_metrics) if used_metrics else 4
        text_height = 40 + (num_metrics + 1) * 18  # Title + metrics + avg_score
        new_height = img_pil.height + text_height
        new_img = Image.new('RGB', (img_pil.width, new_height), color='white')
        new_img.paste(img_pil, (0, 0))
        
        # Add text
        draw = ImageDraw.Draw(new_img)
        
        # Try to use a better font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        except:
            font = ImageFont.load_default()
            title_font = font
        
        y_offset = img_pil.height + 5
        
        # Title
        if title:
            draw.text((5, y_offset), title, fill='black', font=title_font)
            y_offset += 18
        
        # Metrics with both raw and normalized values
        text_lines = []
        
        # Format each metric with raw and normalized values
        all_metric_names = [
            ('luminance_entropy', 'Entropy'),
            ('spatial_information', 'SI'),
            ('edge_density', 'Edge'),
            ('variance_of_laplacian', 'VoL'),
            ('brenner_focus', 'Brenner')
        ]
        
        # Filter to only show used metrics if specified
        if used_metrics is not None:
            metric_names = [(key, name) for key, name in all_metric_names if key in used_metrics]
        else:
            metric_names = all_metric_names
        
        for metric_key, display_name in metric_names:
            raw_val = metrics.get(metric_key, 0)
            norm_key = f"{metric_key}_normalized"
            norm_val = metrics.get(norm_key, None)
            
            # Format based on metric type
            if metric_key in ['luminance_entropy', 'edge_density']:
                raw_str = f"{raw_val:.3f}"
            elif metric_key == 'spatial_information':
                raw_str = f"{raw_val:.1f}"
            elif metric_key in ['variance_of_laplacian', 'brenner_focus']:
                raw_str = f"{raw_val:.0f}"
            else:
                raw_str = f"{raw_val:.3f}"
            
            if norm_val is not None:
                line = f"{display_name}: {raw_str} (norm: {norm_val:.3f})"
            else:
                line = f"{display_name}: {raw_str}"
            
            text_lines.append(line)
        
        # Add average score
        text_lines.append(f"Avg Score: {metrics.get('avg_score', 0):.3f}")
        
        for line in text_lines:
            draw.text((5, y_offset), line, fill='black', font=font)
            y_offset += 14
        
        return np.array(new_img)
    
    def save_percentile_samples_with_images(self,
                                           percentile_samples: Dict[str, pd.DataFrame],
                                           output_base_dir: Optional[str] = None,
                                           skip_missing: bool = True,
                                           show_all_metrics: bool = False):
        """
        Save sample images organized by percentile with metrics overlay.
        
        Args:
            percentile_samples: Dictionary of percentile DataFrames
            output_base_dir: Base directory for output
            skip_missing: Continue if images cannot be loaded
            show_all_metrics: If True, show all metrics; if False, only show selected metrics
        """
        if output_base_dir is None:
            output_base_dir = self.output_dir / "percentile_samples"
        else:
            output_base_dir = Path(output_base_dir)
        
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        total_saved = 0
        total_failed = 0
        
        for percentile_key, samples_df in tqdm(percentile_samples.items(), desc="Saving percentile samples"):
            # Create directory for this percentile
            percentile_dir = output_base_dir / percentile_key
            percentile_dir.mkdir(exist_ok=True)
            
            saved_count = 0
            # Save each sample
            for idx, row in samples_df.iterrows():
                # Load image
                img = self.load_image_from_webdataset(row['key'], row['shard_idx'])
                
                if img is None:
                    print(f"Warning: Failed to load image for key {row['key']} from shard {row['shard_idx']}")
                    total_failed += 1
                    if not skip_missing:
                        raise ValueError(f"Could not load image {row['key']} from shard {row['shard_idx']}")
                    continue
                
                # Prepare metrics with both raw and normalized values
                metrics = {
                    'luminance_entropy': row.get('luminance_entropy', 0),
                    'spatial_information': row.get('spatial_information', 0),
                    'edge_density': row.get('edge_density', 0),
                    'variance_of_laplacian': row.get('variance_of_laplacian', 0),
                    'brenner_focus': row.get('brenner_focus', 0),
                    'avg_score': row.get('avg_score', 0),
                    # Include normalized values if they exist
                    'luminance_entropy_normalized': row.get('luminance_entropy_normalized', None),
                    'spatial_information_normalized': row.get('spatial_information_normalized', None),
                    'edge_density_normalized': row.get('edge_density_normalized', None),
                    'variance_of_laplacian_normalized': row.get('variance_of_laplacian_normalized', None),
                    'brenner_focus_normalized': row.get('brenner_focus_normalized', None)
                }
                
                # Create visualization
                title = f"Key: {row['key']}"
                # Use all metrics if show_all_metrics is True, otherwise use only selected metrics
                display_metrics = None if show_all_metrics else self.use_metrics
                vis_img = self.create_image_with_metrics(img, metrics, title, used_metrics=display_metrics)
                
                # Save image
                output_path = percentile_dir / f"{row['key']}.jpg"
                cv2.imwrite(str(output_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                saved_count += 1
                total_saved += 1
            
            print(f"Saved {saved_count}/{len(samples_df)} samples to {percentile_dir}")
        
        print(f"\nTotal: Saved {total_saved} images, Failed to load {total_failed} images")
    
    def generate_analysis_report(self, output_path: Optional[str] = None):
        """Generate comprehensive analysis report."""
        if output_path is None:
            output_path = self.output_dir / "analysis_report.json"
        
        analysis = self.analyze_distribution()
        
        # Add threshold recommendations
        analysis['threshold_recommendations'] = {}
        
        for metric in ['luminance_entropy', 'spatial_information', 
                      'edge_density', 'variance_of_laplacian', 'brenner_focus']:
            if metric not in self.df.columns:
                continue
            
            values = self.df[metric].dropna()
            
            # Recommend thresholds based on percentiles
            recommendations = {
                'very_strict': {  # Top 50%
                    'min': float(values.quantile(0.5)),
                    'description': 'Keep only top 50% quality images'
                },
                'strict': {  # Top 70%
                    'min': float(values.quantile(0.3)),
                    'description': 'Keep top 70% quality images'
                },
                'moderate': {  # Top 85%
                    'min': float(values.quantile(0.15)),
                    'description': 'Keep top 85% quality images'
                },
                'permissive': {  # Top 95%
                    'min': float(values.quantile(0.05)),
                    'description': 'Keep top 95% quality images'
                }
            }
            
            analysis['threshold_recommendations'][metric] = recommendations
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Analysis report saved to {output_path}")
        
        return analysis