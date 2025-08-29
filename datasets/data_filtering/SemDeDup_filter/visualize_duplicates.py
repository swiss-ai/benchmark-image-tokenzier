"""
Visualize and analyze duplicate pairs found by SemDeDup
Extracts duplicate image pairs and creates visualization
"""

import os
import sys
import json
import pickle
import tarfile
import io
from pathlib import Path
from typing import List, Dict, Tuple, Set
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import webdataset as wds
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DuplicateAnalyzer:
    """Analyze and visualize duplicate pairs from SemDeDup results"""
    
    def __init__(self, 
                 semdedup_output_path: str,
                 dataset_path: str,
                 epsilon: float = 0.05):
        """
        Initialize duplicate analyzer
        
        Args:
            semdedup_output_path: Path to SemDeDup output directory
            dataset_path: Path to WebDataset
            epsilon: Epsilon value to analyze
        """
        self.semdedup_path = Path(semdedup_output_path)
        self.dataset_path = Path(dataset_path)
        self.epsilon = epsilon
        
        # Load data
        self.embeddings = None
        self.keys = None
        self.removed_keys = None
        self.duplicate_pairs = []
        
    def load_semdedup_results(self):
        """Load SemDeDup results and identify duplicate pairs"""
        logger.info("Loading SemDeDup results...")
        
        # Load embeddings
        emb_path = self.semdedup_path / "embeddings" / "embeddings.npy"
        metadata_path = self.semdedup_path / "embeddings" / "metadata.json"
        keys_path = self.semdedup_path / "embeddings" / "keys.pkl"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.embeddings = np.memmap(
            emb_path,
            dtype='float32',
            mode='r',
            shape=(metadata['num_samples'], metadata['embedding_dim'])
        )
        
        with open(keys_path, 'rb') as f:
            self.keys = pickle.load(f)
        
        # Create key to index mapping
        self.key_to_idx = {key: idx for idx, key in enumerate(self.keys)}
        
        # Load kept/removed keys
        kept_file = self.semdedup_path / f"kept_samples_eps_{self.epsilon}.txt"
        with open(kept_file, 'r') as f:
            kept_keys = set(line.strip() for line in f)
        
        all_keys = set(self.keys)
        self.removed_keys = all_keys - kept_keys
        
        logger.info(f"Total samples: {len(all_keys)}")
        logger.info(f"Kept samples: {len(kept_keys)}")
        logger.info(f"Removed samples: {len(self.removed_keys)}")
        
    def find_duplicate_pairs(self):
        """Find all duplicate pairs based on clustering and similarity"""
        logger.info("Finding duplicate pairs...")
        
        # Load cluster information
        clusters_path = self.semdedup_path / "clusters" / "sorted_clusters"
        semdedup_results_path = self.semdedup_path / "semdedup_results" / "dataframes"
        
        duplicate_groups = []  # Each group contains indices of duplicates
        
        # Process each cluster
        num_clusters = len(list(clusters_path.glob("cluster_*.npy")))
        
        for cluster_id in tqdm(range(num_clusters), desc="Processing clusters"):
            # Load cluster data
            cluster_file = clusters_path / f"cluster_{cluster_id}.npy"
            cluster_data = np.load(cluster_file)
            
            # Load deduplication results for this cluster
            dedup_file = semdedup_results_path / f"cluster_{cluster_id}.pkl"
            with open(dedup_file, 'rb') as f:
                dedup_df = pickle.load(f)
            
            # Get removed samples in this cluster
            eps_col = f"eps={self.epsilon}"
            if eps_col in dedup_df.columns:
                removed_mask = dedup_df[eps_col].values
                removed_indices = dedup_df[removed_mask]['indices'].values
                
                if len(removed_indices) > 0:
                    # Get cluster member keys
                    cluster_keys = [cluster_data[i][0].decode() if isinstance(cluster_data[i][0], bytes) 
                                   else cluster_data[i][0] for i in range(len(cluster_data))]
                    
                    # For each removed sample, find its nearest neighbor that was kept
                    kept_indices = dedup_df[~removed_mask]['indices'].values
                    
                    if len(kept_indices) > 0:
                        # Compute similarities within cluster
                        cluster_embeddings = []
                        for key in cluster_keys:
                            if key in self.key_to_idx:
                                cluster_embeddings.append(self.embeddings[self.key_to_idx[key]])
                        
                        if len(cluster_embeddings) > 0:
                            cluster_embeddings = np.array(cluster_embeddings)
                            
                            # Compute pairwise similarities
                            similarities = cosine_similarity(cluster_embeddings)
                            
                            # Find duplicate pairs
                            for removed_idx in removed_indices:
                                if removed_idx < len(cluster_keys):
                                    removed_key = cluster_keys[removed_idx]
                                    
                                    # Find most similar kept sample
                                    max_sim = -1
                                    best_kept_idx = None
                                    
                                    for kept_idx in kept_indices:
                                        if kept_idx < len(similarities) and removed_idx < len(similarities):
                                            sim = similarities[removed_idx, kept_idx]
                                            if sim > max_sim:
                                                max_sim = sim
                                                best_kept_idx = kept_idx
                                    
                                    if best_kept_idx is not None and max_sim > (1 - self.epsilon):
                                        kept_key = cluster_keys[best_kept_idx]
                                        self.duplicate_pairs.append({
                                            'removed': removed_key,
                                            'kept': kept_key,
                                            'similarity': float(max_sim),
                                            'cluster_id': cluster_id
                                        })
        
        logger.info(f"Found {len(self.duplicate_pairs)} duplicate pairs")
        
    def extract_images(self, keys: List[str], max_samples: int = 10, max_scan: int = 30000) -> Dict[str, Image.Image]:
        """Extract images from WebDataset for given keys
        
        Args:
            keys: List of keys to extract
            max_samples: Maximum number of samples to extract
            max_scan: Maximum number of samples to scan through dataset (default: 30000)
        """
        logger.info(f"Extracting up to {min(len(keys), max_samples)} images from dataset...")
        
        # Load tar files
        tar_files = sorted(self.dataset_path.glob("*.tar"))
        urls = [str(f) for f in tar_files]
        
        # Create dataset
        dataset = wds.WebDataset(urls, shardshuffle=False).decode("pil")
        
        # Extract images
        images = {}
        keys_to_find = set(keys[:max_samples])
        scanned = 0
        
        for sample in tqdm(dataset, desc="Extracting images", total=max_scan):
            scanned += 1
            if sample["__key__"] in keys_to_find:
                images[sample["__key__"]] = sample["jpg"]
                keys_to_find.remove(sample["__key__"])
                
                if not keys_to_find:
                    break
            
            # Stop if we've scanned too many samples
            if scanned >= max_scan:
                logger.warning(f"Reached max scan limit ({max_scan}). Found {len(images)}/{len(keys[:max_samples])} images")
                break
        
        logger.info(f"Extracted {len(images)} images (scanned {scanned} samples)")
        return images
    
    def visualize_duplicate_pairs(self, output_dir: str, max_pairs: int = 10, max_scan: int = 20000):
        """Visualize duplicate pairs side by side"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Select pairs to visualize
        pairs_to_viz = self.duplicate_pairs[:max_pairs]
        
        if not pairs_to_viz:
            logger.warning("No duplicate pairs to visualize")
            return
        
        # Extract all needed keys
        all_keys = set()
        for pair in pairs_to_viz:
            all_keys.add(pair['removed'])
            all_keys.add(pair['kept'])
        
        # Extract images (limit scan to be efficient)
        images = self.extract_images(list(all_keys), max_samples=len(all_keys), max_scan=max_scan)
        
        # Create visualizations
        for i, pair in enumerate(pairs_to_viz):
            removed_key = pair['removed']
            kept_key = pair['kept']
            similarity = pair['similarity']
            cluster_id = pair['cluster_id']
            
            if removed_key in images and kept_key in images:
                # Create figure
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                
                # Display images
                ax1.imshow(images[removed_key])
                ax1.set_title(f"REMOVED\n{removed_key}", fontsize=10, color='red')
                ax1.axis('off')
                
                ax2.imshow(images[kept_key])
                ax2.set_title(f"KEPT\n{kept_key}", fontsize=10, color='green')
                ax2.axis('off')
                
                # Add similarity score
                fig.suptitle(f"Duplicate Pair {i+1} | Similarity: {similarity:.4f} | Cluster: {cluster_id}", 
                           fontsize=12, fontweight='bold')
                
                # Save figure
                output_file = output_path / f"duplicate_pair_{i+1:03d}.png"
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved visualization: {output_file}")
    
    def create_duplicate_grid(self, output_dir: str, max_groups: int = 5, max_scan: int = 10000):
        """Create a grid visualization of duplicate groups"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Group duplicates by kept sample
        groups = defaultdict(list)
        for pair in self.duplicate_pairs:
            groups[pair['kept']].append(pair)
        
        # Sort groups by size
        sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Visualize top groups
        for group_idx, (kept_key, duplicates) in enumerate(sorted_groups[:max_groups]):
            # Collect all keys in group
            all_keys = [kept_key] + [d['removed'] for d in duplicates[:8]]  # Limit to 9 total
            
            # Extract images (limit scan to be efficient)
            images = self.extract_images(all_keys, max_samples=len(all_keys), max_scan=max_scan)
            
            if kept_key in images:
                # Create grid
                n_images = min(len(images), 9)
                n_cols = min(3, n_images)
                n_rows = (n_images + n_cols - 1) // n_cols
                
                fig = plt.figure(figsize=(4*n_cols, 4*n_rows))
                
                # Plot kept image first (highlighted)
                ax = plt.subplot(n_rows, n_cols, 1)
                ax.imshow(images[kept_key])
                ax.set_title("KEPT (Original)", fontsize=10, color='green', fontweight='bold')
                ax.axis('off')
                ax.patch.set_edgecolor('green')
                ax.patch.set_linewidth(3)
                
                # Plot removed duplicates
                plot_idx = 2
                for dup in duplicates[:8]:
                    if dup['removed'] in images and plot_idx <= n_images:
                        ax = plt.subplot(n_rows, n_cols, plot_idx)
                        ax.imshow(images[dup['removed']])
                        ax.set_title(f"Removed (Sim: {dup['similarity']:.3f})", 
                                   fontsize=9, color='red')
                        ax.axis('off')
                        plot_idx += 1
                
                fig.suptitle(f"Duplicate Group {group_idx+1} - {len(duplicates)} duplicates removed", 
                           fontsize=14, fontweight='bold')
                
                # Save figure
                output_file = output_path / f"duplicate_group_{group_idx+1:02d}.png"
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved group visualization: {output_file}")
    
    def save_duplicate_report(self, output_dir: str):
        """Save detailed duplicate analysis report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame from duplicate pairs
        df = pd.DataFrame(self.duplicate_pairs)
        
        if not df.empty:
            # Save full duplicate pairs list
            df.to_csv(output_path / "duplicate_pairs.csv", index=False)
            
            # Create summary statistics
            summary = {
                'epsilon': self.epsilon,
                'total_samples': len(self.keys),
                'removed_samples': len(self.removed_keys),
                'duplicate_pairs_found': len(self.duplicate_pairs),
                'average_similarity': float(df['similarity'].mean()),
                'min_similarity': float(df['similarity'].min()),
                'max_similarity': float(df['similarity'].max()),
                'clusters_with_duplicates': int(df['cluster_id'].nunique())
            }
            
            # Group statistics
            kept_counts = df.groupby('kept').size().sort_values(ascending=False)
            summary['max_duplicates_per_image'] = int(kept_counts.iloc[0]) if len(kept_counts) > 0 else 0
            summary['images_with_multiple_duplicates'] = int((kept_counts > 1).sum())
            
            # Save summary
            with open(output_path / "duplicate_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save top duplicate groups
            top_groups = []
            for kept_key, count in kept_counts.head(10).items():
                group_duplicates = df[df['kept'] == kept_key]
                top_groups.append({
                    'kept_key': kept_key,
                    'num_duplicates': int(count),
                    'similarities': group_duplicates['similarity'].tolist(),
                    'removed_keys': group_duplicates['removed'].tolist()
                })
            
            with open(output_path / "top_duplicate_groups.json", 'w') as f:
                json.dump(top_groups, f, indent=2)
            
            logger.info(f"Saved duplicate analysis report to {output_path}")
            
            # Print summary
            print("\n" + "="*60)
            print("DUPLICATE ANALYSIS SUMMARY")
            print("="*60)
            print(f"Epsilon: {summary['epsilon']}")
            print(f"Total samples: {summary['total_samples']}")
            print(f"Removed samples: {summary['removed_samples']}")
            print(f"Duplicate pairs found: {summary['duplicate_pairs_found']}")
            print(f"Average similarity: {summary['average_similarity']:.4f}")
            print(f"Similarity range: [{summary['min_similarity']:.4f}, {summary['max_similarity']:.4f}]")
            print(f"Max duplicates per image: {summary['max_duplicates_per_image']}")
            print(f"Images with multiple duplicates: {summary['images_with_multiple_duplicates']}")
            print("="*60)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize SemDeDup duplicate pairs')
    parser.add_argument('--semdedup-output', type=str, 
                       default='test_output/semdedup_output',
                       help='Path to SemDeDup output directory')
    parser.add_argument('--dataset-path', type=str,
                       default='test_output/test_subset',
                       help='Path to WebDataset')
    parser.add_argument('--output-dir', type=str,
                       default='duplicate_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--epsilon', type=float, default=0.05,
                       help='Epsilon value to analyze')
    parser.add_argument('--max-pairs', type=int, default=100,
                       help='Maximum number of pairs to visualize')
    parser.add_argument('--max-groups', type=int, default=10,
                       help='Maximum number of duplicate groups to visualize')
    parser.add_argument('--max-scan', type=int, default=20000,
                       help='Maximum samples to scan when extracting images')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = DuplicateAnalyzer(
        semdedup_output_path=args.semdedup_output,
        dataset_path=args.dataset_path,
        epsilon=args.epsilon
    )
    
    # Run analysis
    logger.info("Starting duplicate analysis...")
    
    # Load results
    analyzer.load_semdedup_results()
    
    # Find duplicate pairs
    analyzer.find_duplicate_pairs()
    
    # Create visualizations
    logger.info("Creating visualizations...")
    analyzer.visualize_duplicate_pairs(args.output_dir, max_pairs=args.max_pairs, max_scan=args.max_scan)
    analyzer.create_duplicate_grid(args.output_dir, max_groups=args.max_groups, max_scan=args.max_scan)
    
    # Save report
    analyzer.save_duplicate_report(args.output_dir)
    
    logger.info(f"Analysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()