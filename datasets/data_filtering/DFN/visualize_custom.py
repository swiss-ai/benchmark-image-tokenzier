#!/usr/bin/env python3
"""
Customizable visualization for DFN scores - saves individual images with scores and captions
"""

import os
import sys
import json
import webdataset as wds
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io
import torch
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import argparse
from tqdm import tqdm

# Clean environment
if 'LD_LIBRARY_PATH' in os.environ:
    del os.environ['LD_LIBRARY_PATH']
if 'PYTHONPATH' in os.environ:
    del os.environ['PYTHONPATH']

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
from dfn_filter import DFNFilter


def create_individual_image_with_info(
    image: Image.Image,
    caption: str,
    score: float,
    sample_id: str,
    percentile_category: str,
    output_path: str
):
    """Create a single image with score and caption overlay"""
    
    # Image dimensions
    img_width = 512
    img_height = 512
    info_height = 150
    margin = 20
    
    # Total canvas size
    canvas_width = img_width + 2 * margin
    canvas_height = img_height + info_height + 2 * margin
    
    # Create canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to use better fonts
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        score_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        score_font = ImageFont.load_default()
    
    # Resize and paste image
    img_resized = image.resize((img_width, img_height), Image.Resampling.LANCZOS)
    canvas.paste(img_resized, (margin, margin))
    
    # Determine quality color based on score
    if score >= 5.0:
        border_color = '#2ecc71'  # Green
        quality = 'Excellent'
    elif score >= 4.0:
        border_color = '#3498db'  # Blue
        quality = 'Good'
    elif score >= 3.0:
        border_color = '#f39c12'  # Orange
        quality = 'Medium'
    elif score >= 2.0:
        border_color = '#e67e22'  # Dark Orange
        quality = 'Poor'
    else:
        border_color = '#e74c3c'  # Red
        quality = 'Very Poor'
    
    # Draw border around image
    draw.rectangle([margin-3, margin-3, margin+img_width+2, margin+img_height+2], 
                   outline=border_color, width=3)
    
    # Draw score badge on top-right corner of image
    badge_width = 100
    badge_height = 30
    badge_x = margin + img_width - badge_width - 10
    badge_y = margin + 10
    
    # Semi-transparent background for badge
    badge_img = Image.new('RGBA', (badge_width, badge_height), (0, 0, 0, 180))
    canvas.paste(badge_img, (badge_x, badge_y), badge_img)
    
    # Draw score on badge
    draw.text((badge_x + badge_width//2, badge_y + badge_height//2), 
             f"{score:.2f}", fill='white', font=score_font, anchor='mm')
    
    # Draw info section below image
    info_y = margin + img_height + 10
    
    # Sample ID and percentile category
    draw.text((margin, info_y), f"ID: {sample_id}", fill='black', font=text_font)
    draw.text((canvas_width - margin, info_y), f"{percentile_category}", 
             fill=border_color, font=title_font, anchor='ra')
    
    # Quality indicator
    draw.text((margin, info_y + 20), f"Quality: {quality}", 
             fill=border_color, font=score_font)
    
    # Caption (word-wrapped)
    caption_y = info_y + 45
    max_width = img_width
    
    # Word wrap caption
    words = caption.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        # Estimate text width (rough approximation)
        if len(test_line) > 70:  # Approximately 70 chars per line
            if current_line:
                lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    
    if current_line:
        lines.append(current_line)
    
    # Draw caption lines (max 3 lines)
    for i, line in enumerate(lines[:3]):
        if i == 2 and len(lines) > 3:
            line = line[:67] + "..."
        draw.text((margin, caption_y + i * 18), line, fill='#2c3e50', font=text_font)
    
    # Add score bar visualization
    bar_y = canvas_height - 30
    bar_width = img_width
    bar_height = 10
    
    # Background bar
    draw.rectangle([margin, bar_y, margin + bar_width, bar_y + bar_height], 
                   fill='#ecf0f1', outline='#bdc3c7')
    
    # Score bar (normalized to 0-7 range)
    normalized_score = max(0, min(7, score)) / 7.0
    filled_width = int(bar_width * normalized_score)
    draw.rectangle([margin, bar_y, margin + filled_width, bar_y + bar_height], 
                   fill=border_color)
    
    # Save image
    canvas.save(output_path, quality=95)
    return canvas


def collect_samples_by_percentile(
    dataset_path: str,
    model: DFNFilter,
    num_samples_to_process: int = 500,
    samples_per_category: int = 5
) -> Dict:
    """Collect samples categorized by percentiles"""
    
    print(f"Processing {num_samples_to_process} samples to collect examples...")
    
    # First, collect all scores to determine percentiles
    all_samples = []
    
    # Process first shard or multiple shards
    shard_files = sorted(Path(dataset_path).glob("*.tar"))[:3]  # Use first 3 shards for better distribution
    
    processed = 0
    for shard_file in shard_files:
        if processed >= num_samples_to_process:
            break
            
        print(f"Processing shard: {shard_file.name}")
        dataset = wds.WebDataset(str(shard_file), shardshuffle=False).decode()
        
        for sample in tqdm(dataset, desc="Collecting samples"):
            if processed >= num_samples_to_process:
                break
            
            try:
                # Extract image
                if 'jpg' in sample:
                    img_data = sample['jpg']
                elif 'png' in sample:
                    img_data = sample['png']
                else:
                    continue
                
                # Convert to PIL Image
                if isinstance(img_data, bytes):
                    img = Image.open(io.BytesIO(img_data)).convert('RGB')
                else:
                    img = Image.fromarray(img_data).convert('RGB')
                
                # Extract caption
                caption = ""
                if 'json' in sample:
                    json_data = sample['json']
                    if isinstance(json_data, bytes):
                        json_data = json_data.decode('utf-8')
                    if isinstance(json_data, str):
                        json_data = json.loads(json_data)
                    
                    # Get GPT response
                    if 'conversations' in json_data:
                        for conv in json_data['conversations']:
                            if conv.get('from') in ['gpt', 'assistant']:
                                caption = conv.get('value', '')
                                break
                
                if not caption:
                    continue
                
                # Compute DFN score
                score = model.compute_filtering_score(img, caption)
                
                all_samples.append({
                    'image': img,
                    'caption': caption,
                    'score': score,
                    'sample_id': sample.get('__key__', f'sample_{processed}')
                })
                
                processed += 1
                
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
    
    print(f"\nProcessed {len(all_samples)} samples total")
    
    if not all_samples:
        print("No samples collected!")
        return {}
    
    # Calculate percentiles
    scores = [s['score'] for s in all_samples]
    percentiles = {
        'top_10': np.percentile(scores, 90),
        'top_25': np.percentile(scores, 75),
        'median': np.percentile(scores, 50),
        'bottom_25': np.percentile(scores, 25),
        'bottom_10': np.percentile(scores, 10)
    }
    
    print("\nPercentile thresholds:")
    for name, value in percentiles.items():
        print(f"  {name}: {value:.4f}")
    
    # Categorize samples
    categories = {
        'top_10_percent': {'range': f'Top 10% (>{percentiles["top_10"]:.2f})', 'samples': []},
        'top_10_25_percent': {'range': f'Top 10-25% ({percentiles["top_25"]:.2f}-{percentiles["top_10"]:.2f})', 'samples': []},
        'top_25_50_percent': {'range': f'Top 25-50% ({percentiles["median"]:.2f}-{percentiles["top_25"]:.2f})', 'samples': []},
        'bottom_25_50_percent': {'range': f'Bottom 25-50% ({percentiles["bottom_25"]:.2f}-{percentiles["median"]:.2f})', 'samples': []},
        'bottom_10_25_percent': {'range': f'Bottom 10-25% ({percentiles["bottom_10"]:.2f}-{percentiles["bottom_25"]:.2f})', 'samples': []},
        'bottom_10_percent': {'range': f'Bottom 10% (<{percentiles["bottom_10"]:.2f})', 'samples': []},
    }
    
    # Sort samples by score
    all_samples.sort(key=lambda x: x['score'], reverse=True)
    
    # Assign samples to categories
    for sample in all_samples:
        score = sample['score']
        
        if score > percentiles['top_10']:
            category = 'top_10_percent'
        elif score > percentiles['top_25']:
            category = 'top_10_25_percent'
        elif score > percentiles['median']:
            category = 'top_25_50_percent'
        elif score > percentiles['bottom_25']:
            category = 'bottom_25_50_percent'
        elif score > percentiles['bottom_10']:
            category = 'bottom_10_25_percent'
        else:
            category = 'bottom_10_percent'
        
        if len(categories[category]['samples']) < samples_per_category:
            categories[category]['samples'].append(sample)
    
    print("\nSamples collected per category:")
    for cat_name, cat_info in categories.items():
        print(f"  {cat_name}: {len(cat_info['samples'])} samples")
    
    return categories, percentiles


def main():
    """Main function with customizable parameters"""
    
    parser = argparse.ArgumentParser(description='Customizable DFN score visualization')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples to process for finding examples')
    parser.add_argument('--samples_per_category', type=int, default=5,
                        help='Number of example images to save per percentile category')
    parser.add_argument('--output_dir', type=str, default='./custom_visualization',
                        help='Output directory for visualizations')
    parser.add_argument('--dataset_path', type=str, 
                        default='/capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain/webdataset_output',
                        help='Path to webdataset')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DFN Score Custom Visualization")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Processing {args.num_samples} samples")
    print(f"  - Saving {args.samples_per_category} examples per category")
    print(f"  - Output directory: {args.output_dir}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize model
    print("Initializing DFN model...")
    model = DFNFilter(model_name='hf-hub:apple/DFN-public', device='cuda:0', batch_size=8)
    
    # Collect samples
    categories, percentiles = collect_samples_by_percentile(
        args.dataset_path, 
        model,
        args.num_samples,
        args.samples_per_category
    )
    
    if not categories:
        print("No samples collected!")
        return
    
    # Create visualizations
    print("\nCreating individual visualizations...")
    
    # Create subdirectories for each category
    for cat_name in categories.keys():
        (output_dir / cat_name).mkdir(exist_ok=True)
    
    # Save individual images
    all_saved_files = []
    
    for cat_name, cat_info in categories.items():
        print(f"\nProcessing {cat_name}...")
        
        for idx, sample in enumerate(cat_info['samples']):
            filename = f"{cat_name}_{idx+1:02d}_score{sample['score']:.2f}.png"
            filepath = output_dir / cat_name / filename
            
            create_individual_image_with_info(
                sample['image'],
                sample['caption'],
                sample['score'],
                sample['sample_id'],
                cat_info['range'],
                str(filepath)
            )
            
            all_saved_files.append({
                'category': cat_name,
                'file': str(filepath),
                'score': sample['score'],
                'caption': sample['caption'][:100]
            })
            
            print(f"  Saved: {filename}")
    
    # Save metadata
    metadata = {
        'configuration': {
            'num_samples_processed': args.num_samples,
            'samples_per_category': args.samples_per_category
        },
        'percentiles': {k: float(v) for k, v in percentiles.items()},
        'categories': {
            cat_name: {
                'range': cat_info['range'],
                'num_samples': len(cat_info['samples']),
                'scores': [s['score'] for s in cat_info['samples']]
            }
            for cat_name, cat_info in categories.items()
        },
        'files': all_saved_files
    }
    
    metadata_file = output_dir / 'visualization_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to {metadata_file}")
    
    # Create summary image showing one example from each category
    print("\nCreating summary image...")
    summary_images = []
    summary_captions = []
    summary_scores = []
    
    for cat_name, cat_info in categories.items():
        if cat_info['samples']:
            sample = cat_info['samples'][0]  # Take first sample from each category
            summary_images.append(sample['image'])
            summary_captions.append(f"{cat_info['range']}\n{sample['caption'][:50]}...")
            summary_scores.append(sample['score'])
    
    if summary_images:
        # Create a grid of examples
        from visualize_scores import create_visualization_grid
        create_visualization_grid(
            summary_images, summary_captions, summary_scores,
            str(output_dir / 'summary_all_categories.png'),
            title="DFN Score Distribution - One Example Per Category"
        )
    
    print("\n" + "=" * 80)
    print("Visualization complete!")
    print(f"Files saved to: {output_dir}/")
    print(f"  - {len(categories)} category directories")
    print(f"  - {len(all_saved_files)} individual images")
    print(f"  - visualization_metadata.json")
    print(f"  - summary_all_categories.png")
    print("=" * 80)


if __name__ == "__main__":
    main()