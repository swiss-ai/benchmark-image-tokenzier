#!/usr/bin/env python3
"""
Visualize DFN scores with example images and captions
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
from typing import List, Tuple

# Clean environment
if 'LD_LIBRARY_PATH' in os.environ:
    del os.environ['LD_LIBRARY_PATH']
if 'PYTHONPATH' in os.environ:
    del os.environ['PYTHONPATH']

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
from dfn_filter import DFNFilter


def create_visualization_grid(
    images: List[Image.Image],
    captions: List[str],
    scores: List[float],
    output_path: str,
    title: str = "DFN Score Examples"
):
    """Create a grid visualization of images with captions and scores"""
    
    # Calculate grid dimensions
    n_samples = len(images)
    cols = min(3, n_samples)
    rows = (n_samples + cols - 1) // cols
    
    # Image dimensions
    img_size = 256
    padding = 20
    text_height = 80
    
    # Create canvas
    canvas_width = cols * (img_size + padding) + padding
    canvas_height = rows * (img_size + text_height + padding) + padding + 40
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to use a better font, fallback to default
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        score_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        score_font = ImageFont.load_default()
    
    # Draw title
    draw.text((canvas_width // 2, 20), title, fill='black', font=title_font, anchor='mt')
    
    # Draw images with captions and scores
    for idx, (img, caption, score) in enumerate(zip(images, captions, scores)):
        row = idx // cols
        col = idx % cols
        
        # Calculate position
        x = col * (img_size + padding) + padding
        y = row * (img_size + text_height + padding) + padding + 40
        
        # Resize and paste image
        img_resized = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        canvas.paste(img_resized, (x, y))
        
        # Draw border around image (color based on score)
        if score >= 5.0:
            border_color = 'green'
            quality = 'High'
        elif score >= 3.5:
            border_color = 'orange'
            quality = 'Medium'
        else:
            border_color = 'red'
            quality = 'Low'
        
        draw.rectangle([x-2, y-2, x+img_size+1, y+img_size+1], outline=border_color, width=3)
        
        # Draw score
        score_text = f"Score: {score:.2f} ({quality})"
        draw.text((x + img_size // 2, y + img_size + 5), score_text, 
                 fill=border_color, font=score_font, anchor='mt')
        
        # Draw caption (truncated)
        caption_lines = []
        words = caption.split()
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) > 35:  # Max chars per line
                if current_line:
                    caption_lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            caption_lines.append(current_line)
        
        # Draw up to 3 lines of caption
        for i, line in enumerate(caption_lines[:3]):
            if i == 2 and len(caption_lines) > 3:
                line = line[:32] + "..."
            draw.text((x + img_size // 2, y + img_size + 25 + i * 15), 
                     line, fill='black', font=text_font, anchor='mt')
    
    # Save visualization
    canvas.save(output_path)
    print(f"Saved visualization to {output_path}")


def collect_samples_by_score_range(
    dataset_path: str,
    model: DFNFilter,
    num_samples_per_category: int = 3
) -> dict:
    """Collect samples from different score ranges"""
    
    categories = {
        'very_high': {'min': 5.0, 'max': float('inf'), 'samples': []},
        'high': {'min': 4.0, 'max': 5.0, 'samples': []},
        'medium': {'min': 3.0, 'max': 4.0, 'samples': []},
        'low': {'min': 2.0, 'max': 3.0, 'samples': []},
        'very_low': {'min': float('-inf'), 'max': 2.0, 'samples': []},
    }
    
    # Process first shard
    shard_file = f"{dataset_path}/llava558k-000000-000000000.tar"
    dataset = wds.WebDataset(shard_file, shardshuffle=False).decode()
    
    processed = 0
    max_samples = 500  # Process up to 500 samples to find examples
    
    print(f"Collecting samples from different score ranges...")
    
    for sample in dataset:
        if processed >= max_samples:
            break
            
        # Check if we have enough samples in all categories
        all_complete = all(
            len(cat['samples']) >= num_samples_per_category 
            for cat in categories.values()
        )
        if all_complete:
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
            
            # Add to appropriate category
            for cat_name, cat_info in categories.items():
                if cat_info['min'] <= score < cat_info['max']:
                    if len(cat_info['samples']) < num_samples_per_category:
                        cat_info['samples'].append({
                            'image': img,
                            'caption': caption,
                            'score': score,
                            'sample_id': sample.get('__key__', f'sample_{processed}')
                        })
                    break
            
            processed += 1
            
            if processed % 50 == 0:
                print(f"  Processed {processed} samples...")
                for cat_name, cat_info in categories.items():
                    print(f"    {cat_name}: {len(cat_info['samples'])} samples")
                    
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    print(f"\nCollected samples summary:")
    for cat_name, cat_info in categories.items():
        print(f"  {cat_name}: {len(cat_info['samples'])} samples")
    
    return categories


def save_examples_to_json(categories: dict, output_file: str):
    """Save example data to JSON for reference"""
    
    examples = {}
    for cat_name, cat_info in categories.items():
        examples[cat_name] = []
        for sample in cat_info['samples']:
            examples[cat_name].append({
                'sample_id': sample['sample_id'],
                'score': float(sample['score']),
                'caption': sample['caption'][:200],  # Truncate long captions
                'caption_length': len(sample['caption'])
            })
    
    with open(output_file, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"Saved examples data to {output_file}")


def create_score_distribution_plot(scores: List[float], output_path: str):
    """Create a simple histogram visualization of score distribution"""
    
    # Create histogram
    hist, bins = np.histogram(scores, bins=20)
    
    # Create image
    width, height = 800, 400
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw title
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    draw.text((width // 2, 20), "DFN Score Distribution", fill='black', font=font, anchor='mt')
    
    # Draw histogram
    margin = 50
    plot_width = width - 2 * margin
    plot_height = height - 100
    
    max_count = max(hist)
    bar_width = plot_width / len(hist)
    
    for i, count in enumerate(hist):
        bar_height = int(count / max_count * plot_height) if max_count > 0 else 0
        x1 = margin + i * bar_width
        x2 = x1 + bar_width - 2
        y1 = height - margin - bar_height
        y2 = height - margin
        
        # Color based on score range
        score = (bins[i] + bins[i+1]) / 2
        if score >= 5.0:
            color = 'green'
        elif score >= 3.5:
            color = 'orange'
        else:
            color = 'red'
        
        draw.rectangle([x1, y1, x2, y2], fill=color)
    
    # Draw axes
    draw.line([(margin, height - margin), (width - margin, height - margin)], fill='black', width=2)
    draw.line([(margin, height - margin), (margin, 50)], fill='black', width=2)
    
    # Draw labels
    draw.text((width // 2, height - 20), "DFN Score", fill='black', font=font, anchor='mt')
    draw.text((20, height // 2), "Count", fill='black', font=font, anchor='mm')
    
    # Save plot
    img.save(output_path)
    print(f"Saved distribution plot to {output_path}")


def main():
    """Main visualization function"""
    
    print("=" * 80)
    print("DFN Score Visualization")
    print("=" * 80)
    
    # Setup paths
    dataset_path = "/capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain/webdataset_output"
    output_dir = Path("./visualization_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize model
    print("\nInitializing DFN model...")
    model = DFNFilter(model_name='hf-hub:apple/DFN-public', device='cuda:0', batch_size=8)
    
    # Collect samples
    print("\nCollecting samples from different score ranges...")
    categories = collect_samples_by_score_range(dataset_path, model, num_samples_per_category=3)
    
    # Create visualizations for each category
    for cat_name, cat_info in categories.items():
        if cat_info['samples']:
            images = [s['image'] for s in cat_info['samples']]
            captions = [s['caption'] for s in cat_info['samples']]
            scores = [s['score'] for s in cat_info['samples']]
            
            output_path = output_dir / f"dfn_examples_{cat_name}.png"
            score_range = f"{cat_info['min']:.1f} - {cat_info['max']:.1f}" if cat_info['max'] != float('inf') else f"> {cat_info['min']:.1f}"
            
            create_visualization_grid(
                images, captions, scores,
                str(output_path),
                title=f"DFN Scores: {cat_name.replace('_', ' ').title()} Quality (Score {score_range})"
            )
    
    # Create combined visualization
    print("\nCreating combined visualization...")
    all_images = []
    all_captions = []
    all_scores = []
    
    # Take 2 samples from each category for combined view
    for cat_name, cat_info in categories.items():
        for sample in cat_info['samples'][:2]:
            all_images.append(sample['image'])
            all_captions.append(sample['caption'])
            all_scores.append(sample['score'])
    
    if all_images:
        create_visualization_grid(
            all_images, all_captions, all_scores,
            str(output_dir / "dfn_examples_combined.png"),
            title="DFN Score Examples - All Categories"
        )
    
    # Save examples to JSON
    save_examples_to_json(categories, str(output_dir / "dfn_examples_data.json"))
    
    # Create score distribution from all processed scores
    all_scores_list = [s['score'] for cat in categories.values() for s in cat['samples']]
    if all_scores_list:
        create_score_distribution_plot(
            all_scores_list,
            str(output_dir / "dfn_score_distribution.png")
        )
    
    # Print summary statistics
    if all_scores_list:
        print("\n" + "=" * 80)
        print("Score Statistics from Samples:")
        print(f"  Total samples: {len(all_scores_list)}")
        print(f"  Mean score: {np.mean(all_scores_list):.4f}")
        print(f"  Std deviation: {np.std(all_scores_list):.4f}")
        print(f"  Min score: {np.min(all_scores_list):.4f}")
        print(f"  Max score: {np.max(all_scores_list):.4f}")
        print(f"  Median score: {np.median(all_scores_list):.4f}")
        print("=" * 80)
    
    print(f"\nAll visualizations saved to {output_dir}/")
    print("Files created:")
    for file in sorted(output_dir.glob("*")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()