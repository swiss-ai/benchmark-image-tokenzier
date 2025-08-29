#!/usr/bin/env python3
"""
Save sample images with scores and captions for visual inspection
"""

import os
import sys
import json
import argparse
import tarfile
import io
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import webdataset as wds
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def get_image_from_tar(tar_dir: str, sample_id: str) -> Image.Image:
    """Extract image from tar files by sample ID"""
    
    # List all tar files
    tar_files = sorted(Path(tar_dir).glob("*.tar"))
    
    for tar_path in tar_files:
        try:
            with tarfile.open(tar_path, 'r') as tar:
                # Look for the image file with this ID
                for member in tar.getmembers():
                    # Check for .jpg or .png with matching ID
                    if sample_id in member.name and (member.name.endswith('.jpg') or member.name.endswith('.png')):
                        img_data = tar.extractfile(member).read()
                        img = Image.open(io.BytesIO(img_data))
                        return img.convert('RGB')
        except Exception as e:
            continue
    
    return None

def create_annotated_image(img: Image.Image, caption: str, scores: Dict[str, int], 
                          output_size: Tuple[int, int] = (1000, 1200)) -> Image.Image:
    """Create an annotated image with scores and caption - improved text clarity"""
    
    # Resize image to fit in output while maintaining aspect ratio
    img_width, img_height = output_size[0] - 100, int(output_size[1] * 0.5)
    img.thumbnail((img_width, img_height), Image.Resampling.LANCZOS)
    
    # Create new image with pure white background for better contrast
    annotated = Image.new('RGB', output_size, 'white')
    
    # Enable better text rendering
    draw = ImageDraw.Draw(annotated)
    
    # Draw subtle frame for image
    img_x = (output_size[0] - img.width) // 2
    img_y = 40
    
    # Add subtle shadow effect
    shadow_color = '#E8E8E8'
    for offset in range(3, 0, -1):
        draw.rectangle([img_x - offset, img_y - offset, 
                       img_x + img.width + offset, img_y + img.height + offset], 
                       outline=shadow_color, width=1)
    
    # Paste the image
    annotated.paste(img, (img_x, img_y))
    
    # Load fonts with better sizes and spacing
    # Using larger font sizes and different weights for better clarity
    try:
        # Primary font choice - DejaVu (usually clearest)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        score_label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        score_value_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        caption_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
        percent_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        try:
            # Fallback to Liberation fonts
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 30)
            score_label_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 22)
            score_value_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 24)
            caption_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 19)
            percent_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
        except:
            # Default fonts as last resort
            title_font = ImageFont.load_default()
            score_label_font = ImageFont.load_default()
            score_value_font = ImageFont.load_default()
            caption_font = ImageFont.load_default()
            percent_font = ImageFont.load_default()
    
    # Starting Y position for text (below image with more spacing)
    text_y = img_y + img.height + 60
    
    # Draw scores section with subtle background
    scores_bg_y = text_y - 15
    scores_height = 240  # Increased height for better spacing
    
    # Draw subtle background with rounded corners effect
    draw.rectangle([40, scores_bg_y, output_size[0] - 40, scores_bg_y + scores_height], 
                   fill='#FAFAFA', outline='#E0E0E0', width=1)
    
    # Draw scores title with better spacing
    draw.text((60, text_y), "MLM Filter Scores", font=title_font, fill='#1A1A1A')
    text_y += 50  # More space after title
    
    # Define metric display with better formatting
    metric_info = [
        ('image_text_matching', 'Image-Text Matching'),
        ('object_detail_fulfillment', 'Object Detail Fulfillment'),
        ('caption_text_quality', 'Caption Text Quality'),
        ('semantic_understanding', 'Semantic Understanding')
    ]
    
    # Single column layout for better readability
    row_y = text_y
    left_margin = 60
    
    for i, (metric, display_name) in enumerate(metric_info):
        score_key = f"{metric}_score"
        if score_key in scores:
            score = scores[score_key]
            
            # Color code with gradient
            if score >= 80:
                score_color = '#1B5E20'  # Dark green
                bar_color = '#4CAF50'    # Green
            elif score >= 60:
                score_color = '#F57C00'  # Dark orange
                bar_color = '#FFA726'    # Orange
            elif score >= 40:
                score_color = '#E65100'  # Darker orange
                bar_color = '#FF9800'    # Orange
            else:
                score_color = '#B71C1C'  # Dark red
                bar_color = '#EF5350'    # Red
            
            # Draw metric name (left aligned with better spacing)
            metric_text = display_name
            draw.text((left_margin, row_y), metric_text, font=score_label_font, fill='#2C2C2C')
            
            # Draw score value (right aligned with padding)
            score_text = f"{score}"
            # Calculate actual text width for proper right alignment
            try:
                bbox = draw.textbbox((0, 0), score_text, font=score_value_font)
                text_width = bbox[2] - bbox[0]
            except:
                text_width = len(score_text) * 12
            
            score_x = output_size[0] - 100 - text_width
            draw.text((score_x, row_y - 2), score_text, font=score_value_font, fill=score_color)
            
            # Draw progress bar with better positioning
            bar_x = left_margin
            bar_y = row_y + 30  # More space between text and bar
            bar_width = output_size[0] - (2 * left_margin) - 20
            bar_height = 10  # Slightly taller for visibility
            
            # Background bar
            draw.rectangle([bar_x, bar_y, bar_x + bar_width, bar_y + bar_height], 
                          fill='#E0E0E0', outline=None)
            
            # Score bar
            score_bar_width = int(bar_width * score / 100)
            if score_bar_width > 0:
                draw.rectangle([bar_x, bar_y, bar_x + score_bar_width, bar_y + bar_height], 
                              fill=bar_color, outline=None)
            
            # Add score percentage label next to the bar
            percentage_text = f"{score}%"
            try:
                bbox = draw.textbbox((0, 0), percentage_text, font=percent_font)
                text_width = bbox[2] - bbox[0]
            except:
                text_width = len(percentage_text) * 8
            
            # Position percentage at the end of the bar
            percent_x = bar_x + bar_width + 10
            draw.text((percent_x, bar_y - 2), percentage_text, font=percent_font, fill='#757575')
            
            # Move to next row with more spacing
            row_y += 55  # Increased spacing between metrics
    
    # Draw caption section with better spacing
    text_y = scores_bg_y + scores_height + 30
    caption_bg_y = text_y - 15
    
    # Calculate caption height needed
    caption_height = min(280, 100 + len(caption) // 8)
    draw.rectangle([40, caption_bg_y, output_size[0] - 40, caption_bg_y + caption_height], 
                   fill='#FAFAFA', outline='#E0E0E0', width=1)
    
    draw.text((60, text_y), "Caption", font=title_font, fill='#1A1A1A')
    text_y += 45  # More space after caption title
    
    # Better text wrapping with improved spacing
    max_width = output_size[0] - 140  # More padding on sides
    words = caption.split()
    lines = []
    current_line = []
    
    for word in words:
        if current_line:
            test_line = ' '.join(current_line) + ' ' + word  # Add space
        else:
            test_line = word
            
        # Use actual font to measure text width
        try:
            bbox = draw.textbbox((0, 0), test_line, font=caption_font)
            text_width = bbox[2] - bbox[0]
        except:
            # Fallback for older Pillow versions
            text_width = len(test_line) * 10
        
        if text_width > max_width:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                # Word is too long, split it
                lines.append(word[:45] + '...')
                current_line = []
        else:
            current_line.append(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Draw wrapped caption with better line height
    line_spacing = 26  # Increased line spacing for readability
    max_lines = 7
    for i, line in enumerate(lines[:max_lines]):
        # Add slight indentation for caption text
        draw.text((70, text_y), line, font=caption_font, fill='#1A1A1A')
        text_y += line_spacing
    
    if len(lines) > max_lines:
        draw.text((70, text_y), "... (truncated)", font=caption_font, fill='#757575')
    
    return annotated

def save_sample_images(parquet_path: str, tar_dir: str, output_dir: Path):
    """Load samples and save images with annotations"""
    
    # Load the parquet file with all results
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} samples from parquet")
    
    # Load the sample JSON files
    sample_dir = output_dir / 'sample_examples'
    
    metrics = ['image_text_matching', 'object_detail_fulfillment', 
               'caption_text_quality', 'semantic_understanding']
    
    for metric in metrics:
        metric_dir = sample_dir / metric
        
        # Create image directories (with parents)
        high_img_dir = metric_dir / 'high_scoring_images'
        low_img_dir = metric_dir / 'low_scoring_images'
        high_img_dir.mkdir(parents=True, exist_ok=True)
        low_img_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing {metric}...")
        
        # Load high scoring samples
        high_json = metric_dir / 'high_scoring_samples.json'
        if high_json.exists():
            with open(high_json, 'r') as f:
                high_data = json.load(f)
            
            for i, sample in enumerate(high_data['samples'][:10]):
                sample_id = sample['sample_id']
                caption = sample['caption']
                
                # Get image from tar
                img = get_image_from_tar(tar_dir, sample_id)
                if img:
                    # Create annotated image
                    annotated = create_annotated_image(img, caption, sample)
                    
                    # Save image
                    output_path = high_img_dir / f"high_{i+1:02d}_{sample_id}.jpg"
                    annotated.save(output_path, quality=90)
                    print(f"  Saved high sample {i+1}: {output_path.name}")
                else:
                    print(f"  Warning: Could not find image for {sample_id}")
        
        # Load low scoring samples
        low_json = metric_dir / 'low_scoring_samples.json'
        if low_json.exists():
            with open(low_json, 'r') as f:
                low_data = json.load(f)
            
            for i, sample in enumerate(low_data['samples'][:10]):
                sample_id = sample['sample_id']
                caption = sample['caption']
                
                # Get image from tar
                img = get_image_from_tar(tar_dir, sample_id)
                if img:
                    # Create annotated image
                    annotated = create_annotated_image(img, caption, sample)
                    
                    # Save image
                    output_path = low_img_dir / f"low_{i+1:02d}_{sample_id}.jpg"
                    annotated.save(output_path, quality=90)
                    print(f"  Saved low sample {i+1}: {output_path.name}")
                else:
                    print(f"  Warning: Could not find image for {sample_id}")

def create_html_gallery(output_dir: Path):
    """Create HTML gallery for easy viewing"""
    
    sample_dir = output_dir / 'sample_examples'
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>MLM Filter Sample Gallery</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        h2 {
            color: #555;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
        }
        h3 {
            color: #666;
            margin-top: 30px;
        }
        .metric-section {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .image-card {
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            overflow: hidden;
        }
        .image-card img {
            width: 100%;
            height: auto;
            display: block;
        }
        .high-scoring {
            border-left: 4px solid #4CAF50;
        }
        .low-scoring {
            border-left: 4px solid #f44336;
        }
    </style>
</head>
<body>
    <h1>MLM Filter Sample Gallery</h1>
"""
    
    metrics = ['image_text_matching', 'object_detail_fulfillment', 
               'caption_text_quality', 'semantic_understanding']
    
    for metric in metrics:
        metric_name = metric.replace('_', ' ').title()
        html_content += f"""
    <div class="metric-section">
        <h2>{metric_name}</h2>
"""
        
        # High scoring samples
        high_img_dir = sample_dir / metric / 'high_scoring_images'
        if high_img_dir.exists():
            high_images = sorted(high_img_dir.glob("*.jpg"))
            if high_images:
                html_content += """
        <h3>High Scoring Samples</h3>
        <div class="image-grid">
"""
                for img_path in high_images[:5]:
                    rel_path = img_path.relative_to(output_dir)
                    html_content += f"""
            <div class="image-card high-scoring">
                <img src="{rel_path}" alt="{img_path.name}">
            </div>
"""
                html_content += """
        </div>
"""
        
        # Low scoring samples
        low_img_dir = sample_dir / metric / 'low_scoring_images'
        if low_img_dir.exists():
            low_images = sorted(low_img_dir.glob("*.jpg"))
            if low_images:
                html_content += """
        <h3>Low Scoring Samples</h3>
        <div class="image-grid">
"""
                for img_path in low_images[:5]:
                    rel_path = img_path.relative_to(output_dir)
                    html_content += f"""
            <div class="image-card low-scoring">
                <img src="{rel_path}" alt="{img_path.name}">
            </div>
"""
                html_content += """
        </div>
"""
        
        html_content += """
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    # Save HTML file
    html_path = output_dir / 'sample_gallery.html'
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"\nHTML gallery saved to: {html_path}")

def main():
    parser = argparse.ArgumentParser(description='Save sample images with annotations')
    parser.add_argument('--parquet', type=str, 
                       default='./mlm_scores_final_output/mlm_scores_final.parquet',
                       help='Path to results parquet file')
    parser.add_argument('--tar_dir', type=str,
                       default='/capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain/webdataset_output',
                       help='Directory containing tar files')
    parser.add_argument('--output_dir', type=str, 
                       default='./mlm_analysis_output',
                       help='Output directory for analysis')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("="*70)
    print("SAVING SAMPLE IMAGES WITH ANNOTATIONS")
    print("="*70)
    
    # Save sample images
    save_sample_images(args.parquet, args.tar_dir, output_dir)
    
    # Create HTML gallery
    create_html_gallery(output_dir)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Images saved to: {output_dir}/sample_examples/*/{{high,low}}_scoring_images/")
    print(f"View gallery at: {output_dir}/sample_gallery.html")

if __name__ == "__main__":
    main()