#!/usr/bin/env python3
"""
Save top and bottom 10% of images based on mean of 4 MLM scores
with cleaner visualization layout
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
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def get_image_from_tar(tar_dir: str, sample_id: str) -> Optional[Image.Image]:
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

def create_clean_annotated_image(img: Image.Image, caption: str, scores: Dict[str, float], 
                                mean_score: float, output_size: Tuple[int, int] = (800, 1000)) -> Image.Image:
    """Create a clean annotated image with image on top and centered text below"""
    
    # Resize image to fit width while maintaining aspect ratio
    img_width = output_size[0] - 40  # 20px padding on each side
    img_height = int(output_size[1] * 0.55)  # 55% of height for image
    img.thumbnail((img_width, img_height), Image.Resampling.LANCZOS)
    
    # Create new image with white background
    annotated = Image.new('RGB', output_size, 'white')
    draw = ImageDraw.Draw(annotated)
    
    # Center and paste the image at top with small margin
    img_x = (output_size[0] - img.width) // 2
    img_y = 20
    
    # Add subtle border around image
    border_padding = 2
    draw.rectangle([img_x - border_padding, img_y - border_padding, 
                   img_x + img.width + border_padding, img_y + img.height + border_padding], 
                   outline='#CCCCCC', width=1)
    
    # Paste the image
    annotated.paste(img, (img_x, img_y))
    
    # Load fonts for better readability
    try:
        # Try DejaVu fonts first (usually clearest)
        header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        score_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        caption_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        mean_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        try:
            # Fallback to Liberation fonts
            header_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 20)
            score_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
            caption_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 14)
            mean_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 18)
        except:
            # Default fonts as last resort
            header_font = ImageFont.load_default()
            score_font = ImageFont.load_default()
            caption_font = ImageFont.load_default()
            mean_font = ImageFont.load_default()
    
    # Calculate text area starting position (below image)
    text_start_y = img_y + img.height + 30
    text_x = 40  # Left margin for text
    line_height = 24
    
    # Draw mean score prominently at top
    mean_color = '#2E7D32' if mean_score >= 60 else '#D32F2F' if mean_score < 40 else '#F57C00'
    mean_text = f"Mean Score: {mean_score:.1f}"
    
    # Center the mean score
    try:
        bbox = draw.textbbox((0, 0), mean_text, font=mean_font)
        mean_width = bbox[2] - bbox[0]
    except:
        mean_width = len(mean_text) * 10
    
    mean_x = (output_size[0] - mean_width) // 2
    draw.text((mean_x, text_start_y), mean_text, font=mean_font, fill=mean_color)
    
    # Draw separator line
    text_start_y += 35
    draw.line([(40, text_start_y), (output_size[0] - 40, text_start_y)], fill='#E0E0E0', width=1)
    text_start_y += 15
    
    # Draw individual scores in a compact 2x2 grid
    score_labels = [
        ('Image-Text Match', 'image_text_matching_score'),
        ('Object Detail', 'object_detail_fulfillment_score'),
        ('Caption Quality', 'caption_text_quality_score'),
        ('Semantic Understanding', 'semantic_understanding_score')
    ]
    
    # Calculate positions for 2x2 grid
    col_width = (output_size[0] - 80) // 2
    row_height = 30
    
    for i, (label, score_key) in enumerate(score_labels):
        row = i // 2
        col = i % 2
        
        x = 40 + col * col_width
        y = text_start_y + row * row_height
        
        if score_key in scores:
            score_val = scores[score_key]
            score_color = '#2E7D32' if score_val >= 80 else '#D32F2F' if score_val < 40 else '#F57C00'
            
            # Draw label and score
            text = f"{label}: {score_val}"
            draw.text((x, y), text, font=score_font, fill='#333333')
    
    # Draw separator before caption
    text_start_y += 70
    draw.line([(40, text_start_y), (output_size[0] - 40, text_start_y)], fill='#E0E0E0', width=1)
    text_start_y += 15
    
    # Draw caption header
    draw.text((40, text_start_y), "Caption:", font=header_font, fill='#333333')
    text_start_y += 30
    
    # Wrap and draw caption text
    max_width = output_size[0] - 80
    words = caption.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word]) if current_line else word
        
        try:
            bbox = draw.textbbox((0, 0), test_line, font=caption_font)
            text_width = bbox[2] - bbox[0]
        except:
            text_width = len(test_line) * 8
        
        if text_width > max_width:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                lines.append(word[:50] + '...')
                current_line = []
        else:
            current_line.append(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Draw caption lines (max 6 lines to fit)
    max_lines = 6
    for i, line in enumerate(lines[:max_lines]):
        draw.text((40, text_start_y + i * 20), line, font=caption_font, fill='#555555')
    
    if len(lines) > max_lines:
        draw.text((40, text_start_y + max_lines * 20), "...", font=caption_font, fill='#888888')
    
    return annotated

def save_mean_based_samples(parquet_path: str, tar_dir: str, output_dir: Path, 
                           num_samples: int = 20, percentile: float = 10.0):
    """Save top and bottom percentile of images based on mean score"""
    
    # Load the parquet file
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} samples from parquet")
    
    # Calculate mean score for each sample
    score_columns = ['image_text_matching_score', 'object_detail_fulfillment_score',
                    'caption_text_quality_score', 'semantic_understanding_score']
    
    # Check if all score columns exist
    missing_cols = [col for col in score_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}")
        score_columns = [col for col in score_columns if col in df.columns]
    
    # Calculate mean score
    df['mean_score'] = df[score_columns].mean(axis=1)
    
    # Remove samples with invalid scores
    valid_df = df[df['mean_score'] >= 0].copy()
    print(f"Found {len(valid_df)} samples with valid scores")
    
    # Calculate percentile thresholds
    bottom_threshold = np.percentile(valid_df['mean_score'], percentile)
    top_threshold = np.percentile(valid_df['mean_score'], 100 - percentile)
    
    print(f"\nScore thresholds:")
    print(f"  Bottom {percentile}%: <= {bottom_threshold:.2f}")
    print(f"  Top {percentile}%: >= {top_threshold:.2f}")
    print(f"  Mean score range: {valid_df['mean_score'].min():.2f} - {valid_df['mean_score'].max():.2f}")
    
    # Get top and bottom samples
    bottom_samples = valid_df[valid_df['mean_score'] <= bottom_threshold].nsmallest(num_samples, 'mean_score')
    top_samples = valid_df[valid_df['mean_score'] >= top_threshold].nlargest(num_samples, 'mean_score')
    
    # Create output directories
    output_path = output_dir / 'mean_score_samples'
    top_dir = output_path / 'top_10_percent'
    bottom_dir = output_path / 'bottom_10_percent'
    top_dir.mkdir(parents=True, exist_ok=True)
    bottom_dir.mkdir(parents=True, exist_ok=True)
    
    # Save top samples
    print(f"\nSaving top {percentile}% samples (highest mean scores)...")
    saved_top = 0
    for idx, (_, row) in enumerate(top_samples.iterrows()):
        if saved_top >= num_samples:
            break
            
        sample_id = row['sample_id']
        caption = row.get('caption', '')
        mean_score = row['mean_score']
        
        # Get scores dict
        scores = {col: row[col] for col in score_columns if col in row}
        
        # Get image from tar
        img = get_image_from_tar(tar_dir, sample_id)
        if img:
            # Create annotated image
            annotated = create_clean_annotated_image(img, caption, scores, mean_score)
            
            # Save image
            output_path = top_dir / f"top_{saved_top+1:02d}_mean{mean_score:.1f}_{sample_id}.jpg"
            annotated.save(output_path, quality=95)
            print(f"  Saved top sample {saved_top+1}: mean={mean_score:.2f}, id={sample_id}")
            saved_top += 1
        else:
            print(f"  Warning: Could not find image for {sample_id}")
    
    # Save bottom samples
    print(f"\nSaving bottom {percentile}% samples (lowest mean scores)...")
    saved_bottom = 0
    for idx, (_, row) in enumerate(bottom_samples.iterrows()):
        if saved_bottom >= num_samples:
            break
            
        sample_id = row['sample_id']
        caption = row.get('caption', '')
        mean_score = row['mean_score']
        
        # Get scores dict
        scores = {col: row[col] for col in score_columns if col in row}
        
        # Get image from tar
        img = get_image_from_tar(tar_dir, sample_id)
        if img:
            # Create annotated image
            annotated = create_clean_annotated_image(img, caption, scores, mean_score)
            
            # Save image
            output_path = bottom_dir / f"bottom_{saved_bottom+1:02d}_mean{mean_score:.1f}_{sample_id}.jpg"
            annotated.save(output_path, quality=95)
            print(f"  Saved bottom sample {saved_bottom+1}: mean={mean_score:.2f}, id={sample_id}")
            saved_bottom += 1
        else:
            print(f"  Warning: Could not find image for {sample_id}")
    
    # Save summary statistics
    summary = {
        'total_samples': len(df),
        'valid_samples': len(valid_df),
        'percentile': percentile,
        'bottom_threshold': float(bottom_threshold),
        'top_threshold': float(top_threshold),
        'mean_score_stats': {
            'min': float(valid_df['mean_score'].min()),
            'max': float(valid_df['mean_score'].max()),
            'mean': float(valid_df['mean_score'].mean()),
            'median': float(valid_df['mean_score'].median()),
            'std': float(valid_df['mean_score'].std())
        },
        'saved_samples': {
            'top': saved_top,
            'bottom': saved_bottom
        }
    }
    
    summary_path = output_dir / 'mean_score_samples' / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    
    return saved_top, saved_bottom

def create_mean_score_html_gallery(output_dir: Path):
    """Create HTML gallery for mean score based samples"""
    
    sample_dir = output_dir / 'mean_score_samples'
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>MLM Filter Mean Score Gallery</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            font-size: 1.1em;
            margin-bottom: 40px;
        }
        h2 {
            color: #444;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 40px;
        }
        .section {
            margin: 30px 0;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 25px;
            margin: 25px 0;
        }
        .image-card {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .image-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }
        .image-card img {
            width: 100%;
            height: auto;
            display: block;
        }
        .top-scoring {
            border-top: 4px solid #4CAF50;
        }
        .bottom-scoring {
            border-top: 4px solid #f44336;
        }
        .stats-box {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            border-left: 4px solid #667eea;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .stat-item {
            text-align: center;
            padding: 10px;
            background: white;
            border-radius: 6px;
        }
        .stat-label {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        .stat-value {
            color: #333;
            font-size: 1.4em;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>MLM Filter Analysis</h1>
        <p class="subtitle">Top and Bottom 10% of Images Based on Mean Score</p>
"""
    
    # Load summary if exists
    summary_path = sample_dir / 'summary.json'
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        html_content += f"""
        <div class="stats-box">
            <h3>Dataset Statistics</h3>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-label">Total Samples</div>
                    <div class="stat-value">{summary['total_samples']:,}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Mean Score Range</div>
                    <div class="stat-value">{summary['mean_score_stats']['min']:.1f} - {summary['mean_score_stats']['max']:.1f}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Average Mean Score</div>
                    <div class="stat-value">{summary['mean_score_stats']['mean']:.1f}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Top 10% Threshold</div>
                    <div class="stat-value">≥ {summary['top_threshold']:.1f}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Bottom 10% Threshold</div>
                    <div class="stat-value">≤ {summary['bottom_threshold']:.1f}</div>
                </div>
            </div>
        </div>
"""
    
    # Top scoring samples
    top_dir = sample_dir / 'top_10_percent'
    if top_dir.exists():
        top_images = sorted(top_dir.glob("*.jpg"))
        if top_images:
            html_content += """
        <div class="section">
            <h2>Top 10% - Highest Quality Samples</h2>
            <div class="image-grid">
"""
            for img_path in top_images:
                rel_path = img_path.relative_to(output_dir)
                html_content += f"""
                <div class="image-card top-scoring">
                    <img src="{rel_path}" alt="{img_path.stem}" loading="lazy">
                </div>
"""
            html_content += """
            </div>
        </div>
"""
    
    # Bottom scoring samples
    bottom_dir = sample_dir / 'bottom_10_percent'
    if bottom_dir.exists():
        bottom_images = sorted(bottom_dir.glob("*.jpg"))
        if bottom_images:
            html_content += """
        <div class="section">
            <h2>Bottom 10% - Lowest Quality Samples</h2>
            <div class="image-grid">
"""
            for img_path in bottom_images:
                rel_path = img_path.relative_to(output_dir)
                html_content += f"""
                <div class="image-card bottom-scoring">
                    <img src="{rel_path}" alt="{img_path.stem}" loading="lazy">
                </div>
"""
            html_content += """
            </div>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    # Save HTML file
    html_path = output_dir / 'mean_score_gallery.html'
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML gallery saved to: {html_path}")
    return html_path

def main():
    parser = argparse.ArgumentParser(description='Save top/bottom 10% images based on mean MLM score')
    parser.add_argument('--parquet', type=str, 
                       default='./mlm_scores_final_output/mlm_scores_final.parquet',
                       help='Path to results parquet file')
    parser.add_argument('--tar_dir', type=str,
                       default='/capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain/webdataset_output',
                       help='Directory containing tar files')
    parser.add_argument('--output_dir', type=str, 
                       default='./mlm_analysis_output',
                       help='Output directory for analysis')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples to save from each group')
    parser.add_argument('--percentile', type=float, default=10.0,
                       help='Percentile threshold (default: 10 for top/bottom 10 percent)')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("SAVING TOP/BOTTOM SAMPLES BASED ON MEAN SCORE")
    print("="*70)
    
    # Save sample images based on mean score
    saved_top, saved_bottom = save_mean_based_samples(
        args.parquet, 
        args.tar_dir, 
        output_dir,
        args.num_samples,
        args.percentile
    )
    
    # Create HTML gallery
    html_path = create_mean_score_html_gallery(output_dir)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Saved {saved_top} top samples and {saved_bottom} bottom samples")
    print(f"Images saved to: {output_dir}/mean_score_samples/")
    print(f"View gallery at: {html_path}")

if __name__ == "__main__":
    main()