#!/usr/bin/env python3
"""
Check the actual scores in the output parquet file
"""

import pandas as pd
import sys

# Read the parquet file
df = pd.read_parquet('mlm_scores_output/mlm_scores_final_fixed.parquet')

print('='*60)
print('SCORE ANALYSIS')
print('='*60)
print(f'DataFrame shape: {df.shape}')
print(f'Columns: {list(df.columns)}')

print('\n' + '='*60)
print('FIRST 5 SAMPLES')
print('='*60)
for i in range(min(5, len(df))):
    row = df.iloc[i]
    print(f"\nSample {i+1}:")
    print(f"  ID: {row['sample_id']}")
    print(f"  Caption: {row['caption'][:100]}...")
    print(f"  Scores:")
    print(f"    image_text_matching: {row['image_text_matching_score']}")
    print(f"    object_detail_fulfillment: {row['object_detail_fulfillment_score']}")
    print(f"    caption_text_quality: {row['caption_text_quality_score']}")
    print(f"    semantic_understanding: {row['semantic_understanding_score']}")

print('\n' + '='*60)
print('SCORE DISTRIBUTIONS')
print('='*60)
for metric in ['image_text_matching', 'object_detail_fulfillment', 'caption_text_quality', 'semantic_understanding']:
    col_name = f"{metric}_score"
    if col_name in df.columns:
        print(f"\n{metric}:")
        print(f"  Value counts:")
        value_counts = df[col_name].value_counts().sort_index()
        for val, count in value_counts.items():
            print(f"    {val}: {count} ({100*count/len(df):.1f}%)")
        print(f"  Mean: {df[col_name].mean():.1f}")
        print(f"  Std: {df[col_name].std():.1f}")

print('\n' + '='*60)
print('SCORE VALIDITY CHECK')
print('='*60)

# Check if scores are in expected range
expected_scores = {15, 25, 65, 75, 85}
for metric in ['image_text_matching', 'object_detail_fulfillment', 'caption_text_quality', 'semantic_understanding']:
    col_name = f"{metric}_score"
    if col_name in df.columns:
        unique_scores = set(df[col_name].unique())
        unexpected = unique_scores - expected_scores
        if unexpected:
            print(f"WARNING: {metric} has unexpected scores: {unexpected}")
        else:
            print(f"✓ {metric}: All scores are in expected discrete values")

# Check for any -1 (error) values
error_counts = {}
for metric in ['image_text_matching', 'object_detail_fulfillment', 'caption_text_quality', 'semantic_understanding']:
    col_name = f"{metric}_score"
    if col_name in df.columns:
        error_count = (df[col_name] == -1).sum()
        if error_count > 0:
            error_counts[metric] = error_count

if error_counts:
    print(f"\nERROR: Found -1 values indicating failed extractions:")
    for metric, count in error_counts.items():
        print(f"  {metric}: {count} errors")
else:
    print(f"\n✓ No error values (-1) found in any metric")

print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f"Total samples processed: {len(df)}")
print(f"All metrics have scores: {all(df[f'{m}_score'].notna().all() for m in ['image_text_matching', 'object_detail_fulfillment', 'caption_text_quality', 'semantic_understanding'])}")