#!/usr/bin/env python3
"""
Run DFN processing on partial dataset with output in current DFN folder
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Process partial dataset with DFN filtering')
    parser.add_argument('--num_shards', type=int, default=5,
                      help='Number of shards to process (default: 5)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for processing (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading (default: 4)')
    parser.add_argument('--num_gpus', type=int, default=4,
                      help='Number of GPUs to use (default: 4)')
    
    args = parser.parse_args()
    
    # Dataset path
    dataset_path = "/capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain/webdataset_output"
    
    # Output path in current DFN folder
    output_path = Path(__file__).parent / "partial_output"
    output_path.mkdir(exist_ok=True)
    
    print(f"Processing {args.num_shards} shards from dataset")
    print(f"Output will be saved to: {output_path}")
    print(f"Using {args.num_gpus} GPUs with batch size {args.batch_size}")
    print("-" * 50)
    
    # Build processing command with bash
    process_cmd = f"""
    bash -c '
    unset LD_LIBRARY_PATH && unset PYTHONPATH
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate dfn_filter
    python process_webdataset.py \\
        --dataset_path {dataset_path} \\
        --output_path {output_path} \\
        --batch_size {args.batch_size} \\
        --num_workers {args.num_workers} \\
        --num_gpus {args.num_gpus} \\
        --max_shards {args.num_shards}
    '
    """
    
    # Execute
    os.system(process_cmd)
    
    # After processing, combine parquet files if multiple exist
    combine_cmd = f"""
    bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate dfn_filter && python -c "
import pandas as pd
from pathlib import Path
import glob

output_path = Path('{output_path}')
parquet_files = list(output_path.glob('dfn_scores_*.parquet'))

if len(parquet_files) > 1:
    print(f'Combining {{len(parquet_files)}} parquet files...')
    dfs = [pd.read_parquet(f) for f in parquet_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_parquet(output_path / 'dfn_scores_partial.parquet', index=False)
    print(f'Combined scores saved to: {{output_path}}/dfn_scores_partial.parquet')
    print(f'Total samples: {{len(combined_df)}}')
    print(f'Score statistics:')
    print(combined_df['dfn_score'].describe())
elif len(parquet_files) == 1:
    import shutil
    shutil.copy(parquet_files[0], output_path / 'dfn_scores_partial.parquet')
    df = pd.read_parquet(parquet_files[0])
    print(f'Scores saved to: {{output_path}}/dfn_scores_partial.parquet')
    print(f'Total samples: {{len(df)}}')
    print(f'Score statistics:')
    print(df['dfn_score'].describe())
"'
    """
    
    os.system(combine_cmd)
    
    print("\n" + "=" * 50)
    print("Processing complete!")
    print(f"Results saved in: {output_path}")
    print("\nNext steps:")
    print("1. Check scores: pandas.read_parquet('partial_output/dfn_scores_partial.parquet')")
    print("2. Visualize: ./run_custom_visualization.sh --input_scores partial_output/dfn_scores_partial.parquet")

if __name__ == "__main__":
    main()