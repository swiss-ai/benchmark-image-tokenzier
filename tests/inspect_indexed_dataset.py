#!/usr/bin/env python3
"""
Utility to inspect Megatron IndexedDataset files.

Usage:
    python inspect_indexed_dataset.py <path_prefix> [options]
    
Options:
    --seq <index>     Extract and display a specific sequence
    --verbose         Show detailed byte-level information
    --compare <path>  Compare two datasets
    
Examples:
    python inspect_indexed_dataset.py /path/to/dataset
    python inspect_indexed_dataset.py /path/to/dataset --seq 0
    python inspect_indexed_dataset.py /path/to/dataset --verbose
"""

import sys
import struct
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Tuple


def read_header(f) -> Tuple[int, int, int, int]:
    """Read IndexedDataset header."""
    magic = f.read(9)
    if magic != b"MMIDIDX\x00\x00":
        raise ValueError(f"Invalid magic header: {magic}")
    
    version = struct.unpack("<Q", f.read(8))[0]
    dtype_code = struct.unpack("<B", f.read(1))[0]
    num_sequences = struct.unpack("<Q", f.read(8))[0]
    num_documents = struct.unpack("<Q", f.read(8))[0]
    
    return version, dtype_code, num_sequences, num_documents


def load_index(prefix: str) -> Dict:
    """Load complete index file."""
    with open(f"{prefix}.idx", "rb") as f:
        version, dtype_code, num_sequences, num_documents = read_header(f)
        
        # Read arrays
        seq_lengths = np.frombuffer(f.read(num_sequences * 4), dtype=np.int32)
        seq_pointers = np.frombuffer(f.read(num_sequences * 8), dtype=np.int64)
        doc_indices = np.frombuffer(f.read((num_documents + 1) * 8), dtype=np.int64)
        
        return {
            "version": version,
            "dtype_code": dtype_code,
            "num_sequences": num_sequences,
            "num_documents": num_documents,
            "seq_lengths": seq_lengths,
            "seq_pointers": seq_pointers,
            "doc_indices": doc_indices,
        }


def extract_sequence(prefix: str, seq_idx: int) -> np.ndarray:
    """Extract a specific sequence."""
    index = load_index(prefix)
    
    if seq_idx >= index["num_sequences"]:
        raise ValueError(f"Sequence {seq_idx} out of range (0-{index['num_sequences']-1})")
    
    offset = index["seq_pointers"][seq_idx]
    length = index["seq_lengths"][seq_idx]
    
    with open(f"{prefix}.bin", "rb") as f:
        f.seek(offset)
        tokens = np.frombuffer(f.read(length * 4), dtype=np.int32)
    
    return tokens


def inspect_dataset(prefix: str, verbose: bool = False):
    """Inspect and display dataset information."""
    idx_path = Path(f"{prefix}.idx")
    bin_path = Path(f"{prefix}.bin")
    
    if not idx_path.exists() or not bin_path.exists():
        print(f"Error: Files not found at {prefix}")
        return
    
    print(f"\n{'='*70}")
    print(f"INDEXED DATASET: {prefix}")
    print(f"{'='*70}")
    
    # Load index
    index = load_index(prefix)
    
    # File sizes
    idx_size = idx_path.stat().st_size
    bin_size = bin_path.stat().st_size
    
    print(f"\n📁 FILES:")
    print(f"  Index: {idx_size:,} bytes")
    print(f"  Binary: {bin_size:,} bytes")
    
    print(f"\n📋 METADATA:")
    print(f"  Version: {index['version']}")
    print(f"  Data type: {'int32' if index['dtype_code'] == 4 else 'uint16' if index['dtype_code'] == 8 else 'unknown'}")
    print(f"  Sequences: {index['num_sequences']:,}")
    print(f"  Documents: {index['num_documents']:,}")
    
    print(f"\n📊 STATISTICS:")
    total_tokens = index['seq_lengths'].sum()
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg tokens/sequence: {index['seq_lengths'].mean():.1f}")
    print(f"  Min sequence length: {index['seq_lengths'].min():,}")
    print(f"  Max sequence length: {index['seq_lengths'].max():,}")
    
    # Verify consistency
    expected_bin_size = total_tokens * 4
    print(f"\n✅ VALIDATION:")
    print(f"  Binary size check: {'PASS' if bin_size == expected_bin_size else 'FAIL'}")
    
    # Check pointers
    expected_ptrs = np.zeros(index['num_sequences'], dtype=np.int64)
    expected_ptrs[1:] = np.cumsum(index['seq_lengths'][:-1]) * 4
    ptr_check = np.array_equal(index['seq_pointers'], expected_ptrs)
    print(f"  Pointer check: {'PASS' if ptr_check else 'FAIL'}")
    
    # Check document indices
    doc_check = index['doc_indices'][0] == 0 and index['doc_indices'][-1] == index['num_sequences']
    print(f"  Document indices check: {'PASS' if doc_check else 'FAIL'}")
    
    # Sample tokens
    print(f"\n🔢 TOKEN SAMPLE:")
    sample_tokens = np.fromfile(f"{prefix}.bin", dtype=np.int32, count=100)
    print(f"  First 10 tokens: {sample_tokens[:10].tolist()}")
    print(f"  Token range: [{sample_tokens.min():,}, {sample_tokens.max():,}]")
    
    # Check for multimodal offset
    if sample_tokens.min() >= 131072:
        print(f"  📌 Multimodal tokenizer detected")
        print(f"     Text vocab offset: 131072")
        print(f"     Image token range: [{sample_tokens.min()-131072:,}, {sample_tokens.max()-131072:,}]")
    
    if verbose:
        print(f"\n🔍 DETAILED VIEW:")
        print(f"  First 5 sequences:")
        for i in range(min(5, index['num_sequences'])):
            print(f"    Seq {i}: {index['seq_lengths'][i]:,} tokens @ offset {index['seq_pointers'][i]:,}")
        
        print(f"\n  First 5 documents:")
        for i in range(min(5, index['num_documents'])):
            start = index['doc_indices'][i]
            end = index['doc_indices'][i+1]
            print(f"    Doc {i}: sequences {start}-{end-1} ({end-start} sequences)")


def compare_datasets(prefix1: str, prefix2: str):
    """Compare two IndexedDatasets."""
    print(f"\n{'='*70}")
    print(f"DATASET COMPARISON")
    print(f"{'='*70}")
    
    try:
        idx1 = load_index(prefix1)
        idx2 = load_index(prefix2)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    print(f"\nDataset 1: {prefix1}")
    print(f"Dataset 2: {prefix2}")
    
    print(f"\n📊 COMPARISON:")
    print(f"{'':30} {'Dataset 1':>15} {'Dataset 2':>15} {'Match':>10}")
    print(f"{'-'*70}")
    
    # Compare metadata
    fields = [
        ("Version", "version"),
        ("Data type", "dtype_code"),
        ("Sequences", "num_sequences"),
        ("Documents", "num_documents"),
    ]
    
    for label, field in fields:
        v1 = idx1[field]
        v2 = idx2[field]
        match = "✓" if v1 == v2 else "✗"
        print(f"{label:30} {v1:>15} {v2:>15} {match:>10}")
    
    # Compare statistics
    t1 = idx1['seq_lengths'].sum()
    t2 = idx2['seq_lengths'].sum()
    match = "✓" if t1 == t2 else "✗"
    print(f"{'Total tokens':30} {t1:>15,} {t2:>15,} {match:>10}")
    
    # If same number of sequences, compare lengths
    if idx1['num_sequences'] == idx2['num_sequences']:
        matching_lengths = np.sum(idx1['seq_lengths'] == idx2['seq_lengths'])
        print(f"\n📝 Sequence length comparison:")
        print(f"  Matching: {matching_lengths}/{idx1['num_sequences']} sequences")
        
        if matching_lengths < idx1['num_sequences']:
            # Find first mismatch
            mismatches = np.where(idx1['seq_lengths'] != idx2['seq_lengths'])[0]
            if len(mismatches) > 0:
                i = mismatches[0]
                print(f"  First mismatch at sequence {i}:")
                print(f"    Dataset 1: {idx1['seq_lengths'][i]:,} tokens")
                print(f"    Dataset 2: {idx2['seq_lengths'][i]:,} tokens")


def main():
    parser = argparse.ArgumentParser(description="Inspect Megatron IndexedDataset files")
    parser.add_argument("prefix", help="Path prefix for dataset files")
    parser.add_argument("--seq", type=int, help="Extract specific sequence")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information")
    parser.add_argument("--compare", help="Compare with another dataset")
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            compare_datasets(args.prefix, args.compare)
        else:
            inspect_dataset(args.prefix, args.verbose)
            
            if args.seq is not None:
                print(f"\n{'='*70}")
                print(f"EXTRACTING SEQUENCE {args.seq}")
                print(f"{'='*70}")
                
                tokens = extract_sequence(args.prefix, args.seq)
                print(f"Length: {len(tokens)} tokens")
                print(f"First 20: {tokens[:20].tolist()}")
                if len(tokens) > 20:
                    print(f"Last 20: {tokens[-20:].tolist()}")
                print(f"Range: [{tokens.min()}, {tokens.max()}]")
                
                # Check for multimodal
                if tokens.min() >= 131072:
                    adj_tokens = tokens - 131072
                    print(f"Adjusted range: [{adj_tokens.min()}, {adj_tokens.max()}]")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()