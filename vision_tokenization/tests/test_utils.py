"""
Shared utility functions for IndexedDataset testing.

This module provides common functionality used across all IndexedDataset test files
to reduce code duplication and ensure consistency in testing approaches.

Functions provided:
- read_index_header(): Read and parse IndexedDataset header (34 bytes)
- read_index_file(): Read complete index file including all arrays
- calculate_expected_pointers(): Calculate expected sequence pointer offsets
- verify_indexed_dataset(): Comprehensive dataset validation
- extract_sequences(): Extract specific sequences from a dataset
- compare_token_sequences(): Compare two sets of token sequences
- create_test_indexed_dataset(): Helper to create test datasets

These utilities handle the low-level details of the IndexedDataset format,
allowing test files to focus on their specific testing goals without
reimplementing common operations.

Usage:
    from test_utils import read_index_header, verify_indexed_dataset
    
    header = read_index_header("dataset.idx")
    result = verify_indexed_dataset("dataset")
"""

import struct
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List


def read_index_header(idx_path: str) -> Dict:
    """Read and parse the IndexedDataset header."""
    with open(idx_path, "rb") as f:
        magic = f.read(9)
        if magic != b"MMIDIDX\x00\x00":
            raise ValueError(f"Invalid magic header: {magic}")
        
        version = struct.unpack("<Q", f.read(8))[0]
        dtype_code = struct.unpack("<B", f.read(1))[0]
        num_sequences = struct.unpack("<Q", f.read(8))[0]
        num_documents = struct.unpack("<Q", f.read(8))[0]
        
        return {
            "magic": magic,
            "version": version,
            "dtype_code": dtype_code,
            "dtype_size": 2 if dtype_code == 8 else 4,  # uint16 vs int32
            "num_sequences": num_sequences,
            "num_documents": num_documents,
            "header_size": f.tell()
        }


def read_index_file(idx_path: str) -> Dict:
    """Read complete index file including arrays."""
    header = read_index_header(idx_path)
    
    with open(idx_path, "rb") as f:
        f.seek(header["header_size"])
        
        # Read arrays
        num_seq = header["num_sequences"]
        num_doc = header["num_documents"]
        
        seq_lengths = np.frombuffer(f.read(num_seq * 4), dtype=np.int32)
        seq_pointers = np.frombuffer(f.read(num_seq * 8), dtype=np.int64)
        doc_indices = np.frombuffer(f.read((num_doc + 1) * 8), dtype=np.int64)
        
        return {
            **header,
            "seq_lengths": seq_lengths,
            "seq_pointers": seq_pointers,
            "doc_indices": doc_indices,
        }


def calculate_expected_pointers(seq_lengths: List[int], dtype_size: int = 4) -> List[int]:
    """Calculate expected sequence pointers given sequence lengths."""
    pointers = [0]
    for i in range(1, len(seq_lengths)):
        pointers.append(pointers[-1] + seq_lengths[i-1] * dtype_size)
    return pointers


def verify_indexed_dataset(prefix: str) -> Dict:
    """
    Verify an IndexedDataset and return detailed information.
    
    Returns:
        Dict with verification results including validity and statistics
    """
    idx_path = Path(f"{prefix}.idx")
    bin_path = Path(f"{prefix}.bin")
    
    if not idx_path.exists() or not bin_path.exists():
        return {"valid": False, "error": "Files not found"}
    
    try:
        # Read index structure
        data = read_index_file(str(idx_path))
        
        # Verify binary file
        bin_size = bin_path.stat().st_size
        total_tokens = data["seq_lengths"].sum()
        expected_bin_size = total_tokens * data["dtype_size"]
        
        # Verify pointers
        expected_ptrs = calculate_expected_pointers(
            data["seq_lengths"].tolist(), 
            data["dtype_size"]
        )
        pointer_match = np.array_equal(data["seq_pointers"], expected_ptrs)
        
        # Verify document indices
        doc_check = (data["doc_indices"][0] == 0 and 
                    data["doc_indices"][-1] == data["num_sequences"])
        
        # Read sample tokens with correct dtype
        dtype = np.uint16 if data["dtype_code"] == 8 else np.int32
        tokens = np.fromfile(bin_path, dtype=dtype, count=min(1000, total_tokens))
        
        return {
            "valid": True,
            "num_sequences": data["num_sequences"],
            "num_documents": data["num_documents"],
            "total_tokens": int(total_tokens),
            "token_range": (int(tokens.min()), int(tokens.max())),
            "avg_sequence_length": float(data["seq_lengths"].mean()),
            "pointer_check": pointer_match,
            "doc_check": doc_check,
            "bin_size_check": bin_size == expected_bin_size,
            "dtype_code": data["dtype_code"],
            "dtype_size": data["dtype_size"]
        }
        
    except Exception as e:
        return {"valid": False, "error": str(e)}


def extract_sequences(prefix: str, indices: List[int] = None) -> List[np.ndarray]:
    """
    Extract sequences from an IndexedDataset.
    
    Args:
        prefix: Dataset prefix
        indices: List of sequence indices to extract (None = all)
        
    Returns:
        List of numpy arrays containing tokens for each sequence
    """
    data = read_index_file(f"{prefix}.idx")
    # Determine dtype from dtype_code
    if data["dtype_code"] == 8:
        dtype = np.uint16
    else:
        dtype = np.int32
    all_tokens = np.fromfile(f"{prefix}.bin", dtype=dtype)
    
    if indices is None:
        indices = range(data["num_sequences"])
    
    sequences = []
    for i in indices:
        if i >= data["num_sequences"]:
            raise IndexError(f"Sequence {i} out of range")
            
        start = data["seq_pointers"][i] // data["dtype_size"]  # Convert bytes to token index
        length = data["seq_lengths"][i]
        sequences.append(all_tokens[start:start + length])
    
    return sequences


def compare_token_sequences(seq1: List[np.ndarray], seq2: List[np.ndarray]) -> Tuple[bool, List[Dict]]:
    """
    Compare two lists of token sequences.
    
    Returns:
        Tuple of (all_match, results) where results contains details for each sequence
    """
    all_match = True
    results = []
    
    for i, (s1, s2) in enumerate(zip(seq1, seq2)):
        match = np.array_equal(s1, s2)
        
        result = {
            "index": i,
            "match": match,
            "len1": len(s1),
            "len2": len(s2),
        }
        
        if not match:
            all_match = False
            if len(s1) != len(s2):
                result["error"] = f"Length mismatch: {len(s1)} vs {len(s2)}"
            else:
                diff_idx = np.where(s1 != s2)[0]
                if len(diff_idx) > 0:
                    result["first_diff_idx"] = int(diff_idx[0])
                    result["first_diff_val1"] = int(s1[diff_idx[0]])
                    result["first_diff_val2"] = int(s2[diff_idx[0]])
        
        results.append(result)
    
    return all_match, results


def create_test_indexed_dataset(prefix: str, sequences: List[List[int]], 
                              multimodal: bool = False, text_vocab_size: int = 0):
    """
    Helper to create a test IndexedDataset.
    
    Args:
        prefix: Output prefix
        sequences: List of token sequences
        multimodal: Whether to use multimodal tokenizer
        text_vocab_size: Text vocabulary size for multimodal
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    if multimodal:
        from vision_tokenization.utils.indexed_dataset_megatron import VisionTokenIndexedDatasetBuilder
        builder = VisionTokenIndexedDatasetBuilder(
            output_prefix=prefix,
            image_vocab_size=32768,
            text_vocab_size=text_vocab_size
        )
        for seq in sequences:
            builder.add_image_tokens(np.array(seq, dtype=np.int32))
    else:
        from vision_tokenization.utils.indexed_dataset_megatron import IndexedDatasetBuilder
        builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
        for seq in sequences:
            builder.add_document(seq, lengths=[len(seq)])
    
    if multimodal:
        builder.finalize()  # VisionTokenIndexedDatasetBuilder handles paths internally
    else:
        builder.finalize(f"{prefix}.idx")  # IndexedDatasetBuilder needs explicit idx path