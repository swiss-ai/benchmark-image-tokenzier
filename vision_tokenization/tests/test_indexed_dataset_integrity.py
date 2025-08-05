#!/usr/bin/env python3
"""
Test suite for IndexedDataset data integrity and recovery operations.

This module focuses on verifying data correctness, handling edge cases, and ensuring
that tokens can be properly recovered from IndexedDatasets, especially in complex
scenarios like multimodal tokenization with vocabulary offsets.

Focus areas:
- Multimodal token recovery (removing text vocabulary offsets)
- Large dataset stress testing (100+ sequences)
- Token value preservation at data type boundaries (uint16/int32 limits)
- Token sequence comparison utilities
- Edge cases in data storage and retrieval

This test suite complements test_indexed_dataset_format.py by focusing on data
integrity rather than format compliance. For end-to-end pipeline tests, see
test_vision_pipeline_integration.py.

Run with:
    pytest test_indexed_dataset_integrity.py -v
    python -m pytest test_indexed_dataset_integrity.py -v
"""

import sys
import os
import struct
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vision_tokenization.utils.indexed_dataset_megatron import (
    IndexedDatasetBuilder, VisionTokenIndexedDatasetBuilder
)
from test_utils import read_index_file, calculate_expected_pointers, compare_token_sequences


class TestDatasetVerification:
    """Test dataset verification, recovery, and comparison functions.
    
    This test file focuses on:
    - Token recovery from multimodal datasets
    - Large dataset stress testing
    - Token value preservation and edge cases
    - Token comparison functionality
    
    For basic format tests, see test_megatron_indexed_dataset.py
    """
    
    @classmethod
    def setup_class(cls):
        """Setup test environment."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_verification_")
        
    @classmethod
    def teardown_class(cls):
        """Cleanup."""
        shutil.rmtree(cls.temp_dir)
    
    
    # Note: test_sequence_pointer_calculation removed - duplicate of test_megatron_indexed_dataset.py::test_sequence_pointers
    # Note: test_document_indices_structure removed - duplicate of test_megatron_indexed_dataset.py::test_document_indices
    
    def test_multimodal_token_recovery(self):
        """Test recovering original tokens from multimodal dataset."""
        # Original vision tokens (before offset)
        original_tokens = [
            np.array([100, 200, 300], dtype=np.int32),
            np.array([1000, 2000], dtype=np.int32),
            np.array([5000, 6000, 7000, 8000], dtype=np.int32),
        ]
        
        text_vocab_size = 131072
        prefix = os.path.join(self.temp_dir, "test_recovery")
        
        # Create dataset with offset
        builder = VisionTokenIndexedDatasetBuilder(
            output_prefix=prefix,
            image_vocab_size=32768,
            text_vocab_size=text_vocab_size
        )
        
        for tokens in original_tokens:
            builder.add_image_tokens(tokens)
        
        builder.finalize()
        
        # Read saved tokens
        saved_tokens_raw = np.fromfile(f"{prefix}.bin", dtype=np.int32)
        
        # Verify offset was applied
        assert saved_tokens_raw.min() >= text_vocab_size, \
            "Multimodal offset not applied"
        
        # Read index structure
        data = read_index_file(f"{prefix}.idx")
        
        # Extract and verify each sequence
        for i, expected_tokens in enumerate(original_tokens):
            start = data["seq_pointers"][i] // 4
            length = data["seq_lengths"][i]
            
            # Extract with offset
            saved_with_offset = saved_tokens_raw[start:start + length]
            
            # Remove offset
            recovered = saved_with_offset - text_vocab_size
            
            assert np.array_equal(recovered, expected_tokens), \
                f"Sequence {i}: recovery failed"
    
    # Note: test_empty_sequences_handling removed - similar to test_megatron_indexed_dataset.py::test_empty_dataset
    
    def test_large_dataset_structure(self):
        """Test structure with many sequences."""
        num_sequences = 100
        prefix = os.path.join(self.temp_dir, "test_large")
        
        builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
        
        # Add many sequences with varying lengths
        for i in range(num_sequences):
            length = 10 + (i % 20)  # Lengths from 10 to 29
            tokens = list(range(i * 100, i * 100 + length))
            builder.add_document(tokens, lengths=[length])
        
        builder.finalize(f"{prefix}.idx")
        
        # Verify structure
        data = read_index_file(f"{prefix}.idx")
        
        assert data["num_sequences"] == num_sequences
        assert data["num_documents"] == num_sequences
        assert len(data["seq_lengths"]) == num_sequences
        assert len(data["seq_pointers"]) == num_sequences
        assert len(data["doc_indices"]) == num_sequences + 1
        
        # Verify pointers are monotonic
        for i in range(1, num_sequences):
            assert data["seq_pointers"][i] > data["seq_pointers"][i-1], \
                f"Pointers not monotonic at {i}"
        
        # Verify total file size
        total_tokens = data["seq_lengths"].sum()
        expected_bin_size = total_tokens * 4
        actual_bin_size = os.path.getsize(f"{prefix}.bin")
        assert actual_bin_size == expected_bin_size, \
            f"Binary size mismatch: {actual_bin_size} != {expected_bin_size}"
    
    def test_token_value_preservation(self):
        """Test that token values are preserved exactly."""
        # Test with edge case values
        test_sequences = [
            [0, 1, 2, 3, 4],           # Small values
            [32767, 32768, 32769],     # Around 2^15
            [65535, 65536, 65537],     # Around 2^16
            [131071, 131072, 131073],  # Around text vocab boundary
        ]
        
        prefix = os.path.join(self.temp_dir, "test_values")
        
        # Test without offset
        builder = VisionTokenIndexedDatasetBuilder(
            output_prefix=prefix,
            image_vocab_size=200000,
            text_vocab_size=0  # No offset
        )
        
        for seq in test_sequences:
            builder.add_image_tokens(np.array(seq, dtype=np.int32))
        
        builder.finalize()
        
        # Read and verify each sequence
        all_tokens = np.fromfile(f"{prefix}.bin", dtype=np.int32)
        data = read_index_file(f"{prefix}.idx")
        
        for i, expected_seq in enumerate(test_sequences):
            start = data["seq_pointers"][i] // 4
            length = data["seq_lengths"][i]
            saved_seq = all_tokens[start:start + length]
            
            assert np.array_equal(saved_seq, expected_seq), \
                f"Sequence {i} values not preserved: {saved_seq.tolist()} != {expected_seq}"


def test_compare_tokens_functionality():
    """Test the token comparison functionality from the notebook."""
    
    # Test cases
    fresh = [
        np.array([1, 2, 3, 4, 5]),
        np.array([10, 20, 30]),
        np.array([100, 200, 300, 400]),
    ]
    
    # Case 1: Perfect match
    saved_match = [
        np.array([1, 2, 3, 4, 5]),
        np.array([10, 20, 30]),
        np.array([100, 200, 300, 400]),
    ]
    
    match, results = compare_token_sequences(fresh, saved_match)
    assert match, "Perfect match test failed"
    
    # Case 2: Length mismatch
    saved_length = [
        np.array([1, 2, 3, 4]),  # Missing last element
        np.array([10, 20, 30]),
        np.array([100, 200, 300, 400]),
    ]
    
    match, results = compare_token_sequences(fresh, saved_length)
    assert not match, "Length mismatch should fail"
    assert results[0]["error"] == "Length mismatch: 5 vs 4"
    
    # Case 3: Value mismatch
    saved_value = [
        np.array([1, 2, 3, 4, 5]),
        np.array([10, 20, 31]),  # Last value different
        np.array([100, 200, 300, 400]),
    ]
    
    match, results = compare_token_sequences(fresh, saved_value)
    assert not match, "Value mismatch should fail"
    assert results[1]["first_diff_idx"] == 2
    assert results[1]["first_diff_val1"] == 30
    assert results[1]["first_diff_val2"] == 31
    
    print("✓ Token comparison functionality test passed")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])