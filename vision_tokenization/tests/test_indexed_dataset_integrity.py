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

import os
import shutil
import sys
import tempfile

import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vision_tokenization.formats.megatron import IndexedDatasetBuilder

from .test_utils import compare_token_sequences, read_index_file


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
            assert data["seq_pointers"][i] > data["seq_pointers"][i - 1], f"Pointers not monotonic at {i}"

        # Verify total file size
        total_tokens = data["seq_lengths"].sum()
        expected_bin_size = total_tokens * 4
        actual_bin_size = os.path.getsize(f"{prefix}.bin")
        assert actual_bin_size == expected_bin_size, f"Binary size mismatch: {actual_bin_size} != {expected_bin_size}"

    def test_token_value_preservation(self):
        """Token values must survive the bin/idx round trip exactly.

        The interesting values are the dtype boundaries and the id ranges
        the Apertus tokenizers actually use:
        2^15, 2^16, the 1.5 base vocab at 131072, and the Apertus 2 vision band at 200064+.
        """
        test_sequences = [
            [0, 1, 2, 3, 4],
            [32767, 32768, 32769],
            [65535, 65536, 65537],
            [131071, 131072, 131073],
            [200063, 200064, 331135, 331136],
        ]

        prefix = os.path.join(self.temp_dir, "test_values")
        builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
        for seq in test_sequences:
            builder.add_item(np.array(seq, dtype=np.int32))
            builder.end_document()
        builder.finalize(f"{prefix}.idx")

        all_tokens = np.fromfile(f"{prefix}.bin", dtype=np.int32)
        data = read_index_file(f"{prefix}.idx")

        for i, expected_seq in enumerate(test_sequences):
            start = data["seq_pointers"][i] // 4
            length = data["seq_lengths"][i]
            saved_seq = all_tokens[start : start + length]

            assert np.array_equal(saved_seq, expected_seq), (
                f"Sequence {i} values not preserved: "
                f"{saved_seq.tolist()} != {expected_seq}"
            )


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
