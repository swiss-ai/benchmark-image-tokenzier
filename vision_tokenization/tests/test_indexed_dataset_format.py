#!/usr/bin/env python3
"""
Test suite for Megatron IndexedDataset format specification.

This module tests the core binary format and file structure of Megatron's IndexedDataset,
ensuring that files are created and read according to the exact specification.

Focus areas:
- Binary file format verification (header structure, magic bytes, version)
- Index file structure validation (sequence lengths, pointers, document indices)
- Basic I/O operations (create, write, read, finalize)
- File size calculations and data type handling (int32, uint16)
- Empty and large dataset edge cases

This is the foundational test suite that ensures IndexedDataset files conform to
Megatron's expected format. For data integrity and recovery tests, see
test_indexed_dataset_integrity.py.

Run with:
    pytest test_indexed_dataset_format.py -v
    python -m pytest test_indexed_dataset_format.py -v
"""

import os
import shutil
import struct
import sys
import tempfile

import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vision_tokenization.utils.indexed_dataset_megatron import IndexedDatasetBuilder, VisionTokenIndexedDatasetBuilder

from .test_utils import calculate_expected_pointers, read_index_file, read_index_header


class TestIndexedDatasetFormat:
    """Test the Megatron IndexedDataset file format."""

    @classmethod
    def setup_class(cls):
        """Create temporary directory for all tests."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_indexed_dataset_")

    @classmethod
    def teardown_class(cls):
        """Clean up temporary directory."""
        shutil.rmtree(cls.temp_dir)

    def test_header_format(self):
        """Test that header is written correctly."""
        prefix = os.path.join(self.temp_dir, "test_header")

        # Create minimal dataset
        builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
        builder.add_document([1, 2, 3], lengths=[3])
        builder.finalize(f"{prefix}.idx")

        # Verify header
        header = read_index_header(f"{prefix}.idx")

        assert header["magic"] == b"MMIDIDX\x00\x00", "Invalid magic header"
        assert header["version"] == 1, "Invalid version"
        assert header["dtype_code"] == 4, "Invalid dtype code for int32"
        assert header["num_sequences"] == 1, "Invalid sequence count"
        assert header["num_documents"] == 1, "Invalid document count"
        assert header["header_size"] == 34, "Header size should be 34 bytes"

    def test_sequence_pointers(self):
        """Test that sequence pointers are calculated correctly."""
        prefix = os.path.join(self.temp_dir, "test_pointers")

        # Create dataset with known sequences
        sequences = [
            [1, 2, 3, 4, 5],  # 5 tokens -> offset 0
            [10, 20, 30],  # 3 tokens -> offset 20 (5*4)
            [100, 200, 300, 400],  # 4 tokens -> offset 32 (20 + 3*4)
        ]

        builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
        for seq in sequences:
            builder.add_document(seq, lengths=[len(seq)])
        builder.finalize(f"{prefix}.idx")

        # Read and verify
        data = read_index_file(f"{prefix}.idx")

        expected_pointers = calculate_expected_pointers([5, 3, 4], 4)
        assert np.array_equal(
            data["seq_pointers"], expected_pointers
        ), f"Pointers mismatch: expected {expected_pointers}, got {data['seq_pointers'].tolist()}"

    def test_document_indices(self):
        """Test document index array format."""
        prefix = os.path.join(self.temp_dir, "test_docs")

        # Create 3 documents
        builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
        builder.add_document([1, 2], lengths=[2])
        builder.add_document([3, 4, 5], lengths=[3])
        builder.add_document([6], lengths=[1])
        builder.finalize(f"{prefix}.idx")

        # Verify
        data = read_index_file(f"{prefix}.idx")

        # Document indices should be [0, 1, 2, 3]
        expected_doc_indices = [0, 1, 2, 3]
        assert np.array_equal(
            data["doc_indices"], expected_doc_indices
        ), f"Document indices mismatch: expected {expected_doc_indices}, got {data['doc_indices'].tolist()}"

    def test_binary_file_content(self):
        """Test that tokens are written correctly to binary file."""
        prefix = os.path.join(self.temp_dir, "test_binary")

        # Create dataset
        tokens = [10, 20, 30, 40, 50]
        builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
        builder.add_document(tokens, lengths=[len(tokens)])
        builder.finalize(f"{prefix}.idx")

        # Read binary file
        saved_tokens = np.fromfile(f"{prefix}.bin", dtype=np.int32)

        assert np.array_equal(saved_tokens, tokens), f"Token mismatch: expected {tokens}, got {saved_tokens.tolist()}"

    def test_multimodal_tokenizer_offset(self):
        """Test vision tokenizer with multimodal offset."""
        prefix = os.path.join(self.temp_dir, "test_multimodal")

        # Original vision tokens
        vision_tokens = np.array([100, 200, 300], dtype=np.int32)
        text_vocab_size = 131072

        # Create vision dataset
        builder = VisionTokenIndexedDatasetBuilder(
            output_prefix=prefix, image_vocab_size=32768, text_vocab_size=text_vocab_size
        )
        builder.add_image_tokens(vision_tokens)
        builder.finalize()

        # Read saved tokens
        saved_tokens = np.fromfile(f"{prefix}.bin", dtype=np.int32)

        # Verify offset was applied
        expected_tokens = vision_tokens + text_vocab_size
        assert np.array_equal(
            saved_tokens, expected_tokens
        ), f"Offset not applied: expected {expected_tokens.tolist()}, got {saved_tokens.tolist()}"

    def test_file_size_calculation(self):
        """Test that file sizes match expected values."""
        prefix = os.path.join(self.temp_dir, "test_filesize")

        # Create dataset with 2 sequences, 2 documents
        builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
        builder.add_document([1, 2, 3], lengths=[3])  # 3 tokens
        builder.add_document([4, 5], lengths=[2])  # 2 tokens
        builder.finalize(f"{prefix}.idx")

        # Calculate expected sizes
        num_sequences = 2
        num_documents = 2
        total_tokens = 5

        expected_idx_size = (
            34  # header
            + (num_sequences * 4)  # sequence lengths
            + (num_sequences * 8)  # sequence pointers
            + ((num_documents + 1) * 8)  # document indices
        )
        expected_bin_size = total_tokens * 4  # int32

        # Check actual sizes
        actual_idx_size = os.path.getsize(f"{prefix}.idx")
        actual_bin_size = os.path.getsize(f"{prefix}.bin")

        assert (
            actual_idx_size == expected_idx_size
        ), f"Index size mismatch: expected {expected_idx_size}, got {actual_idx_size}"
        assert (
            actual_bin_size == expected_bin_size
        ), f"Binary size mismatch: expected {expected_bin_size}, got {actual_bin_size}"

    def test_sequence_extraction(self):
        """Test extracting individual sequences."""
        prefix = os.path.join(self.temp_dir, "test_extract")

        # Create dataset with multiple sequences
        sequences = [[1, 2, 3], [10, 20, 30, 40], [100, 200]]

        builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
        for seq in sequences:
            builder.add_document(seq, lengths=[len(seq)])
        builder.finalize(f"{prefix}.idx")

        # Read index and tokens
        data = read_index_file(f"{prefix}.idx")
        all_tokens = np.fromfile(f"{prefix}.bin", dtype=np.int32)

        # Extract each sequence
        for i, expected_seq in enumerate(sequences):
            start = data["seq_pointers"][i] // 4  # Convert bytes to token index
            length = data["seq_lengths"][i]
            extracted = all_tokens[start : start + length]

            assert np.array_equal(
                extracted, expected_seq
            ), f"Sequence {i} mismatch: expected {expected_seq}, got {extracted.tolist()}"

    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        prefix = os.path.join(self.temp_dir, "test_empty")

        # Create empty dataset
        builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
        builder.finalize(f"{prefix}.idx")

        # Verify
        header = read_index_header(f"{prefix}.idx")
        assert header["num_sequences"] == 0, "Empty dataset should have 0 sequences"
        assert header["num_documents"] == 0, "Empty dataset should have 0 documents"

        # Binary file should be empty
        assert os.path.getsize(f"{prefix}.bin") == 0, "Binary file should be empty"

    def test_large_sequence(self):
        """Test handling of large sequences."""
        prefix = os.path.join(self.temp_dir, "test_large")

        # Create large sequence
        large_seq = list(range(10000))

        builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
        builder.add_document(large_seq, lengths=[len(large_seq)])
        builder.finalize(f"{prefix}.idx")

        # Verify
        data = read_index_file(f"{prefix}.idx")
        assert data["seq_lengths"][0] == 10000, "Large sequence length incorrect"

        # Verify tokens
        saved_tokens = np.fromfile(f"{prefix}.bin", dtype=np.int32)
        assert np.array_equal(saved_tokens, large_seq), "Large sequence tokens mismatch"


def test_pointer_calculation():
    """Test the sequence pointer calculation logic."""
    # Test data
    seq_lengths = [5, 3, 4, 2]  # tokens per sequence
    dtype_size = 4  # int32

    # Calculate pointers
    pointers = []
    current = 0
    for length in seq_lengths:
        pointers.append(current)
        current += length * dtype_size

    expected = [0, 20, 32, 48]
    assert pointers == expected, f"Pointer calculation error: {pointers} != {expected}"


def test_dtype_sizes():
    """Test data type size calculations."""
    assert np.int32().itemsize == 4, "int32 should be 4 bytes"
    assert np.uint16().itemsize == 2, "uint16 should be 2 bytes"
    assert np.int64().itemsize == 8, "int64 should be 8 bytes"


if __name__ == "__main__":
    # Run with pytest if available, otherwise run basic tests
    try:
        import pytest

        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, running basic tests...")
        test_pointer_calculation()
        test_dtype_sizes()
        print("Basic tests passed!")
        print("\nFor full test suite, install pytest: pip install pytest")
