#!/usr/bin/env python3
"""
End-to-end integration tests for the vision tokenization pipeline.

This module tests the complete workflow from raw images to IndexedDataset files,
ensuring all components work together correctly. It validates the integration
between WebDataset, Emu3 vision tokenizer, and IndexedDataset builder.

Focus areas:
- WebDataset creation and loading from PIL images
- Image preprocessing and tokenization with Emu3VisionTokenizer
- Full pipeline validation (images → tokens → IndexedDataset)
- Tokenization consistency (same image produces same tokens)
- Component integration and data flow verification
- Multimodal vocabulary offset handling in practice

This is the integration test suite that ensures the entire vision tokenization
system works as expected. For format tests see test_indexed_dataset_format.py,
and for data integrity tests see test_indexed_dataset_integrity.py.

Requirements:
- CUDA-capable GPU (tests will use CPU if GPU unavailable)
- Emu3VisionTokenizer model files

Run with:
    pytest test_vision_pipeline_integration.py -v
    python -m pytest test_vision_pipeline_integration.py -v
"""

import sys
import os
import json
import struct
import tempfile
import shutil
import numpy as np
import torch
import webdataset as wds
from PIL import Image
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vision_tokenization.utils.indexed_dataset_megatron import VisionTokenIndexedDatasetBuilder
from Tokenizer.Emu3VisionTokenizer import Emu3VisionTokenizer
from test_utils import read_index_header, read_index_file, calculate_expected_pointers


class TestVisionTokenizationPipeline:
    """Test the complete vision tokenization pipeline."""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_vision_pipeline_")
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        print(f"Initializing Emu3VisionTokenizer on {cls.device}...")
        cls.tokenizer = Emu3VisionTokenizer(device=cls.device)
        
    @classmethod
    def teardown_class(cls):
        """Cleanup."""
        shutil.rmtree(cls.temp_dir)
        
    def create_test_images(self, num_images=5, sizes=None):
        """Create test images with different sizes."""
        if sizes is None:
            sizes = [(224, 224), (256, 256), (384, 384), (512, 512), (640, 480)]
        
        images = []
        for i in range(num_images):
            size = sizes[i % len(sizes)]
            # Create image with gradient pattern
            img_array = np.zeros((*size, 3), dtype=np.uint8)
            img_array[:, :, 0] = np.linspace(0, 255, size[0])[:, np.newaxis]  # Red gradient
            img_array[:, :, 1] = np.linspace(0, 255, size[1])[np.newaxis, :]  # Green gradient
            img_array[:, :, 2] = (i * 50) % 255  # Blue constant
            
            img = Image.fromarray(img_array)
            images.append(img)
        
        return images
    
    def test_webdataset_creation(self):
        """Test creating WebDataset from images."""
        # Create test images
        images = self.create_test_images(10)
        
        # Save as WebDataset
        shard_path = os.path.join(self.temp_dir, "test_shard.tar")
        with wds.TarWriter(shard_path) as sink:
            for i, img in enumerate(images):
                key = f"{i:06d}"
                
                # Save image to bytes
                import io
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                
                sample = {
                    "__key__": key,
                    "png": img_bytes.getvalue(),
                    "json": json.dumps({
                        "index": i,
                        "size": img.size,
                        "mode": img.mode
                    }).encode()
                }
                sink.write(sample)
        
        # Verify we can read it back
        dataset = wds.WebDataset(shard_path, shardshuffle=False).decode("pil").to_tuple("png", "json")
        loaded_images = list(dataset)
        
        assert len(loaded_images) == 10, f"Expected 10 images, got {len(loaded_images)}"
        
        # Check first image
        img, meta = loaded_images[0]
        # meta is already a dict when decoded by webdataset
        assert meta["index"] == 0
        assert isinstance(img, Image.Image)
        
        print(f"✓ WebDataset creation test passed")
        
    def test_tokenization_consistency(self):
        """Test that tokenization is consistent for the same image."""
        # Create test image
        img = self.create_test_images(1)[0]
        
        # Tokenize multiple times
        tokens_list = []
        for _ in range(3):
            with torch.no_grad():
                img_tensor = self.tokenizer.preprocess(img)
                indices, _ = self.tokenizer.encode(img_tensor)
                tokens = indices.squeeze(0).flatten().cpu().numpy()
                tokens_list.append(tokens)
        
        # Check all tokenizations are identical
        for i in range(1, len(tokens_list)):
            assert np.array_equal(tokens_list[0], tokens_list[i]), \
                f"Tokenization {i} doesn't match the first"
        
        print(f"✓ Tokenization consistency test passed")
        
    def test_multimodal_offset(self):
        """Test multimodal tokenizer offset application."""
        # Create small test tokens
        test_tokens = np.array([100, 200, 300], dtype=np.int32)
        text_vocab_size = 131072
        image_vocab_size = 32768
        
        # Create dataset with offset
        prefix = os.path.join(self.temp_dir, "test_multimodal")
        builder = VisionTokenIndexedDatasetBuilder(
            output_prefix=prefix,
            image_vocab_size=image_vocab_size,
            text_vocab_size=text_vocab_size
        )
        
        builder.add_image_tokens(test_tokens)
        builder.finalize()
        
        # Read back and verify offset
        saved_tokens = np.fromfile(f"{prefix}.bin", dtype=np.int32)
        expected = test_tokens + text_vocab_size
        
        assert np.array_equal(saved_tokens, expected), \
            f"Expected {expected.tolist()}, got {saved_tokens.tolist()}"
        
        print(f"✓ Multimodal offset test passed")
        
    def test_full_pipeline(self):
        """Test the complete pipeline from images to IndexedDataset."""
        # Create test images
        images = self.create_test_images(5)
        
        # Save as WebDataset
        shard_path = os.path.join(self.temp_dir, "pipeline_test.tar")
        with wds.TarWriter(shard_path) as sink:
            for i, img in enumerate(images):
                import io
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                
                sample = {
                    "__key__": f"{i:06d}",
                    "png": img_bytes.getvalue(),
                    "json": json.dumps({"index": i}).encode()
                }
                sink.write(sample)
        
        # Tokenize all images
        dataset = wds.WebDataset(shard_path, shardshuffle=False).decode("pil").to_tuple("png", "json")
        
        original_tokens = []
        output_prefix = os.path.join(self.temp_dir, "pipeline_output")
        
        builder = VisionTokenIndexedDatasetBuilder(
            output_prefix=output_prefix,
            image_vocab_size=32768,
            text_vocab_size=131072
        )
        
        # Process images
        for img, _ in dataset:
            with torch.no_grad():
                img_tensor = self.tokenizer.preprocess(img)
                indices, _ = self.tokenizer.encode(img_tensor)
                tokens = indices.squeeze(0).flatten().cpu().numpy()
                original_tokens.append(tokens)
                builder.add_image_tokens(tokens)
        
        builder.finalize()
        
        # Verify the saved dataset
        assert os.path.exists(f"{output_prefix}.idx"), "Index file not created"
        assert os.path.exists(f"{output_prefix}.bin"), "Binary file not created"
        
        # Read and verify tokens
        saved_tokens_raw = np.fromfile(f"{output_prefix}.bin", dtype=np.int32)
        saved_tokens = saved_tokens_raw - 131072  # Remove offset
        
        # Read index to get sequence boundaries
        with open(f"{output_prefix}.idx", "rb") as f:
            # Skip header
            f.seek(34)
            
            # Read sequence lengths
            seq_lengths = np.frombuffer(f.read(5 * 4), dtype=np.int32)
            
        # Extract and compare each sequence
        offset = 0
        for i, orig_tokens in enumerate(original_tokens):
            saved_seq = saved_tokens[offset:offset + seq_lengths[i]]
            assert np.array_equal(orig_tokens, saved_seq), \
                f"Sequence {i} mismatch"
            offset += seq_lengths[i]
        
        print(f"✓ Full pipeline test passed")
        
    def test_variable_sequence_lengths(self):
        """Test handling of variable-length sequences."""
        # Create sequences of different lengths
        sequences = [
            np.arange(100, dtype=np.int32),
            np.arange(200, dtype=np.int32),
            np.arange(150, dtype=np.int32),
            np.arange(300, dtype=np.int32),
        ]
        
        prefix = os.path.join(self.temp_dir, "test_variable")
        builder = VisionTokenIndexedDatasetBuilder(
            output_prefix=prefix,
            image_vocab_size=32768,
            text_vocab_size=0  # No offset for this test
        )
        
        for seq in sequences:
            builder.add_image_tokens(seq)
        
        builder.finalize()
        
        # Verify sequence pointers
        data = read_index_file(f"{prefix}.idx")
        dtype_size = data["dtype_size"]
        seq_lengths = data["seq_lengths"]
        seq_pointers = data["seq_pointers"]
        
        # Check pointers are cumulative using correct dtype size
        expected_pointers = calculate_expected_pointers([len(seq) for seq in sequences], dtype_size)
        
        assert np.array_equal(seq_pointers, expected_pointers), \
            f"Sequence pointers incorrect. Expected: {expected_pointers}, Got: {seq_pointers.tolist()}"
        
        print(f"✓ Variable sequence length test passed (dtype size: {dtype_size} bytes)")
        
    def test_index_file_structure(self):
        """Test the exact structure of the index file."""
        # Create a simple dataset
        prefix = os.path.join(self.temp_dir, "test_structure")
        builder = VisionTokenIndexedDatasetBuilder(
            output_prefix=prefix,
            image_vocab_size=32768,
            text_vocab_size=131072
        )
        
        # Add 3 sequences
        for i in range(3):
            tokens = np.array([i * 100 + j for j in range(10)], dtype=np.int32)
            builder.add_image_tokens(tokens)
        
        builder.finalize()
        
        # Read and verify structure
        data = read_index_file(f"{prefix}.idx")
        
        # Verify header
        assert data["magic"] == b"MMIDIDX\x00\x00", "Invalid magic"
        assert data["version"] == 1, "Invalid version"
        assert data["dtype_code"] == 4, "Invalid dtype code"
        assert data["num_sequences"] == 3, "Invalid sequence count"
        assert data["num_documents"] == 3, "Invalid document count"
        
        # Verify arrays
        assert np.array_equal(data["seq_lengths"], [10, 10, 10]), "Invalid lengths"
        assert np.array_equal(data["seq_pointers"], [0, 40, 80]), "Invalid pointers"
        assert np.array_equal(data["doc_indices"], [0, 1, 2, 3]), "Invalid doc indices"
        
        print(f"✓ Index file structure test passed")


def test_compare_with_reference():
    """Test against a known reference case."""
    # Create a known sequence
    reference_tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        prefix = os.path.join(temp_dir, "reference")
        
        # Create using our builder
        from vision_tokenization.utils.indexed_dataset_megatron import IndexedDatasetBuilder
        
        builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
        builder.add_document(reference_tokens.tolist(), lengths=[5])
        builder.finalize(f"{prefix}.idx")
        
        # Verify files
        idx_size = os.path.getsize(f"{prefix}.idx")
        bin_size = os.path.getsize(f"{prefix}.bin")
        
        # Expected sizes
        expected_idx = 34 + 4 + 8 + 16  # header + 1 length + 1 pointer + 2 doc indices
        expected_bin = 5 * 4  # 5 tokens * 4 bytes
        
        assert idx_size == expected_idx, f"Index size: expected {expected_idx}, got {idx_size}"
        assert bin_size == expected_bin, f"Binary size: expected {expected_bin}, got {bin_size}"
        
        # Verify content
        saved_tokens = np.fromfile(f"{prefix}.bin", dtype=np.int32)
        assert np.array_equal(saved_tokens, reference_tokens), "Tokens mismatch"
    
    print(f"✓ Reference comparison test passed")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])