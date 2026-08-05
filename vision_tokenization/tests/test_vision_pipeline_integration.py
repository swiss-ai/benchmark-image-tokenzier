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

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import webdataset as wds
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Tokenizer.Emu3VisionTokenizer import Emu3VisionTokenizer

from .test_utils import calculate_expected_pointers, read_index_file


class TestVisionTokenizationPipeline:
    """Test the complete vision tokenization pipeline."""

    @classmethod
    def setup_class(cls):
        """Setup test environment."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_vision_pipeline_")
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                img.save(img_bytes, format="PNG")

                sample = {
                    "__key__": key,
                    "png": img_bytes.getvalue(),
                    "json": json.dumps({"index": i, "size": img.size, "mode": img.mode}).encode(),
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
            assert np.array_equal(tokens_list[0], tokens_list[i]), f"Tokenization {i} doesn't match the first"

        print(f"✓ Tokenization consistency test passed")
