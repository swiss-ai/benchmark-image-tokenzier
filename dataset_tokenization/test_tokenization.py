#!/usr/bin/env python3
"""
Test script to verify the tokenization pipeline works correctly.
This tests on a small subset of images before running on the full dataset.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier')

from utils.indexed_dataset_megatron import IndexedDatasetBuilder, DType, VisionTokenIndexedDatasetBuilder
from Tokenizer.Emu3VisionTokenizer import Emu3VisionTokenizer


def test_tokenizer():
    """Test the Emu3VisionTokenizer on a single image."""
    print("=" * 80)
    print("Testing Emu3VisionTokenizer")
    print("=" * 80)
    
    # Initialize tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = Emu3VisionTokenizer(device=device)
    
    # Load a test image from LLaVA dataset
    test_image_path = "/capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain/00000/000000010.jpg"
    
    if os.path.exists(test_image_path):
        print(f"Loading test image: {test_image_path}")
        img = Image.open(test_image_path).convert('RGB')
        print(f"Image size: {img.size}")
        
        # Test preprocessing
        img_tensor = tokenizer.preprocess(img)
        print(f"Preprocessed tensor shape: {img_tensor.shape}")
        
        # Test encoding
        with torch.no_grad():
            indices, additional_info = tokenizer.encode(img_tensor)
        print(f"Encoded indices shape: {indices.shape}")
        print(f"Number of tokens: {indices.numel()}")
        
        # Test decoding
        with torch.no_grad():
            reconstructed = tokenizer.decode(indices, additional_info)
        print(f"Reconstructed shape: {reconstructed.shape}")
        
        print("✓ Tokenizer test passed!")
    else:
        print(f"Warning: Test image not found at {test_image_path}")
        print("Creating a synthetic test image...")
        img = Image.new('RGB', (256, 256), color='red')
    
    return True


def test_indexed_dataset():
    """Test the IndexedDataset builder using exact Megatron format."""
    print("\n" + "=" * 80)
    print("Testing Megatron IndexedDataset Builder")
    print("=" * 80)
    
    # Test 1: Basic IndexedDatasetBuilder (reference format)
    print("\nTest 1: Basic IndexedDatasetBuilder")
    test_prefix = "/tmp/test_megatron"
    vocab_size = 2**17  # Emu3 vocab size
    
    # Create builder
    builder = IndexedDatasetBuilder(
        bin_path=f"{test_prefix}.bin",
        dtype=DType.optimal_dtype(vocab_size)
    )
    
    # Add test data (simulating tokenized images)
    test_data = [
        list(range(100)),  # Image 1: 100 tokens
        list(range(50, 200)),  # Image 2: 150 tokens
        list(range(1000, 1080)),  # Image 3: 80 tokens
    ]
    
    print("Adding test documents...")
    for i, tokens in enumerate(test_data):
        builder.add_document(tokens, lengths=[len(tokens)])
        print(f"  Added document {i}: {len(tokens)} tokens")
    
    # Finalize
    builder.finalize(f"{test_prefix}.idx")
    print("✓ Basic IndexedDataset created")
    
    # Check file sizes
    bin_size = os.stat(f"{test_prefix}.bin").st_size
    idx_size = os.stat(f"{test_prefix}.idx").st_size
    print(f"Binary file size: {bin_size} bytes")
    print(f"Index file size: {idx_size} bytes")
    
    # Test 2: VisionTokenIndexedDatasetBuilder wrapper
    print("\n\nTest 2: VisionTokenIndexedDatasetBuilder")
    test_prefix2 = "/tmp/test_vision_tokens"
    builder2 = VisionTokenIndexedDatasetBuilder(test_prefix2, vocab_size=vocab_size)
    
    # Add tokenized "images"
    for i, tokens in enumerate(test_data):
        builder2.add_image_tokens(np.array(tokens))
    
    builder2.finalize()
    
    # Clean up
    for prefix in [test_prefix, test_prefix2]:
        for ext in ['.bin', '.idx']:
            if os.path.exists(prefix + ext):
                os.remove(prefix + ext)
    
    print("\n✓ IndexedDataset tests passed!")
    return True


def test_small_dataset():
    """Test the full pipeline on a small subset of images."""
    print("\n" + "=" * 80)
    print("Testing Full Pipeline on Small Dataset")
    print("=" * 80)
    
    # Test with first 10 images
    import glob
    image_dir = "/capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain"
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*/*.jpg")))[:10]
    
    if not image_paths:
        print("No images found for testing")
        return False
    
    print(f"Testing with {len(image_paths)} images")
    
    # Initialize components
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = Emu3VisionTokenizer(device=device)
    
    test_output = "/tmp/test_llava_tokens"
    builder = VisionTokenIndexedDatasetBuilder(test_output, vocab_size=2**17)
    
    # Process images
    total_tokens = 0
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            # Load and tokenize
            img = Image.open(img_path).convert('RGB')
            img_tensor = tokenizer.preprocess(img)
            indices, _ = tokenizer.encode(img_tensor)
            
            # Flatten and add
            flat_indices = indices.squeeze(0).flatten()
            builder.add_image_tokens(flat_indices)
            
            tokens = flat_indices.numel()
            total_tokens += tokens
            print(f"  Image {i+1}: {os.path.basename(img_path)} -> {tokens} tokens")
    
    # Finalize
    builder.finalize()
    
    print(f"\nTotal tokens: {total_tokens}")
    print(f"Average tokens per image: {total_tokens / len(image_paths):.1f}")
    
    # Verify by reading
    reader = IndexedDatasetReader(test_output)
    print(f"Verified: {len(reader)} sequences in dataset")
    
    # Clean up
    for ext in ['.bin', '.idx', '.meta.json']:
        if os.path.exists(test_output + ext):
            os.remove(test_output + ext)
    
    print("✓ Full pipeline test passed!")
    return True


def main():
    """Run all tests."""
    print("Running tokenization pipeline tests...\n")
    
    tests = [
        ("Tokenizer", test_tokenizer),
        ("IndexedDataset", test_indexed_dataset),
        ("Full Pipeline", test_small_dataset),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} test failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n✓ All tests passed! Ready to run full tokenization.")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)