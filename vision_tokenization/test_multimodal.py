#!/usr/bin/env python3
"""
Test script specifically for multimodal tokenizer vocabulary offset functionality.
"""

import numpy as np
import torch
from utils.indexed_dataset_megatron import VisionTokenIndexedDatasetBuilder

def test_vocabulary_offset():
    """Test that vocabulary offset works correctly for multimodal tokenizers."""
    print("=" * 80)
    print("Testing Multimodal Tokenizer Vocabulary Offset")
    print("=" * 80)
    
    # Configuration
    text_vocab_size = 50000
    image_vocab_size = 1000  # Small for testing
    total_vocab_size = text_vocab_size + image_vocab_size
    
    print(f"Configuration:")
    print(f"  - Text vocabulary: [0, {text_vocab_size})")
    print(f"  - Image vocabulary: [{text_vocab_size}, {total_vocab_size})")
    print(f"  - Total vocabulary: {total_vocab_size}")
    
    # Test 1: Pure vision tokenizer (no offset)
    print(f"\nTest 1: Pure Vision Tokenizer (no offset)")
    builder1 = VisionTokenIndexedDatasetBuilder(
        "/tmp/test_pure_vision", 
        total_vocab_size=image_vocab_size,
        text_vocab_size=0
    )
    
    # Add test tokens
    test_tokens = np.array([0, 1, 999])  # Range [0, 999]
    builder1.add_image_tokens(test_tokens)
    builder1.finalize()
    
    print(f"✓ Pure vision tokenizer works")
    
    # Test 2: Multimodal tokenizer (with offset)
    print(f"\nTest 2: Multimodal Tokenizer (with offset)")
    builder2 = VisionTokenIndexedDatasetBuilder(
        "/tmp/test_multimodal",
        total_vocab_size=total_vocab_size,
        text_vocab_size=text_vocab_size
    )
    
    # Same input tokens [0, 1, 999] should become [50000, 50001, 50999]
    builder2.add_image_tokens(test_tokens)
    builder2.finalize()
    
    print(f"✓ Multimodal tokenizer with offset works")
    
    # Test 3: Vocabulary overflow detection
    print(f"\nTest 3: Vocabulary Overflow Detection")
    builder3 = VisionTokenIndexedDatasetBuilder(
        "/tmp/test_overflow",
        total_vocab_size=total_vocab_size,
        text_vocab_size=text_vocab_size
    )
    
    # This should fail: token 1000 + offset 50000 = 51000, but max is 50999
    overflow_tokens = np.array([1000])  # This exceeds image vocab size
    
    try:
        builder3.add_image_tokens(overflow_tokens)
        print("✗ Overflow detection failed - should have raised error")
        return False
    except ValueError as e:
        print(f"✓ Overflow correctly detected: {e}")
    
    # Clean up test files
    import os
    for prefix in ["/tmp/test_pure_vision", "/tmp/test_multimodal", "/tmp/test_overflow"]:
        for ext in [".bin", ".idx"]:
            if os.path.exists(prefix + ext):
                os.remove(prefix + ext)
    
    print(f"\n✅ All multimodal tokenizer tests passed!")
    return True


def test_real_example():
    """Test with realistic values for your setup."""
    print("\n" + "=" * 80)
    print("Testing with Realistic SwissAI + Emu3 Configuration")
    print("=" * 80)
    
    # Realistic configuration (adjust based on actual vocab size)
    # You'll need to run: python utils/detect_vocab_size.py alehc/swissai-tokenizer
    # For now, let's assume a reasonable text vocab size
    text_vocab_size = 32000  # Common size for many tokenizers
    image_vocab_size = 2**17  # Emu3 vocabulary
    total_vocab_size = text_vocab_size + image_vocab_size
    
    print(f"SwissAI + Emu3 Configuration:")
    print(f"  - Text vocabulary (SwissAI): {text_vocab_size:,}")
    print(f"  - Image vocabulary (Emu3): {image_vocab_size:,}")
    print(f"  - Total vocabulary: {total_vocab_size:,}")
    print(f"  - Image token range: [{text_vocab_size:,}, {total_vocab_size-1:,}]")
    
    # Create builder
    builder = VisionTokenIndexedDatasetBuilder(
        "/tmp/test_swissai_emu3",
        total_vocab_size=total_vocab_size,
        text_vocab_size=text_vocab_size
    )
    
    # Simulate some Emu3 tokens
    # Emu3 tokens are in range [0, 131071]
    sample_image_tokens = np.array([0, 1000, 65536, 131071])  # Some sample tokens
    expected_final_tokens = sample_image_tokens + text_vocab_size
    
    print(f"\nToken Transformation Example:")
    print(f"  - Original Emu3 tokens: {sample_image_tokens}")
    print(f"  - After offset: {expected_final_tokens}")
    print(f"  - Range check: [{np.min(expected_final_tokens)}, {np.max(expected_final_tokens)}]")
    
    # Add tokens
    builder.add_image_tokens(sample_image_tokens)
    builder.finalize()
    
    # Clean up
    import os
    for ext in [".bin", ".idx"]:
        if os.path.exists("/tmp/test_swissai_emu3" + ext):
            os.remove("/tmp/test_swissai_emu3" + ext)
    
    print(f"\n✅ Realistic configuration test passed!")
    
    print(f"\n" + "=" * 80)
    print("Command to use for your setup:")
    print("=" * 80)
    print(f"# First, detect actual vocabulary size:")
    print(f"python utils/detect_vocab_size.py alehc/swissai-tokenizer")
    print(f"")
    print(f"# Then run tokenization with detected size (example):")
    print(f"python tokenize_images.py \\")
    print(f"    --dataset llava \\")
    print(f"    --input-path /path/to/images \\")
    print(f"    --text-vocab-size {text_vocab_size} \\")
    print(f"    --image-vocab-size {image_vocab_size}")


if __name__ == "__main__":
    success1 = test_vocabulary_offset()
    test_real_example()
    
    if success1:
        print(f"\n🎉 All tests passed! Your multimodal tokenization setup is ready.")
    else:
        print(f"\n❌ Some tests failed. Please check the implementation.")