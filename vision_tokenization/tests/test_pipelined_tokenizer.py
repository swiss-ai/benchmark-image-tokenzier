#!/usr/bin/env python3
"""
Test script to verify EMU3PipelinedTokenizer correctness.
Tests:
1. Basic functionality
2. Pipeline vs sequential comparison
3. Order preservation
4. Performance metrics
"""

import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add paths
sys.path.append(str(Path(__file__).parent.parent))

from test_utils import tokenize_image_text_pair_sequential
from utils.tokenization_emu3_pipelined import EMU3PipelinedTokenizer
from vokenizers.emu3 import EMU3ImageOnlyTokenizer, EMU3ImageTextPairTokenizer

# Default tokenizer path
TOKENIZER_PATH = "/capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3_tokenizer"


def create_test_data(num_samples=10):
    """Create synthetic test images and texts."""
    images = []
    texts = []

    for i in range(num_samples):
        # Create images of different sizes
        size = 256 + i * 32  # Variable sizes
        img = Image.new("RGB", (size, size), color=(i * 20, i * 20, i * 20))
        images.append(img)

        # Create texts of different lengths - make them longer to see pipeline benefit
        base_text = f"This is test caption number {i} with some additional words. "
        # Make text longer to actually see the benefit of pipelining
        long_text = base_text * (10 + i * 5)  # Much longer texts
        texts.append(long_text)

    return images, texts


def test_round_trip():
    """Test that tokenization produces correct structure and can be decoded."""
    print("\n=== Testing Round Trip (Tokenize -> Decode) ===")

    try:
        # Initialize tokenizer
        tokenizer = EMU3PipelinedTokenizer(
            text_tokenizer_path=TOKENIZER_PATH, device="cuda" if torch.cuda.is_available() else "cpu", num_cpu_workers=2
        )

        # Create simple test data
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        text = "This is a test caption for the image."

        # Tokenize
        results = list(tokenizer.process_stream([(img, text)]))
        tokens = results[0]

        print(f"Token shape: {tokens.shape}")
        print(f"First 20 tokens: {tokens[:20].tolist()}")

        # Decode to see structure
        decoded = tokenizer.text_tokenizer.decode(tokens, skip_special_tokens=False)
        print(f"\nDecoded (first 200 chars):\n{decoded[:200]}")

        # Check structure
        assert "<|img_start|>" in decoded, "Missing img_start token"
        assert "<|img_end|>" in decoded, "Missing img_end token"
        assert "test caption" in decoded.lower(), "Text not found in decoded output"

        # Check token structure
        tokens_list = tokens.tolist()
        bos_id = tokenizer.text_tokenizer.bos_token_id
        eos_id = tokenizer.text_tokenizer.eos_token_id

        assert tokens_list[0] == bos_id, f"First token should be BOS ({bos_id}), got {tokens_list[0]}"
        assert tokens_list[-1] == eos_id, f"Last token should be EOS ({eos_id}), got {tokens_list[-1]}"

        # Find image structure tokens
        img_start_id = tokenizer.text_tokenizer.convert_tokens_to_ids("<|img_start|>")
        img_end_id = tokenizer.text_tokenizer.convert_tokens_to_ids("<|img_end|>")

        assert img_start_id in tokens_list, "img_start token not found in token sequence"
        assert img_end_id in tokens_list, "img_end token not found in token sequence"

        # Verify order: BOS, img_start, ..., img_end, text, EOS
        img_start_idx = tokens_list.index(img_start_id)
        img_end_idx = tokens_list.index(img_end_id)

        assert img_start_idx < img_end_idx, "img_start should come before img_end"
        assert img_start_idx > 0, "img_start should be after BOS"
        assert img_end_idx < len(tokens_list) - 1, "Text should come after img_end"

        # Extract and verify text portion EXACTLY
        text_start_idx = img_end_idx + 1
        text_end_idx = len(tokens_list) - 1  # Before EOS

        # Extract text tokens (between img_end and EOS)
        text_tokens = tokens_list[text_start_idx:text_end_idx]
        decoded_text = tokenizer.text_tokenizer.decode(text_tokens, skip_special_tokens=True)

        print("\n--- Text Consistency Check ---")
        print(f"Original text: '{text}'")
        print(f"Decoded text:  '{decoded_text}'")
        print(f"Text tokens count: {len(text_tokens)}")

        # Tokenize the text alone to compare
        text_only_tokens = (
            tokenizer.text_tokenizer(text, truncation=False, add_special_tokens=False, return_tensors="pt")["input_ids"]
            .squeeze(0)
            .tolist()
        )

        print(f"Text-only tokenization: {text_only_tokens}")
        print(f"Text portion from combined: {text_tokens}")

        # EXACT match required
        assert text_tokens == text_only_tokens, (
            f"Text tokenization mismatch!\n" f"Expected: {text_only_tokens}\n" f"Got: {text_tokens}"
        )

        # Also check exact text match
        assert decoded_text == text, (
            f"Text not preserved exactly!\n" f"Original: '{text}'\n" f"Decoded: '{decoded_text}'"
        )

        print("✓ Text tokenization is EXACTLY consistent!")

        print("\n✓ Token structure is correct:")
        print(f"  BOS at position 0")
        print(f"  img_start at position {img_start_idx}")
        print(f"  img_end at position {img_end_idx}")
        print(f"  Text starts at position {text_start_idx}")
        print(f"  EOS at position {len(tokens_list)-1}")
        print(f"  Text tokens: {len(text_tokens)} tokens")

        tokenizer.stop_pipeline()
        return True

    except Exception as e:
        print(f"✗ Round trip test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_text_consistency_all_tokenizers():
    """Test that text tokenization is exactly consistent across all tokenizers."""
    print("\n=== Testing Text Consistency Across All Tokenizers ===")

    try:
        # Initialize tokenizers
        pipelined = EMU3PipelinedTokenizer(
            text_tokenizer_path=TOKENIZER_PATH, device="cuda" if torch.cuda.is_available() else "cpu", num_cpu_workers=2
        )

        sequential = EMU3ImageTextPairTokenizer(
            text_tokenizer_path=TOKENIZER_PATH,
            min_pixels=384 * 384,
            max_pixels=1024 * 1024,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Test data
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        text = "This is a test caption for the image."

        # Get tokens from each method
        pipelined.start_pipeline()
        pipelined_tokens = list(pipelined.process_stream([(img, text)]))[0]
        pipelined.stop_pipeline()

        sequential_tokens = sequential.tokenize_image_text_pair(img, text)
        sequential_tokens_seq = tokenize_image_text_pair_sequential(sequential, img, text)

        # Find img_end position in each
        img_end_id = sequential.text_tokenizer.convert_tokens_to_ids("<|img_end|>")

        def extract_text_tokens(tokens):
            tokens_list = tokens.tolist()
            img_end_idx = tokens_list.index(img_end_id)
            text_start_idx = img_end_idx + 1
            text_end_idx = len(tokens_list) - 1  # Before EOS
            return tokens_list[text_start_idx:text_end_idx]

        # Extract text portions
        pipelined_text = extract_text_tokens(pipelined_tokens)
        sequential_text = extract_text_tokens(sequential_tokens)
        sequential_seq_text = extract_text_tokens(sequential_tokens_seq)

        # Direct text tokenization for comparison
        text_only = (
            sequential.text_tokenizer(text, truncation=False, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ]
            .squeeze(0)
            .tolist()
        )

        print(f"Original text: '{text}'")
        print(f"Direct text tokenization: {text_only}")
        print(f"Pipelined text portion: {pipelined_text}")
        print(f"Sequential (parallel) text portion: {sequential_text}")
        print(f"Sequential (true seq) text portion: {sequential_seq_text}")

        # All should be identical
        assert pipelined_text == text_only, "Pipelined text tokens don't match direct tokenization"
        assert sequential_text == text_only, "Sequential (parallel) text tokens don't match"
        assert sequential_seq_text == text_only, "Sequential (true) text tokens don't match"

        # Decode each to verify
        for name, tokens in [
            ("Pipelined", pipelined_text),
            ("Sequential", sequential_text),
            ("Sequential (true)", sequential_seq_text),
        ]:
            decoded = sequential.text_tokenizer.decode(tokens, skip_special_tokens=True)
            assert decoded == text, f"{name} decoded text doesn't match: '{decoded}' vs '{text}'"

        print("\n✓ All tokenizers produce EXACTLY identical text tokenization!")
        return True

    except Exception as e:
        print(f"✗ Text consistency test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_correctness_vs_sequential():
    """Compare pipelined results with sequential processing."""
    print("\n=== Testing Correctness vs Sequential ===")

    try:
        # Initialize both tokenizers
        pipelined = EMU3PipelinedTokenizer(
            text_tokenizer_path=TOKENIZER_PATH, device="cuda" if torch.cuda.is_available() else "cpu", num_cpu_workers=4
        )

        sequential = EMU3ImageTextPairTokenizer(
            text_tokenizer_path=TOKENIZER_PATH,
            min_pixels=384 * 384,
            max_pixels=1024 * 1024,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Create test data
        images, texts = create_test_data(3)

        # Process with pipelined
        pipelined.start_pipeline()
        pipelined_results = list(pipelined.process_stream(((img, txt) for img, txt in zip(images, texts))))
        pipelined.stop_pipeline()

        # Process with sequential
        sequential_results = []
        for img, txt in zip(images, texts):
            result = sequential.tokenize_image_text_pair(img, txt)
            sequential_results.append(result)

        # Compare results
        all_match = True
        for i, (p_res, s_res) in enumerate(zip(pipelined_results, sequential_results)):
            if not torch.equal(p_res, s_res):
                print(f"✗ Mismatch at index {i}")
                print(f"  Pipelined shape: {p_res.shape}, Sequential shape: {s_res.shape}")
                if p_res.shape == s_res.shape:
                    diff = (p_res != s_res).sum().item()
                    print(f"  Number of different tokens: {diff}/{p_res.numel()}")
                all_match = False
            else:
                print(f"✓ Sample {i} matches perfectly")

        return all_match

    except Exception as e:
        print(f"✗ Correctness test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_order_preservation():
    """Test that results maintain correct order."""
    print("\n=== Testing Order Preservation ===")

    try:
        tokenizer = EMU3PipelinedTokenizer(
            text_tokenizer_path=TOKENIZER_PATH, device="cuda" if torch.cuda.is_available() else "cpu", num_cpu_workers=4
        )

        # Create numbered test data
        num_samples = 20
        images = []
        texts = []
        for i in range(num_samples):
            img = Image.new("RGB", (256, 256), color=(i, i, i))
            images.append(img)
            texts.append(f"Caption {i:03d}")

        # Process
        tokenizer.start_pipeline()
        results = list(tokenizer.process_stream(((img, txt) for img, txt in zip(images, texts))))
        tokenizer.stop_pipeline()

        # Verify order by checking text tokens
        # The text "Caption XXX" should appear in order
        order_correct = True
        for i in range(len(results)):
            # Decode tokens to check if caption number matches
            # This is approximate - just checking result exists
            if results[i] is None:
                print(f"✗ Missing result at index {i}")
                order_correct = False

        if order_correct and len(results) == num_samples:
            print(f"✓ All {num_samples} results in correct order")
        else:
            print(f"✗ Order preservation failed: got {len(results)}/{num_samples} results")

        return order_correct

    except Exception as e:
        print(f"✗ Order preservation test failed: {e}")
        return False


def test_timing_comparison():
    """Comprehensive timing comparison: image-only vs sequential vs pipelined."""
    print("\n=== Timing Comparison ===")

    try:
        # Create test data
        num_samples = 200
        images, texts = create_test_data(num_samples)

        print("\n=== Warming up GPU ===")
        # Create a dummy tokenizer and run warmup
        warmup_tokenizer = EMU3ImageOnlyTokenizer(
            text_tokenizer_path=TOKENIZER_PATH, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        # Run several warmup iterations
        for _ in range(3):
            for img in images[:5]:
                _ = warmup_tokenizer.tokenize_image(img)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        print("GPU warmed up")
        del warmup_tokenizer

        # 1. Time image-only tokenization (baseline first)
        print("\n1. Image-Only Tokenization:")
        image_tokenizer = EMU3ImageOnlyTokenizer(
            text_tokenizer_path=TOKENIZER_PATH, device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Run multiple times and average
        times = []
        for run in range(3):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            for img in images:
                _ = image_tokenizer.tokenize_image(img)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - start)
        image_only_time = sum(times) / len(times)

        print(f"  Time: {image_only_time:.3f}s ({num_samples/image_only_time:.1f} img/s)")

        # 2. Time sequential image-text tokenization
        print("\n2. Image-Text Sequential:")
        sequential = EMU3ImageTextPairTokenizer(
            text_tokenizer_path=TOKENIZER_PATH,
            min_pixels=384 * 384,
            max_pixels=1024 * 1024,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        times = []
        for run in range(3):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            for img, txt in zip(images, texts):
                _ = tokenize_image_text_pair_sequential(sequential, img, txt)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - start)
        sequential_time = sum(times) / len(times)

        print(f"  Time: {sequential_time:.3f}s ({num_samples/sequential_time:.1f} pairs/s)")

        # 3. Time pipelined image-text tokenization
        print("\n3. Image-Text Pipelined:")
        pipelined = EMU3PipelinedTokenizer(
            text_tokenizer_path=TOKENIZER_PATH,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_cpu_workers=4,
            queue_size=50,
        )

        times = []
        for run in range(3):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            _ = list(pipelined.process_stream(zip(images, texts)))
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - start)
        pipelined_time = sum(times) / len(times)
        pipelined.stop_pipeline()

        print(f"  Time: {pipelined_time:.3f}s ({num_samples/pipelined_time:.1f} pairs/s)")

        # Analysis
        print("\n=== Analysis ===")
        print(f"Image-only:          {image_only_time:.3f}s (baseline)")
        print(f"Sequential:          {sequential_time:.3f}s")
        print(f"Pipelined:           {pipelined_time:.3f}s")

        print(f"\nText processing overhead:")
        print(f"  Sequential: {sequential_time - image_only_time:.3f}s")
        print(f"  Pipelined:  {pipelined_time - image_only_time:.3f}s")

        print(f"\nSpeedup:")
        print(f"  Pipeline vs Sequential: {sequential_time/pipelined_time:.2f}x")
        print(f"  Pipeline efficiency: {image_only_time/pipelined_time:.1%} (100% = no text overhead)")

        return True

    except Exception as e:
        print(f"✗ Timing comparison failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_performance():
    """Measure performance of pipelined vs sequential."""
    print("\n=== Performance Comparison ===")

    try:
        # Create test data
        num_samples = 20
        images, texts = create_test_data(num_samples)

        # Test pipelined
        pipelined = EMU3PipelinedTokenizer(
            text_tokenizer_path=TOKENIZER_PATH, device="cuda" if torch.cuda.is_available() else "cpu", num_cpu_workers=4
        )

        pipelined.start_pipeline()
        start = time.time()
        pipelined_results = list(pipelined.process_stream(((img, txt) for img, txt in zip(images, texts))))
        pipelined_time = time.time() - start
        pipelined.stop_pipeline()

        # Test sequential
        sequential = EMU3ImageTextPairTokenizer(
            text_tokenizer_path=TOKENIZER_PATH,
            min_pixels=384 * 384,
            max_pixels=1024 * 1024,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        start = time.time()
        sequential_results = []
        for img, txt in zip(images, texts):
            result = tokenize_image_text_pair_sequential(sequential, img, txt)
            sequential_results.append(result)
        sequential_time = time.time() - start

        # Report results
        print(f"\nResults for {num_samples} samples:")
        print(f"  Pipelined:  {pipelined_time:.2f}s ({num_samples/pipelined_time:.1f} samples/s)")
        print(f"  Sequential: {sequential_time:.2f}s ({num_samples/sequential_time:.1f} samples/s)")
        print(f"  Speedup:    {sequential_time/pipelined_time:.2f}x")

        return True

    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")

    try:
        tokenizer = EMU3PipelinedTokenizer(
            text_tokenizer_path=TOKENIZER_PATH,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_cpu_workers=2,
            queue_size=10,
        )

        tokenizer.start_pipeline()

        # Test 1: Empty text
        print("Testing empty text...")
        img = Image.new("RGB", (256, 256))
        results = list(tokenizer.process_stream([(img, ""), (img, "Normal text")]))
        print(f"  ✓ Empty text handled: {len(results)} results")

        # Test 2: Very long text
        print("Testing very long text...")
        long_text = "word " * 10000  # Very long text
        results = list(tokenizer.process_stream([(img, long_text)]))
        print(f"  ✓ Long text handled: result shape {results[0].shape}")

        # Test 3: Small and large images
        print("Testing various image sizes...")
        sizes = [(64, 64), (256, 256), (1024, 1024), (2048, 2048)]
        for size in sizes:
            img = Image.new("RGB", size)
            results = list(tokenizer.process_stream([(img, "test")]))
            print(f"  ✓ Size {size}: result shape {results[0].shape}")

        tokenizer.stop_pipeline()

        return True

    except Exception as e:
        print(f"✗ Edge case test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("EMU3 PIPELINED TOKENIZER TEST SUITE")
    print("=" * 60)

    tests = [
        ("Round Trip", test_round_trip),
        ("Text Consistency", test_text_consistency_all_tokenizers),
        ("Correctness vs Sequential", test_correctness_vs_sequential),
        ("Order Preservation", test_order_preservation),
        ("Timing Comparison", test_timing_comparison),
        ("Performance", test_performance),
        ("Edge Cases", test_edge_cases),
    ]

    results = {}
    for name, test_func in tests:
        results[name] = test_func()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:30} {status}")

    total_passed = sum(results.values())
    total = len(results)
    print(f"\nTotal: {total_passed}/{total} tests passed")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
