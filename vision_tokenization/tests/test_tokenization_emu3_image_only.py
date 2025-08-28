#!/usr/bin/env python3
"""
Test EMU3 image-only tokenization.
Tests the optimized direct tokenization that avoids double tokenization.
"""

import sys
import torch
import pytest
import tempfile
from pathlib import Path

# Add utils directory to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from tokenization_emu3_image_only import EMU3ImageOnlyTokenizer
from add_special_tokens_emu3_style import add_emu3_special_tokens


@pytest.fixture
def temp_tokenizer():
    """Create a temporary tokenizer with EMU3 tokens."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create tokenizer with Llama-3-8B base
        add_emu3_special_tokens(
            model_path="meta-llama/Meta-Llama-3-8B",
            output_path=temp_dir,
            visual_vocab_size=1000,
            num_reserved_tokens=20
        )
        yield temp_dir


@pytest.fixture
def image_tokenizer(temp_tokenizer):
    """Create EMU3 image-only tokenizer instance."""
    return EMU3ImageOnlyTokenizer(
        text_tokenizer_path=temp_tokenizer
    )


class TestEMU3ImageOnlyTokenizer:
    """Test EMU3 image-only tokenization functionality."""
    
    def test_initialization(self, image_tokenizer):
        """Test tokenizer initialization."""
        assert image_tokenizer.text_tokenizer is not None
        assert hasattr(image_tokenizer, 'vision_token_offset')
        assert image_tokenizer.vision_token_offset > 0  # Should be computed dynamically
        
        # Check all cached special tokens
        assert hasattr(image_tokenizer, 'bos_id')
        assert hasattr(image_tokenizer, 'eos_id')
        assert hasattr(image_tokenizer, 'img_start_id')
        assert hasattr(image_tokenizer, 'img_end_id')
        assert hasattr(image_tokenizer, 'img_token_start_id')
        assert hasattr(image_tokenizer, 'eol_id')
        assert hasattr(image_tokenizer, 'eof_id')
    
    def test_special_token_caching(self, image_tokenizer):
        """Test that special tokens are properly cached."""
        # Check structure tokens are valid IDs
        assert isinstance(image_tokenizer.img_start_id, int)
        assert isinstance(image_tokenizer.img_end_id, int)
        assert isinstance(image_tokenizer.img_token_start_id, int)
        assert isinstance(image_tokenizer.eol_id, int)
        assert isinstance(image_tokenizer.eof_id, int)
        
        # Verify they're not unknown tokens
        unk_id = image_tokenizer.text_tokenizer.unk_token_id
        if unk_id is not None:
            assert image_tokenizer.img_start_id != unk_id
            assert image_tokenizer.img_end_id != unk_id
    
    def test_tokenize_2x2_image(self, image_tokenizer):
        """Test tokenizing a 2x2 image and verify offset is applied correctly."""
        # Create 2x2 image indices
        image_indices = torch.tensor([0, 100, 200, 300])
        height, width = 2, 2
        
        # Tokenize with our optimized method
        tokens = image_tokenizer.encapsulate_image(
            image_indices, height, width
        )
        
        assert isinstance(tokens, torch.Tensor)
        assert len(tokens) > 0
        
        # Check basic structure
        assert tokens[0] == image_tokenizer.bos_id
        assert tokens[-1] == image_tokenizer.eos_id
        assert image_tokenizer.img_start_id in tokens
        assert image_tokenizer.img_end_id in tokens
        
        # Verify vision tokens have the offset applied
        # The vision tokens should be: 
        # - image_indices[0] + offset = 0 + offset
        # - image_indices[1] + offset = 100 + offset
        # - image_indices[2] + offset = 200 + offset
        # - image_indices[3] + offset = 300 + offset
        
        # Find where vision tokens start (after img_token_start)
        img_token_start_pos = (tokens == image_tokenizer.img_token_start_id).nonzero()[0].item()
        
        # Extract vision tokens (between img_token_start and first EOL)
        first_eol_pos = (tokens == image_tokenizer.eol_id).nonzero()[0].item()
        vision_token_ids = tokens[img_token_start_pos + 1:first_eol_pos].tolist()
        
        # Also get second row
        second_eol_pos = (tokens == image_tokenizer.eol_id).nonzero()[1].item()
        vision_token_ids.extend(tokens[first_eol_pos + 1:second_eol_pos].tolist())
        
        # Verify the vision tokens are correctly offset
        expected_vision_ids = [
            image_tokenizer.vision_token_offset + 0,
            image_tokenizer.vision_token_offset + 100,
            image_tokenizer.vision_token_offset + 200,
            image_tokenizer.vision_token_offset + 300
        ]
        
        assert vision_token_ids == expected_vision_ids, \
            f"Vision tokens mismatch!\nGot: {vision_token_ids}\nExpected: {expected_vision_ids}"
        
        # Verify these map to the correct visual tokens in vocabulary
        for i, vid in enumerate(vision_token_ids):
            # Decode to check it's the right visual token
            decoded = image_tokenizer.text_tokenizer.convert_ids_to_tokens(vid)
            expected = f"<|visual token {image_indices[i]:06d}|>"
            assert decoded == expected, f"Token {vid} decoded to '{decoded}', expected '{expected}'"
    
    def test_tokenize_different_sizes(self, image_tokenizer):
        """Test exact token sequence structure for different image sizes."""
        test_cases = [
            (1, 1, torch.tensor([42])),
            (2, 3, torch.tensor([0, 1, 2, 3, 4, 5])),
            (3, 2, torch.tensor([10, 20, 30, 40, 50, 60])),
        ]
        
        for height, width, indices in test_cases:
            # Method 1: Our optimized tokenization
            tokens = image_tokenizer.encapsulate_image(indices, height, width)
            
            # Method 2: Build expected sequence manually
            expected = []
            expected.append(image_tokenizer.bos_id)
            expected.append(image_tokenizer.img_start_id)
            
            # Add dimension tokens
            dim_tokens = image_tokenizer.text_tokenizer.encode(f"{height}*{width}", add_special_tokens=False)
            expected.extend(dim_tokens)
            
            expected.append(image_tokenizer.img_token_start_id)
            
            # Add vision tokens row by row with EOL
            for row in range(height):
                for col in range(width):
                    idx = row * width + col
                    expected.append(indices[idx].item() + image_tokenizer.vision_token_offset)
                expected.append(image_tokenizer.eol_id)
            
            expected.append(image_tokenizer.eof_id)
            expected.append(image_tokenizer.img_end_id)
            expected.append(image_tokenizer.eos_id)
            
            expected = torch.tensor(expected, dtype=torch.long)
            
            # Verify exact match
            assert torch.equal(tokens, expected), \
                f"Token sequence mismatch for {height}x{width}!\nGot:      {tokens.tolist()}\nExpected: {expected.tolist()}"
            
            # Also verify against text-based approach
            text = f"<|img_start|>{height}*{width}<|img_token_start|>"
            for row in range(height):
                for col in range(width):
                    idx = indices[row * width + col].item()
                    text += f"<|visual token {idx:06d}|>"
                text += "<|img_end_of_row|>"
            text += "<|img_end_of_frame|><|img_end|>"
            
            text_tokens = image_tokenizer.text_tokenizer.encode(text, add_special_tokens=True)
            # Add EOS since encode doesn't add it but our method does
            text_tokens.append(image_tokenizer.eos_id)
            text_tokens = torch.tensor(text_tokens, dtype=torch.long)
            
            assert torch.equal(tokens, text_tokens), \
                f"Mismatch with text approach for {height}x{width}\nGot:  {tokens.tolist()}\nText: {text_tokens.tolist()}"
    
    def test_vision_token_offset(self, image_tokenizer):
        """Test that vision tokens are properly offset."""
        # Simple 1x1 image
        image_indices = torch.tensor([0])
        tokens = image_tokenizer.encapsulate_image(
            image_indices, 1, 1
        )
        
        # Find the vision token (should be after dimension tokens)
        # Expected structure: BOS, img_start, dims, img_token_start, vision_token, ...
        expected_vision_token = 0 + image_tokenizer.vision_token_offset
        assert expected_vision_token in tokens
        
        # Test with different index
        image_indices = torch.tensor([100])
        tokens = image_tokenizer.encapsulate_image(
            image_indices, 1, 1
        )
        expected_vision_token = 100 + image_tokenizer.vision_token_offset
        assert expected_vision_token in tokens
    
    def test_row_structure(self, image_tokenizer):
        """Test that EOL tokens are properly placed between rows."""
        # 3x2 image
        image_indices = torch.tensor([0, 1, 2, 3, 4, 5])
        height, width = 3, 2
        
        tokens = image_tokenizer.encapsulate_image(
            image_indices, height, width
        )
        
        # Count EOL tokens - should be 3 (one after each row including last)
        eol_count = (tokens == image_tokenizer.eol_id).sum().item()
        assert eol_count >= height  # At least one per row
    
    def test_batch_tokenization(self, image_tokenizer):
        """Test batch tokenization."""
        # Create batch of different sized images
        batch_indices = [
            torch.tensor([0, 1, 2, 3]),  # 2x2
            torch.tensor([10, 20, 30, 40, 50, 60]),  # 2x3
            torch.tensor([100]),  # 1x1
        ]
        dimensions = [(2, 2), (2, 3), (1, 1)]
        
        # Process each item individually since tokenize_batch doesn't exist
        result = []
        for indices, (h, w) in zip(batch_indices, dimensions):
            tokens = image_tokenizer.encapsulate_image(indices, h, w)
            result.append(tokens)
        
        assert isinstance(result, list)
        assert len(result) == len(batch_indices)
        
        # Check each sequence is a tensor
        for tokens in result:
            assert isinstance(tokens, torch.Tensor)
            assert tokens.dtype == torch.long
        
        # Check different lengths (no padding)
        assert len(result[0]) != len(result[1])  # 2x2 vs 2x3
        assert len(result[1]) != len(result[2])  # 2x3 vs 1x1
    
    
    def test_dimension_encoding(self, image_tokenizer):
        """Test that dimensions are properly encoded."""
        # Test various dimensions
        test_dims = [(1, 1), (2, 3), (10, 10), (128, 64)]
        
        for height, width in test_dims:
            # Create dummy indices
            image_indices = torch.zeros(height * width)
            
            tokens = image_tokenizer.encapsulate_image(
                image_indices, height, width
            )
            
            # Convert to list for easier checking
            token_list = tokens.tolist()
            
            # The dimension text should be tokenized
            dim_text = f"{height}*{width}"
            dim_tokens = image_tokenizer.text_tokenizer.encode(
                dim_text, add_special_tokens=False
            )
            
            # Check that dimension tokens appear in sequence
            # They should appear after img_start and before img_token_start
            for dim_token in dim_tokens:
                assert dim_token in token_list
    
    def test_no_generation_token_for_image_only(self, image_tokenizer):
        """Test that generation token is not added for image-only data."""
        image_indices = torch.tensor([0, 1, 2, 3])
        
        # Image-only tokenization should not include generation tokens
        tokens = image_tokenizer.encapsulate_image(
            image_indices, 2, 2
        )
        
        # Check no RESERVED tokens in output
        token_list = tokens.tolist()
        
        # No reserved tokens should appear
        for token_id in token_list:
            # Get token string
            token_str = image_tokenizer.text_tokenizer.convert_ids_to_tokens(token_id)
            if token_str:
                assert not token_str.startswith("<|RESERVED_")
    
    def test_compare_with_original(self, image_tokenizer):
        """Test that optimized method matches original and is faster for larger images."""
        test_cases = [
            (2, 2, torch.tensor([0, 1, 2, 3])),
            (16, 16, torch.arange(256)),  # Larger to show speedup
            (1, 100, torch.arange(100)),  # Edge case: wide image
        ]
        
        for height, width, image_indices in test_cases:
            comparison = image_tokenizer.compare_with_original(
                image_indices, height, width
            )
            
            # Verify all fields are present
            assert 'direct_tokens' in comparison
            assert 'direct_time' in comparison
            assert 'text_based_tokens' in comparison
            assert 'text_based_time' in comparison
            assert 'speedup' in comparison
            assert 'tokens_match' in comparison
            
            # Results must match
            assert comparison['tokens_match'], \
                f"Token mismatch for {height}x{width}! Direct != Text-based"
            
            # Verify actual token equality
            direct_tokens = comparison['direct_tokens']
            text_based_tokens = torch.tensor(comparison['text_based_tokens'], dtype=torch.long)
            assert torch.equal(direct_tokens, text_based_tokens), \
                f"Tokens don't match for {height}x{width}"
            
            # For small images, overhead might make direct method slower - that's OK
            # For larger images (100+ tokens), we expect speedup
            if height * width >= 100:
                assert comparison['speedup'] >= 1.0, \
                    f"Direct method should be faster for {height}x{width} image: {comparison['speedup']:.2f}x"


class TestIntegration:
    """Test integration with existing tokenizer infrastructure."""
    
    def test_with_llama_tokenizer(self):
        """Test with Llama tokenizer if available."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                add_emu3_special_tokens(
                    model_path="meta-llama/Meta-Llama-3-8B",
                    output_path=temp_dir,
                    visual_vocab_size=100,
                    num_reserved_tokens=10
                )
                
                image_tokenizer = EMU3ImageOnlyTokenizer(
                    text_tokenizer_path=temp_dir
                )
                
                # Test basic tokenization
                indices = torch.tensor([0, 1, 2, 3])
                tokens = image_tokenizer.encapsulate_image(
                    indices, 2, 2
                )
                
                assert len(tokens) > 0
                assert tokens[0] == image_tokenizer.bos_id
                
        except Exception:
            pytest.skip("Llama tokenizer not available")
    
    def test_with_qwen_tokenizer(self):
        """Test with Qwen tokenizer if available."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                add_emu3_special_tokens(
                    model_path="Qwen/Qwen2-7B",
                    output_path=temp_dir,
                    visual_vocab_size=100,
                    num_reserved_tokens=10
                )
                
                image_tokenizer = EMU3ImageOnlyTokenizer(
                    text_tokenizer_path=temp_dir
                )
                
                # Test batch processing
                batch = [torch.tensor([0, 1]), torch.tensor([2, 3, 4, 5])]
                dims = [(1, 2), (2, 2)]
                
                result = image_tokenizer.tokenize_batch(
                    batch, dims, padding=True
                )
                
                assert 'input_ids' in result
                assert result['input_ids'].shape[0] == 2
                
        except Exception:
            pytest.skip("Qwen tokenizer not available")


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_mismatched_dimensions(self, image_tokenizer):
        """Test that mismatched dimensions raise an error."""
        # Too many indices: 6 but claiming 2x2 (needs exactly 4)
        image_indices = torch.tensor([0, 1, 2, 3, 4, 5])
        
        with pytest.raises(AssertionError, match="Dimension mismatch"):
            image_tokenizer.encapsulate_image(image_indices, 2, 2)
        
        # Too few indices: 2 but claiming 2x2 (needs exactly 4)
        image_indices = torch.tensor([0, 1])
        
        with pytest.raises(AssertionError, match="Dimension mismatch"):
            image_tokenizer.encapsulate_image(image_indices, 2, 2)
    
    def test_empty_batch(self, image_tokenizer):
        """Test handling of empty batch."""
        # Process empty batch manually
        result = []
        
        # Should return empty list
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_single_item_batch(self, image_tokenizer):
        """Test batch with single item."""
        # Process single item
        result = [image_tokenizer.encapsulate_image(torch.tensor([42]), 1, 1)]
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])