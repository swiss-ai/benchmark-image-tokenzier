#!/usr/bin/env python3
"""
Test updating a text tokenizer with EMU3-style vision tokens and generation mode.
This tests the complete workflow from a standard text tokenizer to a multimodal tokenizer.
"""

import os
import sys
import json
import tempfile
import pytest
from pathlib import Path

# Add utils directory to path (sibling directory)
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from transformers import AutoTokenizer
from add_special_tokens_emu3_style import (
    add_emu3_special_tokens,
    get_vision_token_id,
    load_vision_token_mapping
)
from hidden_mode_token_registry import HiddenModeTokenRegistry


# Default test model - use whatever is available
DEFAULT_MODEL = "Qwen/Qwen2-0.5B"  # Small Qwen model for testing
# Alternative models:
# "meta-llama/Llama-2-7b-hf"
# "meta-llama/Llama-3-8B"
# "mistralai/Mistral-7B-v0.1"
# "Qwen/Qwen2-7B"
# "swissai/swissbert"


@pytest.fixture
def temp_tokenizer_path():
    """Create a temporary directory for tokenizer testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(params=[
    "meta-llama/Meta-Llama-3-8B",
    "Qwen/Qwen2-7B",
    "alehc/swissai-tokenizer",
    "gpt2"
])
def base_model(request):
    """Test with multiple tokenizer models including SwissAI."""
    model = request.param
    try:
        # Try to load the tokenizer to verify it's available
        AutoTokenizer.from_pretrained(model)
        return model
    except Exception as e:
        pytest.skip(f"Model {model} not available: {str(e)[:50]}")


@pytest.fixture
def small_tokenizer(temp_tokenizer_path, base_model):
    """Create a small tokenizer for testing."""
    output_path = os.path.join(temp_tokenizer_path, "test_tokenizer")
    tokenizer, stats = add_emu3_special_tokens(
        model_path=base_model,
        output_path=output_path,
        visual_vocab_size=100,
        num_reserved_tokens=20
    )
    return output_path, tokenizer, stats, base_model


class TestTokenizerUpdate:
    """Test basic tokenizer update functionality."""
    
    def test_add_structure_tokens(self, temp_tokenizer_path, base_model):
        """Test adding EMU3 structure tokens."""
        output_path = os.path.join(temp_tokenizer_path, "test_tokenizer")
        tokenizer, stats = add_emu3_special_tokens(
            model_path=base_model,
            output_path=output_path,
            visual_vocab_size=100,
            num_reserved_tokens=20
        )
        
        # Check structure tokens exist
        structure_tokens = [
            "<|img_start|>", "<|img_end|>",
            "<|img_token_start|>", "<|img_end_of_row|>",
            "<|img_end_of_frame|>"
        ]
        
        for token in structure_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            assert token_id != tokenizer.unk_token_id, f"{token} should exist"
    
    def test_add_reserved_tokens(self, temp_tokenizer_path, base_model):
        """Test adding reserved tokens."""
        output_path = os.path.join(temp_tokenizer_path, "test_tokenizer")
        num_reserved = 30
        
        tokenizer, stats = add_emu3_special_tokens(
            model_path=base_model,
            output_path=output_path,
            visual_vocab_size=50,
            num_reserved_tokens=num_reserved
        )
        
        assert stats['reserved_tokens_added'] == num_reserved
        
        # Check first and last reserved tokens
        first_token = tokenizer.convert_tokens_to_ids("<|RESERVED_000|>")
        last_token = tokenizer.convert_tokens_to_ids(f"<|RESERVED_{num_reserved-1:03d}|>")
        
        assert first_token != tokenizer.unk_token_id
        assert last_token != tokenizer.unk_token_id
    
    def test_add_visual_tokens(self, temp_tokenizer_path, base_model):
        """Test adding visual tokens."""
        output_path = os.path.join(temp_tokenizer_path, "test_tokenizer")
        visual_vocab_size = 1000
        
        tokenizer, stats = add_emu3_special_tokens(
            model_path=base_model,
            output_path=output_path,
            visual_vocab_size=visual_vocab_size,
            num_reserved_tokens=10
        )
        
        assert stats['visual_tokens_added'] == visual_vocab_size
        
        # Check boundary visual tokens
        first_visual = tokenizer.convert_tokens_to_ids("<|visual token 000000|>")
        last_visual = tokenizer.convert_tokens_to_ids(f"<|visual token {visual_vocab_size-1:06d}|>")
        
        assert first_visual != tokenizer.unk_token_id
        assert last_visual != tokenizer.unk_token_id
    
    def test_vocab_size_increase(self, temp_tokenizer_path, base_model):
        """Test that vocabulary size increases correctly."""
        output_path = os.path.join(temp_tokenizer_path, "test_tokenizer")
        
        # Get original size
        original_tokenizer = AutoTokenizer.from_pretrained(base_model)
        original_size = len(original_tokenizer)
        
        # Add tokens
        tokenizer, stats = add_emu3_special_tokens(
            model_path=base_model,
            output_path=output_path,
            visual_vocab_size=100,
            num_reserved_tokens=20
        )
        
        # Verify size increase
        expected_increase = (
            stats['structure_tokens_added'] +
            stats['reserved_tokens_added'] +
            stats['visual_tokens_added']
        )
        actual_increase = stats['final_vocab_size'] - stats['original_vocab_size']
        
        assert actual_increase == expected_increase


class TestVisionTokenMapping:
    """Test vision token ID mapping functionality."""
    
    def test_save_and_load_mapping(self, small_tokenizer):
        """Test saving and loading vision token mapping."""
        output_path, tokenizer, stats, _ = small_tokenizer
        
        # Load mapping
        mapping = load_vision_token_mapping(output_path)
        
        assert 'vision_token_ids' in mapping
        assert 'visual_vocab_size' in mapping
        assert mapping['visual_vocab_size'] == 100
    
    def test_vision_index_to_token_id(self, small_tokenizer):
        """Test converting vision indices to token IDs."""
        output_path, tokenizer, stats, _ = small_tokenizer
        
        # Test various indices
        test_indices = [0, 10, 50, 99]
        
        for idx in test_indices:
            token_id = get_vision_token_id(idx, output_path)
            expected_token = f"<|visual token {idx:06d}|>"
            expected_id = tokenizer.convert_tokens_to_ids(expected_token)
            assert token_id == expected_id
    
    def test_invalid_vision_index(self, small_tokenizer):
        """Test that invalid vision indices raise errors."""
        output_path, _, _, _ = small_tokenizer
        
        with pytest.raises(ValueError):
            get_vision_token_id(100, output_path)  # Out of range
        
        with pytest.raises(ValueError):
            get_vision_token_id(-1, output_path)  # Negative


class TestGenerationMode:
    """Test generation mode functionality."""
    
    def test_generation_token_selection(self, small_tokenizer):
        """Test that generation token is selected from reserved tokens."""
        output_path, tokenizer, _, _ = small_tokenizer
        
        # Check generation config exists
        gen_config_path = os.path.join(output_path, "generation_mode_token.json")
        assert os.path.exists(gen_config_path)
        
        # Load and verify
        with open(gen_config_path, 'r') as f:
            config = json.load(f)
        
        assert 'generation_token' in config
        assert 'generation_token_id' in config
        assert config['generation_token'].startswith("<|RESERVED_")
    
    def test_generation_token_deterministic(self):
        """Test that generation token selection is deterministic."""
        manager1 = HiddenModeTokenRegistry(seed="test_seed")
        manager2 = HiddenModeTokenRegistry(seed="test_seed")
        
        token1, idx1 = manager1.select_generation_token(50)
        token2, idx2 = manager2.select_generation_token(50)
        
        assert token1 == token2
        assert idx1 == idx2
    
    def test_attach_to_tokenizer(self, small_tokenizer):
        """Test attaching generation mode to tokenizer."""
        output_path, _, _, _ = small_tokenizer
        
        # Load tokenizer and attach generation mode
        tokenizer = AutoTokenizer.from_pretrained(output_path, trust_remote_code=True)
        gen_manager = HiddenModeTokenRegistry(output_path)
        gen_manager.attach_to_tokenizer(tokenizer)
        
        assert hasattr(tokenizer, 'gen_token')
        assert hasattr(tokenizer, 'gen_token_id')
        assert tokenizer.gen_token is not None
        assert tokenizer.gen_token_id is not None
    
    def test_deployment_mode(self, small_tokenizer):
        """Test that generation mode is hidden in deployment."""
        output_path, _, _, _ = small_tokenizer
        
        # Remove generation config
        gen_config_path = os.path.join(output_path, "generation_mode_token.json")
        os.remove(gen_config_path)
        
        # Try to load generation mode (should fail)
        with pytest.raises(FileNotFoundError):
            HiddenModeTokenRegistry(output_path)


class TestFullScale:
    """Test full-scale EMU3 tokenizer configuration."""
    
    @pytest.mark.slow
    def test_full_emu3_tokenizer(self, temp_tokenizer_path, base_model):
        """Test creating full EMU3 tokenizer with 32K visual tokens."""
        output_path = os.path.join(temp_tokenizer_path, "emu3_full")
        
        tokenizer, stats = add_emu3_special_tokens(
            model_path=base_model,
            output_path=output_path,
            visual_vocab_size=32768,
            num_reserved_tokens=100
        )
        
        assert stats['visual_tokens_added'] == 32768
        assert stats['reserved_tokens_added'] == 100
        
        # Test boundary tokens
        first_visual = tokenizer.convert_tokens_to_ids("<|visual token 000000|>")
        last_visual = tokenizer.convert_tokens_to_ids("<|visual token 032767|>")
        
        assert first_visual != tokenizer.unk_token_id
        assert last_visual != tokenizer.unk_token_id


class TestTokenization:
    """Test actual tokenization with the updated tokenizer."""
    
    def test_tokenize_multimodal_sequence(self, small_tokenizer):
        """Test tokenizing a multimodal sequence."""
        output_path, _, _, _ = small_tokenizer
        tokenizer = AutoTokenizer.from_pretrained(output_path, trust_remote_code=True)
        
        # Create a multimodal sequence
        text = "The image <|img_start|>2*2<|img_token_start|>"
        text += "<|visual token 000000|><|visual token 000001|><|img_end_of_row|>"
        text += "<|visual token 000002|><|visual token 000003|><|img_end_of_row|>"
        text += "<|img_end_of_frame|><|img_end|> shows a cat."
        
        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Verify it tokenizes without unknown tokens
        decoded = tokenizer.decode(tokens, skip_special_tokens=False)
        # Check for unknown token indicators (varies by tokenizer)
        if tokenizer.unk_token:
            assert tokenizer.unk_token not in decoded, "Unknown token found in decoded text"
        assert "[UNK]" not in decoded, "UNK marker found in decoded text"
    
    def test_generation_mode_sequence(self, small_tokenizer):
        """Test tokenizing with generation mode token."""
        output_path, _, _, _ = small_tokenizer
        
        # Load tokenizer with generation mode
        tokenizer = AutoTokenizer.from_pretrained(output_path, trust_remote_code=True)
        gen_manager = HiddenModeTokenRegistry(output_path)
        gen_manager.attach_to_tokenizer(tokenizer)
        
        # Create sequence with generation token
        if tokenizer.bos_token:
            text = tokenizer.bos_token + tokenizer.gen_token + "Generate an image"
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # Check generation token is at position 1
            if len(tokens) > 1:
                assert tokens[1] == tokenizer.gen_token_id


@pytest.mark.parametrize("model_name", [
    "meta-llama/Meta-Llama-3-8B",  # Llama-3
    "Qwen/Qwen2-7B",               # Qwen2
    "alehc/swissai-tokenizer",     # SwissAI
    "gpt2",                        # GPT-2
])
def test_specific_models(model_name, temp_tokenizer_path):
    """Test that tokenizer update works with specific models."""
    # Skip if model not available
    try:
        AutoTokenizer.from_pretrained(model_name)
    except:
        pytest.skip(f"Model {model_name} not available")
    
    output_path = os.path.join(temp_tokenizer_path, f"{model_name.replace('/', '_')}_vision")
    
    tokenizer, stats = add_emu3_special_tokens(
        model_path=model_name,
        output_path=output_path,
        visual_vocab_size=100,
        num_reserved_tokens=20
    )
    
    assert stats['final_vocab_size'] > stats['original_vocab_size']
    
    # Verify a sample token
    token_id = tokenizer.convert_tokens_to_ids("<|img_start|>")
    assert token_id != tokenizer.unk_token_id


class TestLlamaSpecific:
    """Test Llama-specific tokenizer features."""
    
    def test_llama_special_tokens(self, small_tokenizer):
        """Test that Llama special tokens are preserved."""
        output_path, tokenizer, _, model_name = small_tokenizer
        
        if "llama" in model_name.lower():
            # Check Llama-specific tokens are preserved
            if hasattr(tokenizer, 'bos_token'):
                assert tokenizer.bos_token is not None
            if hasattr(tokenizer, 'eos_token'):
                assert tokenizer.eos_token is not None


class TestQwenSpecific:
    """Test Qwen-specific tokenizer features."""
    
    def test_qwen_special_tokens(self, small_tokenizer):
        """Test that Qwen special tokens are preserved."""
        output_path, tokenizer, _, model_name = small_tokenizer
        
        if "qwen" in model_name.lower():
            # Qwen uses special tokens like <|endoftext|>
            # Check they are preserved
            original_special_tokens = tokenizer.all_special_tokens
            assert len(original_special_tokens) > 0


if __name__ == "__main__":
    # Run pytest with verbose output
    pytest.main([__file__, "-v", "-s"])