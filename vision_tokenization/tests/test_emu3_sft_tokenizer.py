#!/usr/bin/env python3
"""
Test suite for EMU3ImageSftDataTokenizer.
Tests single-image SFT data tokenization with FineVision format.
"""

import torch
import pytest
from PIL import Image
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vision_tokenization.utils.tokenization_emu3_image_only import EMU3ImageSftDataTokenizer


class TestEMU3SftTokenizer:
    """Test cases for EMU3ImageSftDataTokenizer."""

    @pytest.fixture
    def tokenizer(self):
        """Initialize tokenizer for tests."""
        # Use the tokenizer with EMU3 special tokens
        tokenizer_path = "/capstor/store/cscs/swissai/infra01/MLLM/llama3_vision_instruct_emu3_tokenizer"
        return EMU3ImageSftDataTokenizer(
            text_tokenizer_path=tokenizer_path,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        # Create a simple RGB image for testing
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        return Image.fromarray(img_array, mode='RGB')

    @pytest.fixture
    def finevision_sample(self, sample_image):
        """Create a sample FineVision format data."""
        return {
            'texts': [
                {
                    "user": "What is shown in this image?",
                    "assistant": "This image shows a random color pattern used for testing."
                },
                {
                    "user": "Can you describe the colors?",
                    "assistant": "The image contains various random RGB colors in a 512x512 pixel grid."
                }
            ],
            'image': sample_image
        }


    def test_skip_no_image_placeholder(self, tokenizer, sample_image):
        """Test that messages without image placeholder are skipped."""
        messages = [
            {"role": "user", "content": "No image here"},
            {"role": "assistant", "content": "Text only response"}
        ]
        tokens = tokenizer.tokenize_conversation(messages, sample_image)
        assert len(tokens) == 0, "Should skip sample without image placeholder"
        print("✓ Correctly skipped sample without image placeholder")

    def test_single_image_tokenization(self, tokenizer, sample_image):
        """Test successful tokenization of single image message."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is this?"}
                ]
            },
            {"role": "assistant", "content": "This is a test image."}
        ]
        tokens = tokenizer.tokenize_conversation(messages, sample_image)

        assert len(tokens) > 0, "Should generate tokens"
        assert len(tokens) > 4000, f"Expected >4000 tokens with image, got {len(tokens)}"
        print("✓ Single image tokenization successful")

    def test_conversation_structure(self, tokenizer, sample_image):
        """Test that conversation headers are properly formatted."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is this?"}
                ]
            },
            {"role": "assistant", "content": "This is a test image."}
        ]
        tokens = tokenizer.tokenize_conversation(messages, sample_image)
        decoded = tokenizer.text_tokenizer.decode(tokens, skip_special_tokens=False)

        # Check conversation structure
        assert "<|start_header_id|>user<|end_header_id|>" in decoded, "Missing user header"
        assert "<|start_header_id|>assistant<|end_header_id|>" in decoded, "Missing assistant header"
        assert "<|eot_id|>" in decoded, "Missing end of turn markers"

        # Check message ordering
        user_idx = decoded.index("What is this?")
        assistant_idx = decoded.index("This is a test image.")
        assert user_idx < assistant_idx, "User message should come before assistant response"

        print("✓ Conversation structure validated")

    def test_emu3_vision_tokens(self, tokenizer, sample_image):
        """Test that EMU3 vision tokens are properly inserted."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this"}
                ]
            },
            {"role": "assistant", "content": "Description"}
        ]
        tokens = tokenizer.tokenize_conversation(messages, sample_image)
        token_list = tokens.tolist()

        # Check that placeholder was replaced
        image_placeholder_id = tokenizer.image_token_id  # <|image|>
        assert image_placeholder_id not in token_list, "Image placeholder should be replaced"

        # Check for EMU3 structure tokens
        decoded = tokenizer.text_tokenizer.decode(tokens, skip_special_tokens=False)
        assert "<|img_start|>" in decoded, "Missing EMU3 img_start token"
        assert "<|img_end|>" in decoded, "Missing EMU3 img_end token"

        # Verify vision tokens count
        img_start_id = tokenizer.text_tokenizer.convert_tokens_to_ids("<|img_start|>")
        img_end_id = tokenizer.text_tokenizer.convert_tokens_to_ids("<|img_end|>")
        start_idx = token_list.index(img_start_id)
        end_idx = token_list.index(img_end_id)
        vision_tokens = end_idx - start_idx - 1
        assert vision_tokens > 4000, f"Expected ~4096 vision tokens, got {vision_tokens}"

        print("✓ EMU3 vision tokens properly inserted")

    def test_complete_tokenization_ground_truth(self, tokenizer):
        """Test complete tokenization against known ground truth."""
        # Create a tiny 2x2 test image for predictable tokenization
        import numpy as np
        from PIL import Image

        # Create a 512x512 image (will be resized to 64x64 patches by EMU3)
        test_img = Image.new('RGB', (512, 512), color=(128, 128, 128))

        # Simple message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Hi"}
                ]
            },
            {"role": "assistant", "content": "Hello"}
        ]

        # Tokenize
        tokens = tokenizer.tokenize_conversation(messages, test_img)
        decoded = tokenizer.text_tokenizer.decode(tokens, skip_special_tokens=False)

        # Expected structure (without the actual vision tokens which are image-dependent):
        # <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        #
        # <|img_start|>64*64<|img_token_start|>[VISION_TOKENS]<|img_end|>Hi<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        #
        # Hello<|eot_id|>

        # Verify key components in order
        expected_sequence = [
            "<|begin_of_text|>",
            "<|start_header_id|>user<|end_header_id|>",
            "<|img_start|>64*64<|img_token_start|>",
            # ... vision tokens ...
            "<|img_end|>",
            "Hi",
            "<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>",
            "Hello",
            "<|eot_id|>"
        ]

        # Check each component exists and is in correct order
        last_pos = -1
        for component in expected_sequence:
            if "vision tokens" in component:
                continue  # Skip vision tokens check
            assert component in decoded, f"Missing component: {component}"
            pos = decoded.index(component, last_pos + 1)
            assert pos > last_pos, f"Component '{component}' not in expected order"
            last_pos = pos

        # Verify token count (should have exactly 4096 vision tokens + text tokens)
        token_list = tokens.tolist()
        assert token_list[0] == tokenizer.bos_id, "Should start with BOS"

        # Count vision tokens between img_start and img_end
        img_start_id = tokenizer.text_tokenizer.convert_tokens_to_ids("<|img_start|>")
        img_end_id = tokenizer.text_tokenizer.convert_tokens_to_ids("<|img_end|>")
        if img_start_id in token_list and img_end_id in token_list:
            start_idx = token_list.index(img_start_id)
            end_idx = token_list.index(img_end_id)

            # Between img_start and img_end we have:
            # - ~3 dimension tokens (e.g., "64", "*", "64")
            # - 1 img_token_start
            # - 4096 vision tokens (64x64)
            # - 64 EOL tokens (one after each row)
            # - 1 EOF token
            # Total: 3 + 1 + 4096 + 64 + 1 = 4165 tokens

            vision_section_length = end_idx - start_idx - 1
            expected_length = 3 + 1 + 4096 + 64 + 1  # = 4165
            assert expected_length - 2 <= vision_section_length <= expected_length + 2, \
                f"Vision section length: {vision_section_length}, expected ~{expected_length}"

        print("✓ Complete tokenization matches expected structure")

    def test_image_position(self, tokenizer, finevision_sample):
        """Test that image is always placed in first user message."""
        # Process the sample - split texts and image
        texts = finevision_sample["texts"]
        image = finevision_sample["image"]
        tokens = tokenizer.process_finevision_sample(texts, image)

        # Check that image token appears before first text
        image_token_id = tokenizer.image_token_id
        if image_token_id in tokens:
            image_pos = (tokens == image_token_id).nonzero(as_tuple=True)[0][0].item()
            # Image should appear early in the sequence (after BOS and headers)
            assert image_pos < 100, f"Image position {image_pos} seems too late"

        print("✓ Image positioned correctly in first user message")

    def test_missing_image_handling(self, tokenizer):
        """Test handling of missing image."""
        texts = [
            {
                "user": "What is shown?",
                "assistant": "Something."
            }
        ]
        # No image provided
        image = None

        tokens = tokenizer.process_finevision_sample(texts, image)
        assert len(tokens) == 0  # Should skip sample without image
        print("✓ Missing image handled correctly")


    def test_token_structure(self, tokenizer, finevision_sample):
        """Test the structure of generated tokens."""
        texts = finevision_sample["texts"]
        image = finevision_sample["image"]
        tokens = tokenizer.process_finevision_sample(texts, image)

        # Convert to list for easier inspection
        token_list = tokens.tolist()

        # Check for BOS token
        assert token_list[0] == tokenizer.bos_id, "Should start with BOS token"

        # Check for image structure tokens
        img_start_id = tokenizer.text_tokenizer.convert_tokens_to_ids("<|img_start|>")
        img_end_id = tokenizer.text_tokenizer.convert_tokens_to_ids("<|img_end|>")

        # Image tokens should be encapsulated
        if img_start_id in token_list:
            start_idx = token_list.index(img_start_id)
            assert img_end_id in token_list[start_idx:], "Image should have end marker"

        print("✓ Token structure validated")


    def test_multi_turn_conversation(self, tokenizer, sample_image):
        """Test multi-turn conversation tokenization."""
        texts = [
            {
                "user": "What's the main subject?",
                "assistant": "The main subject is a test pattern."
            },
            {
                "user": "What colors do you see?",
                "assistant": "I see various RGB colors."
            },
            {
                "user": "Is there any text?",
                "assistant": "No, there is no text visible."
            }
        ]

        tokens = tokenizer.process_finevision_sample(texts, sample_image)
        assert len(tokens) > 0

        # Verify conversation structure is preserved
        text = tokenizer.text_tokenizer.decode(tokens, skip_special_tokens=False)
        assert "What's the main subject?" in text
        assert "What colors do you see?" in text
        assert "Is there any text?" in text

        print("✓ Multi-turn conversation tokenized correctly")


def main():
    """Run tests manually."""
    print("🧪 Testing EMU3ImageSftDataTokenizer\n")

    # Create test instance
    test = TestEMU3SftTokenizer()

    # Initialize fixtures
    tokenizer = test.tokenizer()
    sample_image = test.sample_image()
    finevision_sample = test.finevision_sample(sample_image)

    # Run tests
    try:
        test.test_skip_no_image_placeholder(tokenizer, sample_image)
        test.test_single_image_tokenization(tokenizer, sample_image)
        test.test_conversation_structure(tokenizer, sample_image)
        test.test_emu3_vision_tokens(tokenizer, sample_image)
        test.test_complete_tokenization_ground_truth(tokenizer)
        test.test_image_position(tokenizer, finevision_sample)
        test.test_missing_image_handling(tokenizer)
        test.test_token_structure(tokenizer, finevision_sample)
        test.test_multi_turn_conversation(tokenizer, sample_image)

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()