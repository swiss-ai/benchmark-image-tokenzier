"""
Example script showing how to update different tokenizers with vision tokens.
Demonstrates EMU3-style token registration and usage.
"""

from transformers import AutoTokenizer
from tokenizer_updater import VisionTokenizerUpdater
import torch
import numpy as np


def example_emu3_style():
    """Example showing EMU3-style tokenizer update."""
    print("=" * 50)
    print("EMU3-Style Tokenizer Update")
    print("=" * 50)
    
    # Load base tokenizer (e.g., LLaMA, GPT2, etc.)
    base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"Original tokenizer vocab size: {len(base_tokenizer)}")
    
    # Create updater with EMU3-style tokens
    updater = VisionTokenizerUpdater(
        tokenizer=base_tokenizer,
        num_vision_tokens=32768,  # EMU3 uses ~32k vision tokens
        vision_token_format="<|visual_token_{:06d}|>",  # EMU3 format
        add_special_tokens=True,
        add_vision_structure_tokens=True
    )
    
    # Update the tokenizer
    stats = updater.update_tokenizer()
    print(f"\nTokenizer updated:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Example: Encode an image
    print("\n" + "=" * 50)
    print("Example: Encoding a 4x4 image")
    print("=" * 50)
    
    # Simulate vision tokenizer output (16 tokens for 4x4 image)
    vision_indices = list(range(100, 116))  # Dummy indices
    height, width = 4, 4
    
    # Encode with structure tokens
    token_ids = updater.encode_image_tokens(vision_indices, height, width)
    print(f"Vision indices: {vision_indices}")
    print(f"Encoded token IDs: {token_ids[:10]}... (showing first 10)")
    
    # Decode to see the actual tokens
    decoded = updater.tokenizer.decode(token_ids)
    print(f"\nDecoded tokens (first 200 chars):\n{decoded[:200]}...")
    
    return updater


def example_simplified_tokens():
    """Example with simplified token structure."""
    print("\n" + "=" * 50)
    print("Simplified Token Structure")
    print("=" * 50)
    
    # Load base tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Create updater with simplified structure
    updater = VisionTokenizerUpdater(
        tokenizer=base_tokenizer,
        num_vision_tokens=8192,  # Smaller vocabulary
        vision_token_format="<v{:04d}>",  # Shorter format
        add_special_tokens=True,
        add_vision_structure_tokens=True
    )
    
    # Override with simplified structure tokens
    updater.vision_structure_tokens = {
        "boi_token": "<IMG>",
        "eoi_token": "</IMG>",
        "eol_token": "<BR>",  # Line break
        "img_token": "",  # Not used in simplified version
        "eof_token": "",  # Not used in simplified version
    }
    
    # Update tokenizer
    stats = updater.update_tokenizer()
    print(f"Updated with {stats['vision_tokens_added']} vision tokens")
    
    return updater


def example_continuous_pretraining_format():
    """Example for continuous pretraining format."""
    print("\n" + "=" * 50)
    print("Continuous Pretraining Format")
    print("=" * 50)
    
    # Load a pretrained LLM tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
    print(f"LLaMA tokenizer vocab size: {len(base_tokenizer)}")
    
    # Create updater optimized for continuous pretraining
    updater = VisionTokenizerUpdater(
        tokenizer=base_tokenizer,
        num_vision_tokens=32768,
        vision_token_format="<|v{:05d}|>",
        add_special_tokens=False,  # LLaMA already has special tokens
        add_vision_structure_tokens=True
    )
    
    # Minimal structure tokens for continuous pretraining
    updater.vision_structure_tokens = {
        "boi_token": "<|begin_image|>",
        "eoi_token": "<|end_image|>",
        "eol_token": "",  # Not needed for continuous features
        "img_token": "",  # Not needed
        "eof_token": "",  # Not needed
    }
    
    # Update tokenizer
    stats = updater.update_tokenizer()
    print(f"\nMinimal update for continuous pretraining:")
    print(f"  Added {stats['vision_tokens_added']} vision tokens")
    print(f"  Added {stats['structure_tokens_added']} structure tokens")
    
    # Example sequence for continuous pretraining
    print("\nExample sequence formats:")
    
    # Image-only
    vision_indices = [1000, 2000, 3000, 4000]
    boi_id = base_tokenizer.convert_tokens_to_ids("<|begin_image|>")
    eoi_id = base_tokenizer.convert_tokens_to_ids("<|end_image|>")
    
    image_only = [boi_id] + [updater.get_vision_token_id(idx) for idx in vision_indices] + [eoi_id]
    print(f"Image-only: BOI + {len(vision_indices)} vision tokens + EOI")
    
    # Text + Image
    text = "This is an image of"
    text_ids = base_tokenizer.encode(text, add_special_tokens=True)
    multimodal = text_ids + image_only
    print(f"Multimodal: {len(text_ids)} text tokens + {len(image_only)} image tokens")
    
    return updater


def example_custom_format():
    """Example with custom token format for specific use case."""
    print("\n" + "=" * 50)
    print("Custom Format for Your Use Case")
    print("=" * 50)
    
    # Base tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    class CustomVisionTokenizer(VisionTokenizerUpdater):
        """Custom tokenizer for specific requirements."""
        
        def encode_image_tokens_minimal(self, vision_indices: list) -> list:
            """Minimal encoding without dimension tokens."""
            token_ids = []
            
            # Just BOI + vision tokens + EOI
            token_ids.append(self.tokenizer.convert_tokens_to_ids(self.vision_structure_tokens["boi_token"]))
            
            for idx in vision_indices:
                token_ids.append(self.get_vision_token_id(idx))
            
            token_ids.append(self.tokenizer.convert_tokens_to_ids(self.vision_structure_tokens["eoi_token"]))
            
            return token_ids
        
        def encode_video_tokens(self, frame_indices_list: list) -> list:
            """Encode video as sequence of frames."""
            token_ids = []
            
            # BOV token
            bov_id = self.tokenizer.convert_tokens_to_ids(self.vision_structure_tokens["bov_token"])
            eov_id = self.tokenizer.convert_tokens_to_ids(self.vision_structure_tokens["eov_token"])
            bof_id = self.tokenizer.convert_tokens_to_ids(self.vision_structure_tokens["bof_token"])
            eof_id = self.tokenizer.convert_tokens_to_ids(self.vision_structure_tokens["eof_token"])
            
            token_ids.append(bov_id)
            
            for frame_idx, frame_indices in enumerate(frame_indices_list):
                # Begin frame with index
                token_ids.append(bof_id)
                
                # Add frame tokens
                for idx in frame_indices:
                    token_ids.append(self.get_vision_token_id(idx))
                
                # End frame
                token_ids.append(eof_id)
            
            token_ids.append(eov_id)
            
            return token_ids
    
    # Create custom tokenizer
    updater = CustomVisionTokenizer(
        tokenizer=base_tokenizer,
        num_vision_tokens=16384,  # Smaller for efficiency
        vision_token_format="<vis_{:05d}>",
        add_special_tokens=True,
        add_vision_structure_tokens=True
    )
    
    stats = updater.update_tokenizer()
    print(f"Custom tokenizer created with {stats['final_vocab_size']} tokens")
    
    # Example usage
    print("\nMinimal image encoding:")
    image_indices = [100, 200, 300, 400]
    minimal_tokens = updater.encode_image_tokens_minimal(image_indices)
    print(f"  {len(minimal_tokens)} tokens for {len(image_indices)} vision tokens")
    
    print("\nVideo encoding (3 frames):")
    video_frames = [
        [100, 101, 102, 103],  # Frame 0
        [200, 201, 202, 203],  # Frame 1
        [300, 301, 302, 303],  # Frame 2
    ]
    video_tokens = updater.encode_video_tokens(video_frames)
    print(f"  {len(video_tokens)} tokens for {len(video_frames)} frames")
    
    return updater


def main():
    """Run all examples."""
    
    # Example 1: EMU3-style
    emu3_updater = example_emu3_style()
    
    # Example 2: Simplified
    simple_updater = example_simplified_tokens()
    
    # Example 3: Continuous pretraining
    # Note: This will fail if you don't have LLaMA tokenizer
    # continuous_updater = example_continuous_pretraining_format()
    
    # Example 4: Custom format
    custom_updater = example_custom_format()
    
    # Save one example
    print("\n" + "=" * 50)
    print("Saving EMU3-style tokenizer")
    print("=" * 50)
    save_path = "/tmp/emu3_style_tokenizer"
    emu3_updater.save_updated_tokenizer(save_path)
    print(f"Saved to {save_path}")
    
    # Load it back
    print("\nLoading saved tokenizer...")
    loaded_updater = VisionTokenizerUpdater.from_pretrained(save_path)
    print(f"Loaded tokenizer with {len(loaded_updater.tokenizer)} tokens")


if __name__ == "__main__":
    main()