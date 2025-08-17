#!/usr/bin/env python3
"""
Generation Mode Token Utilities (PRIVATE - DO NOT DISTRIBUTE)

This module handles the secret generation mode token for safe multimodal training.
The generation token is hidden among reserved tokens to prevent unauthorized image generation.

IMPORTANT: This file should NOT be included in public model releases.
"""

import json
import os
import random
from typing import Optional, Tuple


class HiddenModeTokenRegistry:
    """Registry for hidden mode tokens that control model capabilities."""
    
    def __init__(self, tokenizer_path: Optional[str] = None, seed: str = "multimodal_gen_2024"):
        self.seed = seed
        self.generation_token = None
        self.generation_token_id = None
        self.generation_token_index = None
        
        if tokenizer_path:
            self.load_from_file(tokenizer_path)
    
    def select_generation_token(self, num_reserved_tokens: int = 100) -> Tuple[str, int]:
        """Randomly select a reserved token for generation mode."""
        random.seed(self.seed)
        self.generation_token_index = random.randint(0, num_reserved_tokens - 1)
        self.generation_token = f"<|RESERVED_{self.generation_token_index:03d}|>"
        return self.generation_token, self.generation_token_index
    
    def save_to_file(self, output_path: str, tokenizer) -> None:
        """Save generation token configuration to a private JSON file."""
        if not self.generation_token:
            raise ValueError("No generation token selected yet")
        
        self.generation_token_id = tokenizer.convert_tokens_to_ids(self.generation_token)
        
        config = {
            "generation_token": self.generation_token,
            "generation_token_id": self.generation_token_id,
            "generation_token_index": self.generation_token_index,
            "seed": self.seed,
            "warning": "PRIVATE FILE - DO NOT DISTRIBUTE"
        }
        
        filepath = os.path.join(output_path, "generation_mode_token.json")
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[PRIVATE] Saved generation token config to {filepath}")
        print(f"[PRIVATE] Generation mode uses: {self.generation_token}")
    
    def load_from_file(self, tokenizer_path: str) -> None:
        """Load generation token configuration from file."""
        filepath = os.path.join(tokenizer_path, "generation_mode_token.json")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Generation mode token file not found at {filepath}. "
                "This is expected for public deployments."
            )
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.generation_token = config["generation_token"]
        self.generation_token_id = config["generation_token_id"]
        self.generation_token_index = config.get("generation_token_index")
        self.seed = config.get("seed", self.seed)
    
    def attach_to_tokenizer(self, tokenizer):
        """
        Attach generation token attributes to tokenizer (Megatron-style).
        Adds gen_token and gen_token_id attributes to the tokenizer.
        """
        if self.generation_token:
            tokenizer.gen_token = self.generation_token
            tokenizer.gen_token_id = self.generation_token_id
        else:
            tokenizer.gen_token = None
            tokenizer.gen_token_id = None
        return tokenizer


# Example usage for Megatron-LM integration
if __name__ == "__main__":
    print("Generation Mode Token Manager")
    print("=" * 50)
    print("\nUsage in Megatron tokenizer wrapper:")
    print("""
    from generation_mode_utils import GenerationModeManager
    
    # During tokenizer initialization:
    try:
        gen_manager = GenerationModeManager(tokenizer_path)
        gen_manager.attach_to_tokenizer(tokenizer)
        # Now tokenizer has .gen_token and .gen_token_id attributes
    except FileNotFoundError:
        # Deployment mode - no generation token
        pass
    
    # In your dataset class, just check:
    if input_ids[1] == tokenizer.gen_token_id:
        # Enable loss on image tokens
        loss_mask[image_positions] = 1.0
    else:
        # Mask image tokens  
        loss_mask[image_positions] = 0.0
    """)