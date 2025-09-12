
#!/usr/bin/env python3
"""
EMU3 tokenization for image-only data.
Avoids double tokenization by directly merging image indices with text tokens.
"""

import torch
from typing import List, Tuple, Dict
from transformers import AutoTokenizer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Tokenizer'))
from Tokenizer.Emu3VisionTokenizer import Emu3VisionTokenizer


class EMU3ImageOnlyTokenizer:
    """
    Direct tokenization for EMU3 image-only sequences.
    Skips text generation step and directly combines tokens.
    """
    
    def __init__(self, text_tokenizer_path: str, device: str = "cuda"):
        """
        Initialize with text tokenizer that has EMU3 vision tokens and image tokenizer.
        
        Args:
            text_tokenizer_path: Path to text tokenizer with EMU3 tokens
            device: Device for image tokenizer (default: "cuda")
        """
        
        # Load tokenizer with trust_remote_code for custom EMU3Tokenizer class
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path, trust_remote_code=True)
        self.image_tokenizer = Emu3VisionTokenizer(device=device)
        
        # Cache for dimension tokens to avoid repeated encoding
        self.dim_cache = {}
        
        # Cache frequently used token IDs
        self._cache_special_tokens()
    
    def _cache_special_tokens(self):
        """Cache special token IDs to avoid repeated lookups."""
        # Structure tokens
        assert self.text_tokenizer.bos_token is not None, "BOS token must be defined"
        assert self.text_tokenizer.eos_token is not None, "EOS token must be defined"

        self.bos_id = self.text_tokenizer.bos_token_id
        self.eos_id = self.text_tokenizer.eos_token_id

        # EMU3 special tokens
        self.img_start_id = self.text_tokenizer.convert_tokens_to_ids("<|img_start|>")
        self.img_end_id = self.text_tokenizer.convert_tokens_to_ids("<|img_end|>")
        self.img_token_start_id = self.text_tokenizer.convert_tokens_to_ids("<|img_token_start|>")
        self.eol_id = self.text_tokenizer.convert_tokens_to_ids("<|img_end_of_row|>")
        self.eof_id = self.text_tokenizer.convert_tokens_to_ids("<|img_end_of_frame|>")

        # Compute the actual offset for vision tokens
        # Vision tokens are "<|visual token 000000|>" through "<|visual token XXXXXX|>"
        first_vision_token = self.text_tokenizer.convert_tokens_to_ids("<|visual token 000000|>")
        self.vision_token_offset = first_vision_token

    def _get_dim_tokens(self, height: int, width: int) -> List[int]:
        """
        Get dimension tokens with caching to avoid repeated encoding.
        
        Args:
            height: Image height in tokens
            width: Image width in tokens
            
        Returns:
            List of token IDs for the dimension string
        """
        dim_key = f"{height}*{width}"
        if dim_key not in self.dim_cache:
            # Encode and cache the dimension tokens
            self.dim_cache[dim_key] = self.text_tokenizer.encode(
                dim_key, 
                add_special_tokens=False
            )
        return self.dim_cache[dim_key]

    def encapsulate_image(
        self, 
        image_indices: torch.Tensor, 
        height: int, 
        width: int
    ) -> torch.Tensor:
        """
        Directly tokenize image-only data without intermediate text conversion.
        
        Args:
            image_indices: Tensor of image indices from vision tokenizer [H*W]
            height: Image height in tokens
            width: Image width in tokens
            
        Returns:
            Token IDs ready for model input
        """
        num_tokens_needed = height * width
        assert image_indices.numel() == num_tokens_needed, \
            f"Dimension mismatch: {height}x{width} needs {num_tokens_needed} indices, got {image_indices.numel()}"
        
        # Pre-allocate output tensor for efficiency
        # Structure: BOS + img_start + dims(~3) + img_token_start + vision_tokens + EOLs + EOF + img_end + EOS
        # Use cached dimension tokens to avoid repeated encoding
        dim_tokens = self._get_dim_tokens(height, width)
        
        # Calculate total size
        total_size = (
            1 +  # BOS
            1 +  # img_start
            len(dim_tokens) +  # dimension tokens
            1 +  # img_token_start
            num_tokens_needed +  # vision tokens
            height +  # EOL after each row
            1 +  # EOF
            1 +  # img_end
            1    # EOS
        )
        
        # Pre-allocate the entire output tensor
        output = torch.empty(total_size, dtype=torch.long)
        
        # Fill in the tokens using slicing (no Python list operations)
        idx = 0
        
        # Fixed tokens at the beginning
        output[idx] = self.bos_id
        output[idx + 1] = self.img_start_id
        idx += 2
        
        # Dimension tokens
        output[idx:idx + len(dim_tokens)] = torch.tensor(dim_tokens, dtype=torch.long)
        idx += len(dim_tokens)
        
        output[idx] = self.img_token_start_id
        idx += 1
        
        # Vision tokens with EOL markers - fully vectorized
        image_indices = image_indices.view(height, width)
        vision_tokens_with_offset = image_indices + self.vision_token_offset
        
        # Create vision part with EOL tokens in one operation
        vision_part = torch.empty((height, width + 1), dtype=torch.long)
        vision_part[:, :width] = vision_tokens_with_offset
        vision_part[:, -1] = self.eol_id
        
        # Copy all rows at once
        output[idx:idx + height*(width+1)] = vision_part.flatten()
        idx += height * (width + 1)
        
        # Final tokens
        output[idx] = self.eof_id
        output[idx + 1] = self.img_end_id
        output[idx + 2] = self.eos_id
        
        return output
    
    def tokenize_image(self, image) -> torch.Tensor:
        """
        Complete pipeline: PIL image → vision indices → EMU3 encapsulated tokens.
        
        Args:
            image: PIL Image
            
        Returns:
            Token sequence with EMU3 structure tokens (BOS, img_start, dims, EOL, EOS, etc.)
        """
        assert self.image_tokenizer is not None, "Image tokenizer required for processing images"
        
        # Step 1: Preprocess image (PIL → tensor)
        img_tensor = self.image_tokenizer.preprocess(image)
        
        # Step 2: Encode to vision indices
        indices, _ = self.image_tokenizer.encode(img_tensor)
        
        # Step 3: Get dimensions and flatten
        # [1, H, W] → [H, W] → [H*W]
        indices_2d = indices.squeeze(0)
        height, width = indices_2d.shape
        image_indices = indices_2d.flatten()
        
        # Step 4: Encapsulate with EMU3 structure tokens
        return self.encapsulate_image(image_indices, height, width)

    def translate_image_to_text(self, image) -> str:
        """
        Translate a PIL image to EMU3 text representation with special tokens.
        
        Args:
            image: PIL Image
            
        Returns:
            Text string with EMU3 special tokens like:
            '<|img_start|>32*32<|img_token_start|><|visual token 000000|>...<|img_end|>'
        """
        # First tokenize the image to get token IDs
        token_ids = self.tokenize_image(image)
        
        # Decode to text using the text tokenizer
        text = self.text_tokenizer.decode(token_ids, skip_special_tokens=False)

        return text
    
    def compare_with_original(
        self,
        image_indices: torch.Tensor,
        height: int,
        width: int
    ) -> Dict[str, any]:
        """
        Compare direct tokenization with text-based two-stage approach.
        Useful for verification and benchmarking.
        """
        import time
        
        # Direct tokenization (our method)
        start = time.time()
        direct_tokens = self.encapsulate_image(image_indices, height, width)
        direct_time = time.time() - start
        
        # Text-based approach (reference)
        start = time.time()
        # Stage 1: Convert to text representation
        text = f"<|img_start|>{height}*{width}<|img_token_start|>"
        for row in range(height):
            for col in range(width):
                idx = row * width + col
                text += f"<|visual token {image_indices[idx]:06d}|>"
            text += "<|img_end_of_row|>"  # EOL after every row including last
        text += "<|img_end_of_frame|><|img_end|>"
        
        # Stage 2: Tokenize text (adds BOS but not EOS)
        text_based_tokens = self.text_tokenizer.encode(text, add_special_tokens=True)
        text_based_time = time.time() - start
        
        # Add EOS to match our direct method which always includes it
        text_based_tokens.append(self.eos_id)
        text_based_tensor = torch.tensor(text_based_tokens, dtype=torch.long)
        
        return {
            'direct_tokens': direct_tokens,
            'direct_time': direct_time,
            'text_based_tokens': text_based_tokens,
            'text_based_time': text_based_time,
            'speedup': text_based_time / direct_time if direct_time > 0 else 0,
            'tokens_match': torch.equal(direct_tokens, text_based_tensor)
        }


# Example usage
if __name__ == "__main__":
    # Initialize EMU3 image-only tokenizer
    # Use the tokenizer with EMU3 special tokens
    tokenizer = EMU3ImageOnlyTokenizer(
        text_tokenizer_path="/iopsstor/scratch/cscs/xyixuan/llama3_emu3_tokenizer",
        device="cuda"
    )
    
    # Simulate image indices from vision tokenizer
    height, width = 2, 2
    image_indices = torch.tensor([0, 100, 200, 300])  # 2x2 image
    
    # Tokenize directly
    tokens = tokenizer.encapsulate_image(image_indices, height, width)
    print(f"Tokenized sequence length: {len(tokens)}")
    print(f"Token IDs: {tokens[:20]}...")  # Show first 20 tokens
    
    # Compare with original approach
    comparison = tokenizer.compare_with_original(image_indices, height, width)
    print(f"\nSpeedup: {comparison['speedup']:.2f}x")
    print(f"Tokens match: {comparison['tokens_match']}")

