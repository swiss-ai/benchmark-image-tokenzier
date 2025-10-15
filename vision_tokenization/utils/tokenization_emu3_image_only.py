
#!/usr/bin/env python3
"""
EMU3 tokenization for image-only data.
Avoids double tokenization by directly merging image indices with text tokens.
"""

import torch
from typing import List, Dict, Any
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
    
    def __init__(self,
                 text_tokenizer_path: str,
                 device: str = "cuda",
                 min_pixels: int = None,
                 max_pixels: int = None):
        """
        Initialize with text tokenizer that has EMU3 vision tokens and image tokenizer.

        Args:
            text_tokenizer_path: Path to text tokenizer with EMU3 tokens
            device: Device for image tokenizer (default: "cuda")
            min_pixels: Minimum pixels for image preprocessing (default: 512*512)
            max_pixels: Maximum pixels for image preprocessing (default: 1024*1024)
        """

        # Store device
        self.device = device

        # Load tokenizer with trust_remote_code for custom EMU3Tokenizer class
        # Use fast tokenizer for better performance
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_tokenizer_path,
            trust_remote_code=True,
            use_fast=True
        )

        # Set default values for pixels if not provided
        if min_pixels is None:
            min_pixels = 384 * 384
        if max_pixels is None:
            max_pixels = 1024 * 1024

        self.image_tokenizer = Emu3VisionTokenizer(
            device=self.device,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        
        # Cache for dimension tokens to avoid repeated encodingYOI
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

    @torch.inference_mode()
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
        token_ids_no_eos_bos = token_ids[1:-1]  # Remove BOS and EOS for text conversion
        # Decode to text using the text tokenizer
        text = self.text_tokenizer.decode(token_ids_no_eos_bos, skip_special_tokens=False)

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



class EMU3ImageTextPairTokenizer(EMU3ImageOnlyTokenizer):
    """
    Extended tokenizer for image-text pairs with parallel GPU/CPU processing.
    Image tokenization happens on GPU while text tokenization happens on CPU in parallel.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with same parameters as parent class."""
        super().__init__(*args, **kwargs)
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="TokenizerPool")

    def tokenize_image_text_pair(
        self,
        image,
        text: str,
    ) -> torch.Tensor:
        """
        Tokenize an image-text pair with parallel processing using ThreadPoolExecutor.
        Image is processed on GPU while text is processed on CPU simultaneously.

        Args:
            image: PIL Image to tokenize
            text: Text string to append after image

        Returns:
            Combined tokens: [BOS] + [image tokens without EOS] + [text tokens] + [EOS]
        """
        def tokenize_text_cpu():
            """CPU thread for text tokenization."""
            # Force text tokenization to CPU
            with torch.cuda.device(-1):  # Use CPU
                text_tokens_dict = self.text_tokenizer(
                    text,
                    truncation=False,
                    add_special_tokens=False,
                    return_tensors="pt"
                )
                return text_tokens_dict['input_ids'].squeeze(0)

        # Submit both tasks to executor
        # Image on GPU (usually the bottleneck)
        image_future = self.executor.submit(self.tokenize_image, image)

        # Text on CPU (fast, runs in parallel)
        text_future = self.executor.submit(tokenize_text_cpu)

        # Wait for both and get results
        image_tokens = image_future.result()
        text_tokens = text_future.result()

        # Move text tokens to same device as image tokens for concatenation
        text_tokens = text_tokens.to(image_tokens.device)

        # Combine using cat (can't pre-allocate without knowing text length)
        combined_tokens = torch.cat([
            image_tokens[:-1],  # Image tokens without EOS
            text_tokens,        # Text tokens
            image_tokens[-1:]   # EOS token
        ])

        return combined_tokens

    def tokenize_image_text_pair_sequential(
        self,
        image,
        text: str,
    ) -> torch.Tensor:
        """
        Sequential version for comparison/debugging.
        Same as parent but kept for benchmarking.
        """
        # Get image tokens using parent's tokenize_image method
        image_tokens = self.tokenize_image(image)

        # Tokenize text without special tokens (no BOS/EOS)
        text_tokens_dict = self.text_tokenizer(
            text,
            truncation=False,
            add_special_tokens=False,
            return_tensors="pt"
        )
        text_tokens = text_tokens_dict['input_ids'].squeeze(0)

        # Combine
        combined_tokens = torch.cat([
            image_tokens[:-1],
            text_tokens,
            image_tokens[-1:]
        ])

        return combined_tokens


class EMU3ImageSftDataTokenizer(EMU3ImageOnlyTokenizer):
    """
    Tokenizer for SFT (Supervised Fine-Tuning) data with a single image and text.
    Optimized for single image per conversation (most common case).
    Replaces <|image|> placeholder with actual Emu3 vision tokens.

    Designed for SFT/pretraining with FineVision-style data where conversations
    already include assistant responses.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with same parameters as parent class."""
        super().__init__(*args, **kwargs)

        # Initialize ThreadPoolExecutor for parallel processing
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="TokenizerPool")

        # Cache the image token ID for faster lookup
        self.image_token_id = self.text_tokenizer.convert_tokens_to_ids("<|image|>")

    def _add_special_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Add BOS and EOS tokens if missing.

        This is necessary because:
        - Some chat templates hardcode BOS/EOS in the template (e.g., {{- bos_token }})
        - The add_special_tokens parameter in apply_chat_template is often ignored
        - Different templates have different behavior (some add BOS, some don't)
        - We need to verify and add only if missing to avoid double BOS/EOS tokens

        Uses efficient torch.cat approach (17% faster than F.pad).

        Args:
            tokens: 1D tensor of token IDs (non-empty from apply_chat_template)

        Returns:
            Token tensor with BOS at start and EOS at end
        """
        # Check what's missing (bos_id and eos_id guaranteed non-None by assertions)
        needs_bos = tokens[0] != self.bos_id
        needs_eos = tokens[-1] != self.eos_id

        # Fast path: nothing to do
        if not needs_bos and not needs_eos:
            return tokens

        # Build result with torch.cat
        parts = []
        if needs_bos:
            parts.append(torch.tensor([self.bos_id], dtype=tokens.dtype, device=tokens.device))
        parts.append(tokens)
        if needs_eos:
            parts.append(torch.tensor([self.eos_id], dtype=tokens.dtype, device=tokens.device))

        return torch.cat(parts)

    @torch.inference_mode()
    def tokenize_conversation(
        self,
        messages: List[Dict[str, Any]],
        image: Any = None
    ) -> torch.Tensor:
        """
        Tokenize a conversation with a single image and text.
        Uses parallel processing: text on CPU, image on GPU.

        Args:
            messages: List of message dicts with role and content
                     Content can be string or list of dicts with type="image" or type="text"
            image: Single PIL image corresponding to <|image|> placeholder

        Returns:
            Token tensor with <|image|> placeholder replaced by Emu3 vision tokens
        """
        def tokenize_text_cpu():
            """CPU thread for text tokenization."""
            # Force text operations to CPU
            with torch.cuda.device(-1):  # Use CPU
                try:
                    # Apply chat template and tokenize directly
                    # Note: This does NOT add BOS/EOS tokens automatically
                    text_tokens = self.text_tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,  # Get tokens directly
                        add_generation_prompt=False,
                        return_tensors="pt"
                    )

                    # Safely squeeze - keep at least 1D
                    if text_tokens.ndim > 1:
                        text_tokens = text_tokens.squeeze(0)
                    text_tokens = text_tokens.cpu()  # Ensure CPU tensor

                    # Add BOS/EOS tokens if missing (handles different chat templates)
                    text_tokens = self._add_special_tokens(text_tokens)

                    # Step 3: Find image position
                    if len(text_tokens) == 0:
                        print(f"Warning: Empty text tokens after encoding chat_text")
                        return text_tokens, 0, None

                    image_mask = (text_tokens == self.image_token_id)

                    # Check if image_mask is a tensor (should always be true)
                    if not torch.is_tensor(image_mask):
                        print(f"ERROR: image_mask is not a tensor: {type(image_mask)}")
                        return text_tokens, 0, None

                    num_images = image_mask.sum().item() if image_mask.numel() > 0 else 0

                    if num_images == 1:
                        image_position = image_mask.nonzero(as_tuple=True)[0][0].item()
                    else:
                        image_position = None

                    return text_tokens, num_images, image_position

                except Exception as e:
                    print(f"Error in tokenize_text_cpu: {e}")
                    import traceback
                    traceback.print_exc()
                    return torch.tensor([], dtype=torch.long), 0, None

        # Check if image exists before starting parallel processing
        if image is None:
            print("Warning: No image provided to tokenize_conversation")
            return torch.tensor([], dtype=torch.long)

        # Submit both tasks in parallel
        text_future = self.executor.submit(tokenize_text_cpu)
        image_future = self.executor.submit(lambda: self.tokenize_image(image)[1:-1])  # GPU thread

        # Wait for both results
        text_tokens, num_images, image_position = text_future.result()

        if num_images != 1:
            # Only single image samples are supported
            print(f"Warning: Found {num_images} image placeholders, expected 1. Skipping sample.")
            return torch.tensor([], dtype=torch.long)

        image_tokens = image_future.result()

        # Move text tokens to same device as image tokens for final assembly
        text_tokens = text_tokens.to(image_tokens.device)

        # Replace <|image|> placeholder with actual vision tokens
        final_tokens = self._replace_single_image(
            text_tokens,
            image_position,
            image_tokens
        )

        return final_tokens

    def _replace_single_image(
        self,
        text_tokens: torch.Tensor,
        image_position: int,
        image_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Replace single <|image|> placeholder with vision tokens.
        Optimized using torch.cat for better memory efficiency.

        Args:
            text_tokens: Original tokens with <|image|> placeholder
            image_position: Position of <|image|> token
            image_tokens: Tokenized image (without BOS/EOS)

        Returns:
            Final token tensor with image inserted
        """
        # Use torch.cat which is optimized at C++ level
        parts = []

        # Text before image
        if image_position > 0:
            parts.append(text_tokens[:image_position])

        # Image tokens
        parts.append(image_tokens)

        # Text after image placeholder
        if image_position + 1 < len(text_tokens):
            parts.append(text_tokens[image_position + 1:])

        return torch.cat(parts, dim=0)

    def process_finevision_sample(
        self,
        texts: List[Dict[str, str]],
        image: Any
    ) -> torch.Tensor:
        """
        Process FineVision-style texts with a single image.
        Designed for Ray pipeline where image and texts are loaded separately.

        Args:
            texts: List of {"user": str, "assistant": str} dicts
            image: Single PIL image

        Returns:
            Tokenized tensor ready for model input
        """
        # Check input
        if not texts or not isinstance(texts, list):
            print(f"Warning: Invalid texts input: type={type(texts)}, len={len(texts) if texts else 0}")
            return torch.tensor([], dtype=torch.long)

        if not image:
            print(f"Warning: No image provided to process_finevision_sample")
            return torch.tensor([], dtype=torch.long)

        # Convert FineVision format to standard message format
        messages = []

        # Add image placeholder to the first user message
        for i, conv in enumerate(texts):
            if not isinstance(conv, dict) or 'user' not in conv or 'assistant' not in conv:
                print(f"Warning: Invalid conversation format at index {i}: {conv}")
                continue

            # Add user message
            if i == 0:
                # First message gets the image placeholder
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": conv['user']}
                    ]
                })
            else:
                messages.append({
                    "role": "user",
                    "content": conv['user']
                })

            # Add assistant message
            messages.append({
                "role": "assistant",
                "content": conv['assistant']
            })

        if not messages:
            print("Warning: No valid messages created from texts")
            return torch.tensor([], dtype=torch.long)

        # Process messages without debug output

        # Tokenize with single image
        return self.tokenize_conversation(messages, image)