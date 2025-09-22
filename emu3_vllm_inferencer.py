#!/usr/bin/env python3
"""
Simple EMU3 vLLM Inferencer for Notebook Use
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import re


class EMU3Inferencer:
    """Simple EMU3 inference class for testing in notebooks."""
    
    def __init__(
        self,
        model_path: str = "/iopsstor/scratch/cscs/nirmiger/Megatron-LM/logs/Meg-Runs/image-extension/llama3-3b-2n-8192sl-120gbsz-0.9-0.1/HF",
        tokenizer_path: str = "/capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3_tokenizer",
        tensor_parallel_size: Optional[int] = None,
        max_model_len: int = 8192
    ):
        """
        Initialize EMU3 inferencer.
        
        Args:
            model_path: Path to model checkpoint
            tokenizer_path: Path to tokenizer
            tensor_parallel_size: Number of GPUs for tensor parallelism (auto-detect if None)
            max_model_len: Maximum model sequence length
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        
        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load vision token mapping and special token IDs
        self.vision_mapping = self._load_vision_mapping()
        self.special_token_ids = self._get_special_token_ids()
        
        # Auto-detect tensor parallel size
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        # Initialize vLLM
        print(f"Loading model from {model_path}")
        print(f"Using {tensor_parallel_size} GPUs for tensor parallelism")
        
        self.llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            dtype="auto",
            max_model_len=max_model_len
        )
        
        print("EMU3 Inferencer ready!")
    
    def _load_vision_mapping(self) -> Dict[int, int]:
        """Load vision token mapping (visual_index -> token_id)."""
        mapping_path = os.path.join(self.tokenizer_path, "vision_token_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                data = json.load(f)
                # Convert string keys to integers
                return {int(k): v for k, v in data.get("vision_token_ids", {}).items()}
        return {}
    
    def _get_special_token_ids(self) -> Dict[str, int]:
        """Get special token IDs."""
        special_tokens = {}
        token_names = {
            "img_start": "<|img_start|>",
            "img_end": "<|img_end|>",
            "img_token_start": "<|img_token_start|>",
            "img_end_of_row": "<|img_end_of_row|>",
            "img_end_of_frame": "<|img_end_of_frame|>"
        }
        
        for name, token in token_names.items():
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id != self.tokenizer.unk_token_id:
                special_tokens[name] = token_id
        
        return special_tokens
    
    def generate(
        self,
        prompt: Union[str, List[int]],
        max_tokens: int = 500,
        min_tokens: int = 10,
        temperature: float = 0.3,
        top_p: float = 0.95,
        stop_token_ids: Optional[List[int]] = None,
        return_full_output: bool = True
    ) -> Dict:
        """
        Generate from a prompt (text or token IDs).
        
        Args:
            prompt: Input prompt (text string or list of token IDs)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            stop_token_ids: Stop token IDs (uses img_end by default)
            return_full_output: Return full output dict including token IDs
            
        Returns:
            Dictionary with generated tokens and metadata
        """
        # Set default stop tokens
        if stop_token_ids is None and "img_end" in self.special_token_ids:
            stop_token_ids = [self.special_token_ids["img_end"]]
        
        # Convert prompt to token IDs if it's text
        if isinstance(prompt, str):
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            prompt_ids = prompt
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False
        )
        
        # Generate with token IDs
        outputs = self.llm.generate(
            prompt_token_ids=[prompt_ids],
            sampling_params=sampling_params
        )
        
        output = outputs[0].outputs[0]
        
        # Get generated token IDs
        generated_ids = output.token_ids
        
        # Extract visual tokens from generated IDs
        visual_indices = self.extract_visual_indices_from_tokens(generated_ids)
        
        result = {
            "generated_token_ids": generated_ids,
            "generated_text": self.tokenizer.decode(generated_ids, skip_special_tokens=False),
            "visual_indices": visual_indices,
            "num_visual_tokens": len(visual_indices),
            "prompt_length": len(prompt_ids)
        }
        
        # Add structure token counts
        if generated_ids:
            result["structure_tokens"] = self.count_structure_tokens(generated_ids)
        
        return result
    
    def extract_visual_indices_from_tokens(self, token_ids: List[int]) -> List[int]:
        """
        Extract visual token indices from token IDs.
        
        Args:
            token_ids: List of token IDs from model output
            
        Returns:
            List of visual token indices (0-32767)
        """
        visual_indices = []
        
        # Create reverse mapping from token_id to visual index
        reverse_mapping = {v: k for k, v in self.vision_mapping.items()}
        
        for token_id in token_ids:
            if token_id in reverse_mapping:
                visual_indices.append(reverse_mapping[token_id])
        
        return visual_indices
    
    def count_structure_tokens(self, token_ids: List[int]) -> Dict[str, int]:
        """Count structure tokens in token ID list."""
        counts = {}
        for name, token_id in self.special_token_ids.items():
            counts[name] = token_ids.count(token_id)
        return counts
    
    def predict_structure(self, height: int, width: int) -> Dict:
        """
        Test structure prediction from metadata.
        
        Args:
            height: Image height in tokens
            width: Image width in tokens
            
        Returns:
            Dictionary with prediction results
        """
        # Create metadata prompt as token IDs
        prompt_text = f"<|img_start|>{height}*{width}<|img_token_start|>"
        # Note: add_special_tokens=True would add BOS token if model expects it
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        
        # Generate
        result = self.generate(
            prompt_ids, 
            max_tokens=8192,  # Let model decide when to stop
            temperature=0.1
        )
        
        # Analyze results
        analysis = {
            "prompt": prompt_text,
            "expected_dims": f"{height}x{width}",
            "expected_tokens": height * width,
            "found_visual_tokens": result["num_visual_tokens"],
            "visual_indices": result["visual_indices"],  # All visual indices
            "all_token_ids": result["generated_token_ids"],  # All generated token IDs
            "structure_tokens": result["structure_tokens"],
            "generated_sample": result["generated_text"][:300]
        }
        
        # Check if structure matches
        expected_eol = height  # One EOL after each row
        analysis["structure_correct"] = (
            result["structure_tokens"].get("img_end_of_row", 0) == expected_eol and
            result["num_visual_tokens"] == height * width
        )
        
        return analysis
    
    def complete_image(self, height: int, width: int, given_rows: int = 1) -> Dict:
        """
        Complete a partial image using token IDs.
        
        Args:
            height: Total image height
            width: Total image width
            given_rows: Number of rows to provide as context
            
        Returns:
            Dictionary with completion results
        """
        # Build partial prompt as token IDs
        prompt_parts = []
        
        # Add metadata tokens
        prompt_parts.extend(self.tokenizer.encode(
            f"<|img_start|>{height}*{width}<|img_token_start|>", 
            add_special_tokens=False
        ))
        
        # Add visual tokens for given rows
        token_idx = 0
        for row in range(min(given_rows, height)):
            for col in range(width):
                # Get token ID for this visual index
                if token_idx in self.vision_mapping:
                    prompt_parts.append(self.vision_mapping[token_idx])
                token_idx += 1
            
            # Add end of row token after each row (including the last given row)
            if "img_end_of_row" in self.special_token_ids:
                prompt_parts.append(self.special_token_ids["img_end_of_row"])
        
        # Calculate expected tokens
        total_tokens = height * width
        provided_tokens = given_rows * width
        expected_remaining = total_tokens - provided_tokens
        
        # Generate completion
        result = self.generate(
            prompt_parts,
            max_tokens=expected_remaining * 5,  # Allow some extra for structure tokens
            temperature=0.3
        )
        
        # Analyze completion
        analysis = {
            "height": height,
            "width": width,
            "given_rows": given_rows,
            "provided_tokens": provided_tokens,
            "expected_remaining": expected_remaining,
            "generated_visual_tokens": result["num_visual_tokens"],
            "all_visual_indices": result["visual_indices"],
            "structure_tokens": result["structure_tokens"],
            "completion_correct": result["num_visual_tokens"] == expected_remaining
        }
        
        return analysis
    
    def visualize_tokens(
        self, 
        visual_indices: List[int], 
        height: int, 
        width: int, 
        title: str = "Visual Token Grid"
    ):
        """
        Visualize visual token indices as a grid.
        
        Args:
            visual_indices: List of visual token indices (0-32767)
            height: Grid height
            width: Grid width
            title: Plot title
        """
        # Create grid
        grid = np.full((height, width), -1, dtype=int)
        
        for i, idx in enumerate(visual_indices[:height*width]):
            row = i // width
            col = i % width
            if row < height and col < width:
                grid[row, col] = idx
        
        # Plot
        fig, ax = plt.subplots(figsize=(max(8, width), max(6, height)))
        
        # Use masked array for unfilled cells
        masked_grid = np.ma.masked_where(grid == -1, grid)
        im = ax.imshow(masked_grid, cmap='viridis', aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Visual Token Index')
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        # Add text annotations for small grids
        if height <= 8 and width <= 8:
            for i in range(height):
                for j in range(width):
                    if grid[i, j] != -1:
                        ax.text(j, i, str(grid[i, j]), 
                               ha='center', va='center', 
                               color='white' if grid[i, j] < grid.max()/2 else 'black',
                               fontsize=8)
        
        ax.set_title(title)
        ax.set_xlabel('Width (tokens)')
        ax.set_ylabel('Height (tokens)')
        
        plt.tight_layout()
        return fig
    
    def decode_visual_sequence(self, token_ids: List[int]) -> Dict:
        """
        Decode a sequence of token IDs to understand the image structure.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Dictionary with decoded structure information
        """
        visual_indices = self.extract_visual_indices_from_tokens(token_ids)
        structure_counts = self.count_structure_tokens(token_ids)
        
        # Try to infer dimensions from structure
        num_eol = structure_counts.get("img_end_of_row", 0)
        num_visual = len(visual_indices)
        
        # Estimate dimensions
        if num_eol > 0:
            estimated_height = num_eol + 1
            estimated_width = num_visual // estimated_height if estimated_height > 0 else 0
        else:
            # Assume square if no EOL tokens
            estimated_height = int(np.sqrt(num_visual))
            estimated_width = estimated_height
        
        return {
            "visual_indices": visual_indices,
            "num_visual_tokens": num_visual,
            "structure_tokens": structure_counts,
            "estimated_dims": f"{estimated_height}x{estimated_width}",
            "text": self.tokenizer.decode(token_ids, skip_special_tokens=False)[:500]
        }
    
    def batch_generate(self, prompts: List[Union[str, List[int]]], **kwargs) -> List[Dict]:
        """
        Generate for multiple prompts in batch.
        
        Args:
            prompts: List of prompts (text or token IDs)
            **kwargs: Additional arguments for sampling
            
        Returns:
            List of result dictionaries
        """
        # Convert all prompts to token IDs
        prompt_ids_list = []
        for prompt in prompts:
            if isinstance(prompt, str):
                prompt_ids_list.append(self.tokenizer.encode(prompt, add_special_tokens=False))
            else:
                prompt_ids_list.append(prompt)
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=kwargs.get('temperature', 0.3),
            top_p=kwargs.get('top_p', 0.95),
            max_tokens=kwargs.get('max_tokens', 500),
            stop_token_ids=kwargs.get('stop_token_ids', [self.special_token_ids.get("img_end", -1)]),
            skip_special_tokens=False
        )
        
        # Generate
        outputs = self.llm.generate(
            prompt_token_ids=prompt_ids_list,
            sampling_params=sampling_params
        )
        
        # Process outputs
        results = []
        for output in outputs:
            token_ids = output.outputs[0].token_ids
            visual_indices = self.extract_visual_indices_from_tokens(token_ids)
            
            results.append({
                "token_ids": token_ids,
                "visual_indices": visual_indices,
                "num_visual_tokens": len(visual_indices),
                "text": self.tokenizer.decode(token_ids, skip_special_tokens=False)
            })
        
        return results