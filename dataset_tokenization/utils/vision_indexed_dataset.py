#!/usr/bin/env python3
"""
Vision IndexedDataset using Official Megatron-LM Implementation
==============================================================

This module provides a wrapper around the official Megatron-LM IndexedDataset
for vision token datasets with multimodal tokenizer support.
"""

import sys
import os
import numpy as np
import torch
from typing import Union, Optional

# Try to import official Megatron implementation
try:
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
    MEGATRON_AVAILABLE = True
    print("✅ Using official Megatron-LM IndexedDatasetBuilder")
except ImportError:
    MEGATRON_AVAILABLE = False
    print("⚠️  Official Megatron-Core not available")
    print("   Install with: pip install megatron-core")
    print("   Falling back to custom implementation...")
    
    # Fallback to our custom implementation
    try:
        from .indexed_dataset_megatron import VisionTokenIndexedDatasetBuilder as FallbackBuilder
        FALLBACK_AVAILABLE = True
    except ImportError:
        FALLBACK_AVAILABLE = False


class VisionIndexedDatasetBuilder:
    """
    Vision token dataset builder using official Megatron IndexedDataset.
    
    This provides a clean interface for tokenizing images and saving them
    in Megatron-LM compatible format with proper vocabulary offset support
    for multimodal tokenizers.
    """
    
    def __init__(self, output_prefix: str, total_vocab_size: int = 2**17, text_vocab_size: int = 0):
        """
        Initialize the vision dataset builder.
        
        Args:
            output_prefix: Path prefix for output files (without extension)
            total_vocab_size: Total vocabulary size (text + image tokens)
            text_vocab_size: Size of text vocabulary (image tokens start after this)
        """
        if not MEGATRON_AVAILABLE:
            if FALLBACK_AVAILABLE:
                print("Using custom implementation as fallback")
                self._builder = FallbackBuilder(output_prefix, total_vocab_size, text_vocab_size)
                self._is_official = False
                return
            else:
                raise ImportError(
                    "Neither official Megatron-Core nor fallback implementation available.\n"
                    "Install with: pip install megatron-core"
                )
        
        self.output_prefix = output_prefix
        self.total_vocab_size = total_vocab_size
        self.text_vocab_size = text_vocab_size
        self.image_vocab_size = total_vocab_size - text_vocab_size
        self._is_official = True
        
        # Choose optimal dtype based on vocabulary size
        if total_vocab_size < 65536:
            self.dtype = np.uint16
        else:
            self.dtype = np.int32
        
        # Create official Megatron builder
        self.builder = IndexedDatasetBuilder(f"{output_prefix}.bin", dtype=self.dtype)
        
        # Statistics tracking
        self.num_images = 0
        self.total_tokens = 0
        
        # Log configuration
        self._log_configuration()
    
    def _log_configuration(self):
        """Log the dataset configuration."""
        print(f"Official Megatron IndexedDataset Configuration:")
        if self.text_vocab_size > 0:
            print(f"  - Text vocabulary: {self.text_vocab_size:,}")
            print(f"  - Image vocabulary: {self.image_vocab_size:,}")
            print(f"  - Total vocabulary: {self.total_vocab_size:,}")
            print(f"  - Image token offset: {self.text_vocab_size}")
        else:
            print(f"  - Pure vision vocabulary: {self.total_vocab_size:,}")
        print(f"  - Token dtype: {self.dtype.__name__} ({self.dtype().itemsize} bytes/token)")
    
    def add_image_tokens(self, tokens: Union[torch.Tensor, np.ndarray]):
        """
        Add tokens from a single tokenized image.
        
        Each image is treated as a separate document in the dataset.
        For multimodal tokenizers, applies vocabulary offset automatically.
        
        Args:
            tokens: Flattened token indices from image tokenizer
        """
        # Use fallback implementation if available
        if not self._is_official:
            return self._builder.add_image_tokens(tokens)
        
        # Convert to numpy array
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        
        tokens = tokens.flatten().astype(self.dtype)
        
        # Apply vocabulary offset for multimodal tokenizers
        if self.text_vocab_size > 0:
            tokens = tokens + self.text_vocab_size
            
            # Validate vocabulary bounds
            max_token = np.max(tokens)
            if max_token >= self.total_vocab_size:
                raise ValueError(
                    f"Image token {max_token} exceeds total vocabulary size {self.total_vocab_size}. "
                    f"Original token range: [{np.min(tokens - self.text_vocab_size)}, {np.max(tokens - self.text_vocab_size)}], "
                    f"Expected image vocab size: {self.image_vocab_size}"
                )
        
        # Add to Megatron builder as a document
        # Each image is a separate document with one sequence
        token_tensor = torch.from_numpy(tokens)
        self.builder.add_document(token_tensor, lengths=[len(tokens)])
        
        # Update statistics
        self.num_images += 1
        self.total_tokens += len(tokens)
    
    def finalize(self):
        """Finalize the dataset and write index file."""
        # Use fallback implementation if available
        if not self._is_official:
            return self._builder.finalize()
        
        # Finalize using official Megatron builder
        self.builder.finalize(f"{self.output_prefix}.idx")
        
        # Print completion statistics
        print(f"✅ Vision IndexedDataset created: {self.output_prefix}")
        print(f"  - Format: Official Megatron-LM compatible")
        print(f"  - Images processed: {self.num_images:,}")
        print(f"  - Total tokens: {self.total_tokens:,}")
        
        if self.num_images > 0:
            print(f"  - Avg tokens/image: {self.total_tokens / self.num_images:.1f}")
        
        # Show file sizes
        bin_path = f"{self.output_prefix}.bin"
        idx_path = f"{self.output_prefix}.idx"
        
        if os.path.exists(bin_path) and os.path.exists(idx_path):
            bin_size = os.path.getsize(bin_path)
            idx_size = os.path.getsize(idx_path)
            print(f"  - Binary file: {bin_size / 1024 / 1024:.1f} MB")
            print(f"  - Index file: {idx_size / 1024:.1f} KB")
            
            # Show storage efficiency
            if bin_size > 0:
                ratio = idx_size / bin_size * 100
                print(f"  - Index overhead: {ratio:.2f}% of binary data")


def create_builder(output_prefix: str, **kwargs) -> VisionIndexedDatasetBuilder:
    """
    Factory function to create a vision dataset builder.
    
    Args:
        output_prefix: Path prefix for output files
        **kwargs: Additional arguments for VisionIndexedDatasetBuilder
    
    Returns:
        VisionIndexedDatasetBuilder instance
    """
    return VisionIndexedDatasetBuilder(output_prefix, **kwargs)


def check_megatron_availability() -> dict:
    """
    Check availability of Megatron implementations.
    
    Returns:
        Dictionary with availability status
    """
    status = {
        'official_available': MEGATRON_AVAILABLE,
        'fallback_available': FALLBACK_AVAILABLE if not MEGATRON_AVAILABLE else None,
        'recommended_install': None
    }
    
    if not MEGATRON_AVAILABLE:
        status['recommended_install'] = [
            "pip install megatron-core",
            "# OR from source:",
            "git clone https://github.com/NVIDIA/Megatron-LM.git",
            "cd Megatron-LM && pip install -e ."
        ]
    
    return status


if __name__ == "__main__":
    """Test and demonstrate the vision indexed dataset builder."""
    
    # Check availability
    status = check_megatron_availability()
    
    print("=" * 80)
    print("VISION INDEXED DATASET BUILDER")
    print("=" * 80)
    print(f"Official Megatron: {'✅ Available' if status['official_available'] else '❌ Not available'}")
    
    if not status['official_available']:
        print(f"Fallback implementation: {'✅ Available' if status['fallback_available'] else '❌ Not available'}")
        
        if status['recommended_install']:
            print("\nTo install official Megatron-Core:")
            for line in status['recommended_install']:
                print(f"  {line}")
        
        if not status['fallback_available']:
            print("\n❌ No implementation available. Cannot run test.")
            sys.exit(1)
    
    # Run test
    print(f"\n" + "=" * 60)
    print("RUNNING TEST")
    print("=" * 60)
    
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test configuration
        text_vocab_size = 32000
        image_vocab_size = 2**17
        total_vocab_size = text_vocab_size + image_vocab_size
        
        # Create builder
        builder = VisionIndexedDatasetBuilder(
            output_prefix=f"{temp_dir}/test_vision_dataset",
            total_vocab_size=total_vocab_size,
            text_vocab_size=text_vocab_size
        )
        
        # Sample tokenized images (simulated)
        sample_images = [
            np.random.randint(0, 1000, 256),    # Image 1: 256 tokens
            np.random.randint(0, 1000, 128),    # Image 2: 128 tokens
            np.random.randint(0, 1000, 512),    # Image 3: 512 tokens
            np.random.randint(0, 1000, 64),     # Image 4: 64 tokens
        ]
        
        print(f"Adding {len(sample_images)} sample tokenized images:")
        for i, tokens in enumerate(sample_images):
            print(f"  Image {i+1}: {len(tokens)} tokens, range [{np.min(tokens)}, {np.max(tokens)}]")
            builder.add_image_tokens(tokens)
        
        # Finalize dataset
        builder.finalize()
        
        print(f"\n✅ Test completed successfully!")
        
        # Show what files were created
        created_files = []
        for ext in ['.bin', '.idx']:
            file_path = f"{temp_dir}/test_vision_dataset{ext}"
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                created_files.append(f"  - test_vision_dataset{ext}: {size:,} bytes")
        
        if created_files:
            print(f"\nFiles created:")
            for file_info in created_files:
                print(file_info)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        shutil.rmtree(temp_dir)
    
    print(f"\n" + "=" * 80)
    print("USAGE IN YOUR PIPELINE")
    print("=" * 80)
    print("Replace your current builder with:")
    print("  from utils.vision_indexed_dataset import VisionIndexedDatasetBuilder")
    print("  builder = VisionIndexedDatasetBuilder(output_prefix, total_vocab_size, text_vocab_size)")
    print("  # Same API as before, but uses official Megatron when available")