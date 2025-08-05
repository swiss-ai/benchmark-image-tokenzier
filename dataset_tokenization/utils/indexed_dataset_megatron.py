"""
Megatron-LM Compatible IndexedDataset Implementation for Vision Tokenization
===========================================================================

This is the exact implementation from /iopsstor/scratch/cscs/xyixuan/PDM/notebooks/create_meg_files.ipynb
with minor adaptations for our tokenization pipeline.

This format is directly compatible with Megatron-LM's data loaders.

Workflow for Vision Token Datasets
----------------------------------

1. **Dataset Creation**:
   ```python
   # Initialize builder with vocabulary configuration
   builder = VisionTokenIndexedDatasetBuilder(
       output_prefix="path/to/dataset",
       image_vocab_size=131072,  # e.g., Emu3 vocab size
       text_vocab_size=0          # 0 for pure vision, >0 for multimodal
   )
   ```

2. **Processing Images**:
   ```python
   # For each image in your dataset:
   for image_path in image_paths:
       # Tokenize image using your vision tokenizer (e.g., Emu3)
       tokens = vision_tokenizer.encode(image)  # Returns token indices
       
       # Add to dataset (automatically handles vocab offset if multimodal)
       builder.add_image_tokens(tokens)
   ```

3. **Finalization**:
   ```python
   # Write the dataset files (.bin and .idx)
   builder.finalize()
   ```

4. **Output Files**:
   - `dataset.bin`: Binary file containing token data
   - `dataset.idx`: Index file with metadata and offsets

5. **Loading in Megatron-LM**:
   ```python
   # In your Megatron training script:
   from megatron.data.indexed_dataset import IndexedDataset
   dataset = IndexedDataset("path/to/dataset")
   ```

Key Features
------------
- **Multimodal Support**: Handles vocabulary offset for text+vision tokenizers
- **Optimal Storage**: Auto-selects uint16 for vocab < 65,500 (saves 50% space)
- **Document Structure**: Each image is stored as a separate document
- **Megatron Compatible**: Uses official MMIDIDX header format

Example Usage Scenarios
----------------------
1. **Pure Vision Tokenizer** (e.g., standalone Emu3):
   - Set text_vocab_size=0
   - Tokens stored as-is [0, vocab_size)

2. **Multimodal Tokenizer** (e.g., SwissGPT + Emu3):
   - Set text_vocab_size=32000 (text tokenizer size)
   - Vision tokens automatically offset: token 0 → 32000, token 1 → 32001, etc.
   - Total vocab = text_vocab + vision_vocab

File Format Details
------------------
- Header: MMIDIDX\\x00\\x00 (Megatron memory-mapped format)
- Index file structure:
  - Header (9 bytes)
  - Version (8 bytes)
  - Dtype code (1 byte)
  - Sequence count (8 bytes)
  - Document count (8 bytes)
  - Sequence lengths array (int32)
  - Sequence pointers array (int64)
  - Document indices array (int64)
  - [Optional] Sequence modes array (int8) - for multimodal datasets
"""

import os
import struct
import numpy as np
import torch
from typing import List, Optional, Type, Union
from enum import Enum

# Import the vocabulary detection utility (optional - for standalone usage only)
# Note: This is for convenience in single-process scenarios. In distributed
# settings, detect vocabulary size once on the main process and pass as parameter.
try:
    from .detect_vocab_size import detect_text_vocab_size
except ImportError:
    detect_text_vocab_size = None


# Fixed header for Megatron format
_INDEX_HEADER = b"MMIDIDX\x00\x00"


class DType(Enum):
    """The NumPy data type Enum for writing/reading the IndexedDataset indices"""
    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, value: Type[np.number]) -> int:
        """Get the code from the dtype"""
        return cls[value.__name__].value

    @classmethod
    def dtype_from_code(cls, value: int) -> Type[np.number]:
        """Get the dtype from the code"""
        return getattr(np, cls(value).name)

    @staticmethod
    def size(key: Union[int, Type[np.number]]) -> int:
        """Get the size of the dtype/code in bytes"""
        if isinstance(key, int):
            return DType.dtype_from_code(key)().itemsize
        elif np.number in key.__mro__:
            return key().itemsize
        else:
            raise ValueError

    @staticmethod
    def optimal_dtype(cardinality: Optional[int]) -> Type[np.number]:
        """Get the dtype to use for an index of a certain cardinality
        
        For vision tokenizers, if vocab_size < 65500, we can use uint16 (2 bytes per token)
        Otherwise we need int32 (4 bytes per token)
        """
        if cardinality is not None and cardinality < 65500:
            return np.uint16
        else:
            return np.int32


class IndexedDatasetBuilder:
    """Builder class for the IndexedDataset class
    
    This is the exact implementation from the reference notebook.
    
    Args:
        bin_path (str): The path to the data (.bin) file
        dtype (Type[np.number], optional): The dtype of the index file. Defaults to np.int32.
        multimodal (bool, optional): Whether the dataset is multimodal. Defaults to False.
    """

    def __init__(
        self, bin_path: str, dtype: Type[np.number] = np.int32, multimodal: bool = False
    ) -> None:
        self.data_file = open(bin_path, "wb")
        self.dtype = dtype
        self.multimodal = multimodal

        self.sequence_lengths = []
        self.document_indices = [0]
        self.sequence_modes = [] if self.multimodal else None

    def add_item(self, tensor: torch.Tensor, mode: int = 0) -> None:
        """Add a single item to the dataset

        Args:
            tensor (torch.Tensor): The item to add to the data file
            mode (int, optional): The mode for the item. Defaults to 0.
        """
        np_array = np.array(tensor.numpy() if hasattr(tensor, 'numpy') else tensor, dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.append(np_array.size)
        if self.multimodal:
            self.sequence_modes.append(mode)

    def add_document(
        self, tensor: Union[torch.Tensor, List[int]], lengths: List[int], modes: Optional[List[int]] = None
    ) -> None:
        """Add an entire document to the dataset

        Args:
            tensor (torch.Tensor or List[int]): The document to add
            lengths (List[int]): The lengths of each item in the document
            modes (Optional[List[int]], optional): The modes for each item in the document. Defaults to None.
        """
        np_array = np.array(tensor, dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.extend(lengths)
        self.document_indices.append(len(self.sequence_lengths))
        if self.multimodal:
            self.sequence_modes.extend(modes if modes is not None else [0] * len(lengths))

    def end_document(self) -> None:
        """Finalize the document, for use with IndexedDatasetBuilder.add_item"""
        self.document_indices.append(len(self.sequence_lengths))

    def finalize(self, idx_path: str) -> None:
        """Clean up and write the index (.idx) file

        Args:
            idx_path (str): The path to the index file
        """
        self.data_file.close()
        
        with open(idx_path, 'wb') as idx_writer:
            # Write header
            idx_writer.write(_INDEX_HEADER)
            # Write version
            idx_writer.write(struct.pack("<Q", 1))
            # Write dtype code
            idx_writer.write(struct.pack("<B", DType.code_from_dtype(self.dtype)))
            
            # Write counts
            sequence_count = len(self.sequence_lengths)
            idx_writer.write(struct.pack("<Q", sequence_count))
            document_count = len(self.document_indices) - 1
            idx_writer.write(struct.pack("<Q", document_count))
            
            # Write sequence lengths
            sequence_lengths = np.array(self.sequence_lengths, dtype=np.int32)
            idx_writer.write(sequence_lengths.tobytes(order="C"))
            
            # Write sequence pointers (byte offsets)
            sequence_pointers = self._sequence_pointers(self.sequence_lengths)
            sequence_pointers = np.array(sequence_pointers, dtype=np.int64)
            idx_writer.write(sequence_pointers.tobytes(order="C"))
            
            # Write document indices
            document_indices = np.array(self.document_indices, dtype=np.int64)
            idx_writer.write(document_indices.tobytes(order="C"))
            
            # Write sequence modes if multimodal
            if self.sequence_modes is not None:
                sequence_modes = np.array(self.sequence_modes, dtype=np.int8)
                idx_writer.write(sequence_modes.tobytes(order='C'))

    def _sequence_pointers(self, sequence_lengths: List[int]) -> List[int]:
        """Build the sequence pointers per the sequence lengths and dtype size"""
        itemsize = DType.size(self.dtype)
        curr_ptr = 0
        list_ptr = []
        for length in sequence_lengths:
            list_ptr.append(curr_ptr)
            curr_ptr += length * itemsize
        return list_ptr


def get_idx_path(path_prefix: str) -> str:
    """Get the path to the index file from the prefix"""
    return path_prefix + ".idx"


def get_bin_path(path_prefix: str) -> str:
    """Get the path to the data file from the prefix"""
    return path_prefix + ".bin"


# Convenience wrapper for our use case
class VisionTokenIndexedDatasetBuilder:
    """
    Specialized builder for vision token datasets with multimodal tokenizer support.
    
    This wraps the IndexedDatasetBuilder with optimizations for vision tokens:
    - Automatically selects optimal dtype based on vocabulary size
    - Treats each image as a document
    - Handles vocabulary offset for multimodal tokenizers
    - Provides simple API for adding tokenized images
    """
    
    def __init__(self, output_prefix: str, image_vocab_size: int, text_vocab_size: int):
        """
        Initialize the builder.
        
        Args:
            output_prefix: Path prefix for output files (without extension)
            image_vocab_size: Image vocabulary size (e.g., 131072 for Emu3)
            text_vocab_size: Size of the text vocabulary (image tokens start after this)
                           If 0, no offset is applied (pure vision tokenizer)
                           
        Note for distributed usage:
            In distributed settings, determine text_vocab_size once before launching workers:
            
            # On main process only:
            from dataset_tokenization.utils.detect_vocab_size import detect_text_vocab_size
            vocab_info = detect_text_vocab_size("your-tokenizer")
            text_vocab_size = vocab_info['vocab_size']
            
            # Then pass text_vocab_size as argument to all workers
        """
        self.output_prefix = output_prefix
        self.text_vocab_size = text_vocab_size
        self.image_vocab_size = image_vocab_size
        self.total_vocab_size = text_vocab_size + image_vocab_size
        
        # Choose optimal dtype based on total vocabulary size
        self.dtype = DType.optimal_dtype(self.total_vocab_size)
        
        # Create the builder
        self.builder = IndexedDatasetBuilder(
            bin_path=get_bin_path(output_prefix),
            dtype=self.dtype
        )
        
        # Track statistics
        self.num_images = 0
        self.total_tokens = 0
        
        # Log configuration
        if text_vocab_size > 0:
            print(f"Multimodal tokenizer configuration:")
            print(f"  - Text vocabulary size: {text_vocab_size:,}")
            print(f"  - Image vocabulary size: {self.image_vocab_size:,}")
            print(f"  - Total vocabulary size: {self.total_vocab_size:,}")
            print(f"  - Image token offset: {text_vocab_size}")
            print(f"  - Token dtype: {self.dtype.__name__}")
        else:
            print(f"Pure image tokenizer configuration:")
            print(f"  - Image vocabulary size: {image_vocab_size:,}")
            print(f"  - Token dtype: {self.dtype.__name__}")
        
    def add_image_tokens(self, tokens: Union[torch.Tensor, np.ndarray]):
        """
        Add tokens from a single image with proper vocabulary offset.
        
        Each image is treated as a separate document.
        
        Args:
            tokens: Flattened token indices from the image (from vision tokenizer)
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        
        # Flatten if needed
        tokens = tokens.flatten()
        
        # Apply vocabulary offset for multimodal tokenizers
        if self.text_vocab_size > 0:
            # Image tokens start after the text vocabulary
            # e.g., if text vocab is 50000, image token 0 becomes 50000, token 1 becomes 50001, etc.
            tokens = tokens + self.text_vocab_size
            
            # Validate that tokens don't exceed total vocabulary
            max_token = np.max(tokens)
            if max_token >= self.total_vocab_size:
                raise ValueError(
                    f"Image token {max_token} exceeds total vocabulary size {self.total_vocab_size}. "
                    f"Check your vocabulary configuration."
                )
        
        # Add as a document with single sequence
        self.builder.add_document(tokens, lengths=[len(tokens)])
        
        # Update statistics
        self.num_images += 1
        self.total_tokens += len(tokens)
        
    def finalize(self):
        """Finalize the dataset and print statistics."""
        # Write index file
        self.builder.finalize(get_idx_path(self.output_prefix))
        
        # Print statistics
        print(f"✓ Created IndexedDataset: {self.output_prefix}")
        print(f"  - Format: Megatron-LM compatible")
        print(f"  - Images: {self.num_images:,}")
        print(f"  - Total tokens: {self.total_tokens:,}")
        print(f"  - Avg tokens/image: {self.total_tokens / max(1, self.num_images):.1f}")
        print(f"  - Token dtype: {self.dtype.__name__} ({DType.size(self.dtype)} bytes/token)")
        print(f"  - Binary file size: {os.stat(get_bin_path(self.output_prefix)).st_size / 1024 / 1024:.1f} MB")
        print(f"  - Index file size: {os.stat(get_idx_path(self.output_prefix)).st_size / 1024:.1f} KB")


if __name__ == "__main__":
    """
    Example usage of the Megatron IndexedDataset builders.
    
    This demonstrates how to create IndexedDatasets for both pure vision
    and multimodal tokenization scenarios.
    """
    import numpy as np
    import tempfile
    import shutil
    
    print("=" * 80)
    print("Megatron IndexedDataset Builder Examples")
    print("=" * 80)
    
    # Create temporary directory for examples
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Example 1: Basic IndexedDatasetBuilder (reference format)
        print("\n" + "=" * 60)
        print("Example 1: Basic IndexedDatasetBuilder")
        print("=" * 60)
        print("This is the exact format from the reference notebook.")
        
        # Create builder
        vocab_size = 2**17  # Emu3 vocabulary size
        builder1 = IndexedDatasetBuilder(
            bin_path=f"{temp_dir}/example1.bin",
            dtype=DType.optimal_dtype(vocab_size)
        )
        
        # Add some sample documents (each representing a tokenized image)
        sample_images = [
            list(range(100)),           # Image 1: 100 tokens
            list(range(200, 350)),      # Image 2: 150 tokens  
            list(range(1000, 1080)),    # Image 3: 80 tokens
            list(range(5000, 5256)),    # Image 4: 256 tokens
        ]
        
        print("Adding sample tokenized images as documents:")
        for i, tokens in enumerate(sample_images):
            builder1.add_document(tokens, lengths=[len(tokens)])
            print(f"  - Image {i+1}: {len(tokens)} tokens, range [{min(tokens)}, {max(tokens)}]")
        
        # Finalize
        builder1.finalize(f"{temp_dir}/example1.idx")
        
        # Show file sizes
        bin_size = os.path.getsize(f"{temp_dir}/example1.bin")
        idx_size = os.path.getsize(f"{temp_dir}/example1.idx")
        print(f"\nOutput files:")
        print(f"  - example1.bin: {bin_size:,} bytes")
        print(f"  - example1.idx: {idx_size:,} bytes")
        
        # Example 2: Pure Vision Tokenizer
        print("\n" + "=" * 60)
        print("Example 2: Pure Vision Tokenizer")
        print("=" * 60)
        print("Using VisionTokenIndexedDatasetBuilder without text offset.")
        
        builder2 = VisionTokenIndexedDatasetBuilder(
            output_prefix=f"{temp_dir}/example2_pure_image",
            image_vocab_size=2**17  # Only image tokens
        )
        
        print("\nAdding tokenized images:")
        for i, tokens in enumerate(sample_images):
            builder2.add_image_tokens(np.array(tokens))
            print(f"  - Added image {i+1}: {len(tokens)} tokens")
        
        builder2.finalize()
        
        # Example 3: Multimodal Tokenizer with Offset
        print("\n" + "=" * 60)
        print("Example 3: Multimodal Tokenizer (Text + Image)")
        print("=" * 60)
        print("Simulating SwissAI tokenizer + Emu3 vision tokenizer.")
        
        # Configuration
        text_vocab_size = 32000      # Typical text tokenizer size
        image_vocab_size = 2**17     # Emu3 vocabulary size
        total_vocab_size = text_vocab_size + image_vocab_size
        
        builder3 = VisionTokenIndexedDatasetBuilder(
            output_prefix=f"{temp_dir}/example3_multimodal",
            image_vocab_size=image_vocab_size,
            text_vocab_size=text_vocab_size
        )
        
        print(f"\nVocabulary configuration:")
        print(f"  - Text tokens: [0, {text_vocab_size})")
        print(f"  - Image tokens: [{text_vocab_size}, {text_vocab_size + image_vocab_size})")
        
        print(f"\nAdding images with vocabulary offset:")
        for i, original_tokens in enumerate(sample_images):
            # Simulate realistic Emu3 tokens (smaller range for demo)
            emu3_tokens = np.array(original_tokens) % (2**10)  # Keep small for demo
            final_tokens = emu3_tokens + text_vocab_size
            
            builder3.add_image_tokens(emu3_tokens)
            print(f"  - Image {i+1}: Emu3 tokens [{min(emu3_tokens)}, {max(emu3_tokens)}] → "
                  f"Final tokens [{min(final_tokens)}, {max(final_tokens)}]")
        
        builder3.finalize()
        
        # Example 4: Error Handling - Vocabulary Overflow
        print("\n" + "=" * 60)
        print("Example 4: Error Handling - Vocabulary Overflow")
        print("=" * 60)
        
        builder4 = VisionTokenIndexedDatasetBuilder(
            output_prefix=f"{temp_dir}/example4_overflow",
            image_vocab_size=500,   # 500 vision tokens
            text_vocab_size=500      # 500 text tokens
        )
        
        print("Attempting to add tokens that exceed vocabulary size...")
        try:
            # This should fail: tokens 600-699 + offset 500 = 1100-1199, but max is 999
            overflow_tokens = np.array(range(600, 700))
            builder4.add_image_tokens(overflow_tokens)
            print("❌ Error detection failed!")
        except ValueError as e:
            print(f"✅ Correctly caught overflow error: {e}")
        
        # Example 5: Reading a Dataset
        print("\n" + "=" * 60)
        print("Example 5: Reading an IndexedDataset")
        print("=" * 60)
        print("This would typically be done in your training code.")
        
        print("# Example code for reading the dataset:")
        print("from megatron.data.indexed_dataset import IndexedDataset")
        print(f"dataset = IndexedDataset('{temp_dir}/example1')")
        print("sequence = dataset[0]  # Get first sequence")
        print("print(f'First sequence has {len(sequence)} tokens')")
        
        # Summary
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print("✅ All examples completed successfully!")
        print(f"📁 Example files created in: {temp_dir}")
        print("\nTo use in your pipeline:")
        print("1. For pure vision: VisionTokenIndexedDatasetBuilder(prefix, image_vocab_size=131072)")  
        print("2. For multimodal: VisionTokenIndexedDatasetBuilder(prefix, image_vocab_size=Y, text_vocab_size=X)")
        print("3. Add images: builder.add_image_tokens(token_array)")
        print("4. Finalize: builder.finalize()")
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print(f"\n🧹 Cleaned up temporary directory: {temp_dir}")