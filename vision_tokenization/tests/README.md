# Megatron IndexedDataset Tests and Utilities

This folder contains test suites and utilities for working with Megatron IndexedDataset files.

## Test Organization

The test suite is organized into three distinct areas:

1. **Format Testing** (`test_indexed_dataset_format.py`) - Validates binary format compliance
2. **Integrity Testing** (`test_indexed_dataset_integrity.py`) - Ensures data correctness and recovery
3. **Integration Testing** (`test_vision_pipeline_integration.py`) - Tests end-to-end pipeline

All test files use shared utilities from `test_utils.py` to avoid code duplication.

## Recent Changes

- **File Renaming**: Test files renamed to clearly reflect their purpose
- **Redundancy Removal**: Duplicate tests removed from integrity suite
- **Shared Utilities**: All tests now use common functions from `test_utils.py`

## Files

### `test_utils.py`
**Shared utility functions** used by all test files:
- `read_index_header()` - Read IndexedDataset header information
- `read_index_file()` - Read complete index file structure
- `calculate_expected_pointers()` - Calculate expected sequence pointers
- `verify_indexed_dataset()` - Comprehensive dataset verification
- `extract_sequences()` - Extract specific sequences from datasets
- `compare_token_sequences()` - Compare two sets of token sequences
- `create_test_indexed_dataset()` - Helper for creating test datasets

### `test_indexed_dataset_format.py`
Tests the core Megatron IndexedDataset format specification:
- Binary file format (header structure, magic bytes)
- Index file structure (sequence lengths, pointers, document indices)
- Basic dataset creation and reading operations
- File size calculations and data type handling
- **Uses shared utilities from test_utils.py**

Run with:
```bash
python test_indexed_dataset_format.py
```

### `test_indexed_dataset_integrity.py`
Tests data integrity, recovery operations, and edge cases:
- Multimodal token recovery (removing offsets)
- Large dataset stress testing (100+ sequences)
- Token value preservation at boundaries (uint16 limits)
- Token comparison functionality
- **Uses shared utilities from test_utils.py**

### `test_vision_pipeline_integration.py`
End-to-end integration tests for the vision tokenization pipeline:
- WebDataset creation and loading
- Image preprocessing with Emu3 tokenizer
- Full pipeline testing (images → tokens → IndexedDataset)
- Tokenization consistency verification
- Component integration validation
- **Uses shared utilities from test_utils.py**

### `inspect_indexed_dataset.py`
Utility for inspecting any IndexedDataset files:

```bash
# Basic inspection
python inspect_indexed_dataset.py /path/to/dataset

# Extract specific sequence
python inspect_indexed_dataset.py /path/to/dataset --seq 0

# Verbose output
python inspect_indexed_dataset.py /path/to/dataset --verbose

# Compare two datasets
python inspect_indexed_dataset.py /path/to/dataset1 --compare /path/to/dataset2
```

## IndexedDataset Format

The Megatron IndexedDataset consists of two files:

### Binary File (.bin)
- Contains raw token data (int32 values)
- Tokens are concatenated without any separators
- File size = total_tokens × 4 bytes

### Index File (.idx)
Structure (arrays are NOT interleaved):

1. **Header (34 bytes)**:
   - 9 bytes: Magic string `MMIDIDX\x00\x00`
   - 8 bytes: Version (uint64, always 1)
   - 1 byte: Data type code (4=int32, 8=uint16)
   - 8 bytes: Number of sequences
   - 8 bytes: Number of documents

2. **Data Arrays**:
   - **Sequence lengths**: `num_sequences × 4 bytes` (int32 array)
   - **Sequence pointers**: `num_sequences × 8 bytes` (int64 array, byte offsets)
   - **Document indices**: `(num_documents + 1) × 8 bytes` (int64 array)

### Key Concepts

**Sequence Pointers**: Byte offsets in the .bin file where each sequence starts
```python
pointer[0] = 0
pointer[i] = pointer[i-1] + (length[i-1] × 4)
```

**Document Indices**: Mark boundaries between documents
- For vision datasets, typically each image = 1 document = 1 sequence
- Array contains [0, 1, 2, ..., num_sequences]

**Multimodal Tokenizer**: For vision+text models
- Text tokens: [0, text_vocab_size)
- Image tokens: [text_vocab_size, text_vocab_size + image_vocab_size)
- Image tokens are stored with offset applied

## Example Usage

```python
# Create a dataset
from vision_tokenization.utils.indexed_dataset_megatron import IndexedDatasetBuilder

builder = IndexedDatasetBuilder("output.bin", dtype=np.int32)
builder.add_document([1, 2, 3, 4, 5], lengths=[5])
builder.finalize("output.idx")

# Read with Megatron
from megatron.core.datasets.indexed_dataset import IndexedDataset
dataset = IndexedDataset("output")
tokens = dataset[0]  # Get first sequence
```