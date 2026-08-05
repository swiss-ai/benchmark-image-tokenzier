"""
Megatron-LM Compatible IndexedDataset Implementation for Vision Tokenization
===========================================================================

This is the exact implementation from /iopsstor/scratch/cscs/xyixuan/PDM/notebooks/create_meg_files.ipynb
with minor adaptations for our tokenization pipeline.

This format is directly compatible with Megatron-LM's data loaders.

Usage
-----

``IndexedDatasetBuilder`` writes the .bin/.idx pair;
callers pass complete sequences whose ids are already in the model's id space.
Vision ids are offset at encode time from ``omnimodal_config``, not here —
the writer has no view of modalities,
and the vision band does not begin where the text vocabulary ends
(Apertus 1.5 leaves 200 reserved slots between them).

```python
builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
builder.add_item(sequence)
builder.end_document()
builder.finalize(f"{prefix}.idx")
```

Loading in Megatron-LM::

    from megatron.data.indexed_dataset import IndexedDataset
    dataset = IndexedDataset("path/to/dataset")

Key Features
------------
- **Optimal Storage**: Auto-selects uint16 for vocab < 65,500 (saves 50% space)
- **Megatron Compatible**: Uses official MMIDIDX header format

File Format Details
------------------
- Header: MMIDIDX\\x00\\x00 (Megatron memory-mapped format)
- Index file structure:
  - Header (9 bytes)
  - Version (8 bytes)
  - Dtype code (1 byte)
  - Sequence count (8 bytes)
  - Document count (8 bytes)
  - Document lengths array (int32) - length of each document in tokens
  - Document pointers array (int64) - byte offset of each document in the .bin file
  - Document indices array (int64) - for compatibility, contains [0, 1, 2, ..., #docs]
  - [Optional] Sequence modes array (int8) - for multimodal datasets
"""

import os
import struct
from enum import Enum
from typing import List, Optional, Type, Union

import numpy as np
import torch

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


def read_idx(prefix: str):
    """Read an MMIDIDX .idx: (header_bytes, seq_lengths, seq_pointers, doc_count).

    header_bytes = magic+version+dtype (18 bytes) — reusable verbatim by
    ``write_idx_view``.
    """
    import struct

    with open(prefix + ".idx", "rb") as fh:
        header = fh.read(18)
        n, n_doc = struct.unpack("<QQ", fh.read(16))
        lengths = np.frombuffer(fh.read(n * 4), dtype=np.int32)
        pointers = np.frombuffer(fh.read(n * 8), dtype=np.int64)
    return header, lengths, pointers, n_doc


def write_idx_view(path: str, header: bytes, lengths, pointers) -> None:
    """Write an .idx selecting a subset of an existing .bin (pointers reused
    verbatim — the .bin is shared; one document per sequence)."""
    import struct

    n = len(lengths)
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(struct.pack("<QQ", n, n + 1))
        fh.write(np.ascontiguousarray(lengths, dtype=np.int32).tobytes())
        fh.write(np.ascontiguousarray(pointers, dtype=np.int64).tobytes())
        fh.write(np.arange(n + 1, dtype=np.int64).tobytes())


class IndexedDatasetBuilder:
    """Builder class for the IndexedDataset class

    This is the exact implementation from the reference notebook.

    Args:
        bin_path (str): The path to the data (.bin) file
        dtype (Type[np.number], optional): The dtype of the index file. Defaults to np.int32.
        multimodal (bool, optional): Whether the dataset is multimodal. Defaults to False.
    """

    def __init__(self, bin_path: str, dtype: Type[np.number] = np.int32, multimodal: bool = False) -> None:
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
        np_array = np.array(tensor.numpy() if hasattr(tensor, "numpy") else tensor, dtype=self.dtype)
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

        with open(idx_path, "wb") as idx_writer:
            # Write header
            idx_writer.write(_INDEX_HEADER)
            # Write version
            idx_writer.write(struct.pack("<Q", 1))
            # Write dtype code
            idx_writer.write(struct.pack("<B", DType.code_from_dtype(self.dtype)))

            # Write counts
            # - sequence_count = N
            # - document_count (in file) = N+1 (length of document_indices array)
            # - actual documents = N
            sequence_count = len(self.sequence_lengths)
            idx_writer.write(struct.pack("<Q", sequence_count))

            # IMPORTANT: Write the length of document_indices array, not the number of documents
            # Megatron reads exactly this many elements from the array
            # Megatron then checks: assert sequence_count == document_indices[-1]
            document_count = len(self.document_indices)
            idx_writer.write(struct.pack("<Q", document_count))

            # Write document lengths (stored as sequence_lengths for compatibility)
            sequence_lengths = np.array(self.sequence_lengths, dtype=np.int32)
            idx_writer.write(sequence_lengths.tobytes(order="C"))

            # Write document pointers (byte offsets into .bin file)
            sequence_pointers = self._sequence_pointers(self.sequence_lengths)
            sequence_pointers = np.array(sequence_pointers, dtype=np.int64)
            idx_writer.write(sequence_pointers.tobytes(order="C"))

            # Write document indices (for compatibility, [0, 1, 2, ..., #docs])
            document_indices = np.array(self.document_indices, dtype=np.int64)
            idx_writer.write(document_indices.tobytes(order="C"))

            # Write sequence modes if multimodal
            if self.sequence_modes is not None:
                sequence_modes = np.array(self.sequence_modes, dtype=np.int8)
                idx_writer.write(sequence_modes.tobytes(order="C"))

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


