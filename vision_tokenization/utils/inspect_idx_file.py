#!/usr/bin/env python3
"""
Utility to inspect the contents of Megatron IndexedDataset .idx files.
This shows exactly what's stored in the index files byte by byte.
"""

import struct
from typing import Any, Dict

import numpy as np
from indexed_dataset_megatron import _INDEX_HEADER, DType


def inspect_idx_file(idx_path: str) -> Dict[str, Any]:
    """
    Inspect the contents of a .idx file and show its structure.

    Args:
        idx_path: Path to the .idx file

    Returns:
        Dictionary containing all the parsed information
    """
    print(f"Inspecting IndexedDataset file: {idx_path}")
    print("=" * 80)

    info = {}

    with open(idx_path, "rb") as f:
        # 1. Read and verify header (9 bytes)
        header = f.read(9)
        info["header"] = header
        print(f"1. Header (9 bytes): {header}")
        print(f"   Expected: {_INDEX_HEADER}")
        print(f"   Match: {'✅' if header == _INDEX_HEADER else '❌'}")

        # 2. Read version (8 bytes)
        version_bytes = f.read(8)
        version = struct.unpack("<Q", version_bytes)[0]
        info["version"] = version
        print(f"\n2. Version (8 bytes): {version}")
        print(f"   Raw bytes: {version_bytes.hex()}")

        # 3. Read dtype code (1 byte)
        dtype_code_bytes = f.read(1)
        dtype_code = struct.unpack("<B", dtype_code_bytes)[0]
        dtype = DType.dtype_from_code(dtype_code)
        info["dtype_code"] = dtype_code
        info["dtype"] = dtype
        print(f"\n3. Data Type Code (1 byte): {dtype_code}")
        print(f"   Raw bytes: {dtype_code_bytes.hex()}")
        print(f"   Data type: {dtype.__name__}")
        print(f"   Bytes per token: {dtype().itemsize}")

        # 4. Read sequence count (8 bytes)
        seq_count_bytes = f.read(8)
        sequence_count = struct.unpack("<Q", seq_count_bytes)[0]
        info["sequence_count"] = sequence_count
        print(f"\n4. Sequence Count (8 bytes): {sequence_count:,}")
        print(f"   Raw bytes: {seq_count_bytes.hex()}")

        # 5. Read document count (8 bytes)
        doc_count_bytes = f.read(8)
        document_count = struct.unpack("<Q", doc_count_bytes)[0]
        info["document_count"] = document_count
        print(f"\n5. Document Count (8 bytes): {document_count:,}")
        print(f"   Raw bytes: {doc_count_bytes.hex()}")

        # 6. Read sequence lengths array
        print(f"\n6. Sequence Lengths Array ({sequence_count * 4:,} bytes)")
        seq_lengths_bytes = f.read(sequence_count * 4)  # int32 = 4 bytes each
        sequence_lengths = np.frombuffer(seq_lengths_bytes, dtype=np.int32)
        info["sequence_lengths"] = sequence_lengths
        print(f"   Total bytes: {len(seq_lengths_bytes):,}")
        print(f"   First 10 lengths: {sequence_lengths[:10] if len(sequence_lengths) >= 10 else sequence_lengths}")
        if len(sequence_lengths) > 10:
            print(f"   Last 10 lengths: {sequence_lengths[-10:]}")
        print(f"   Min length: {np.min(sequence_lengths)}")
        print(f"   Max length: {np.max(sequence_lengths)}")
        print(f"   Average length: {np.mean(sequence_lengths):.1f}")
        print(f"   Total tokens: {np.sum(sequence_lengths):,}")

        # 7. Read sequence pointers array
        print(f"\n7. Sequence Pointers Array ({sequence_count * 8:,} bytes)")
        seq_pointers_bytes = f.read(sequence_count * 8)  # int64 = 8 bytes each
        sequence_pointers = np.frombuffer(seq_pointers_bytes, dtype=np.int64)
        info["sequence_pointers"] = sequence_pointers
        print(f"   Total bytes: {len(seq_pointers_bytes):,}")
        print(f"   First 10 pointers: {sequence_pointers[:10] if len(sequence_pointers) >= 10 else sequence_pointers}")
        if len(sequence_pointers) > 10:
            print(f"   Last 10 pointers: {sequence_pointers[-10:]}")
        print(f"   These are byte offsets into the .bin file")

        # 8. Read document indices array
        print(f"\n8. Document Indices Array ({document_count * 8:,} bytes)")
        doc_indices_bytes = f.read(document_count * 8)  # int64 = 8 bytes each
        document_indices = np.frombuffer(doc_indices_bytes, dtype=np.int64)
        info["document_indices"] = document_indices
        print(f"   Total bytes: {len(doc_indices_bytes):,}")
        print(f"   Document boundaries: {document_indices}")
        print(f"   These mark the end of each document (sequence indices)")

        # 9. Check for any remaining bytes (multimodal modes)
        remaining_pos = f.tell()
        f.seek(0, 2)  # Seek to end
        file_size = f.tell()
        remaining_bytes = file_size - remaining_pos

        if remaining_bytes > 0:
            print(f"\n9. Additional Data ({remaining_bytes} bytes)")
            f.seek(remaining_pos)
            additional_data = f.read(remaining_bytes)
            print(f"   Remaining bytes: {additional_data.hex()}")
            print(f"   Possibly sequence modes for multimodal datasets")
        else:
            print(f"\n9. No additional data (standard dataset)")

        info["file_size"] = file_size

    return info


def demonstrate_idx_structure():
    """Create a sample .idx file and show its structure."""
    print("\n" + "=" * 80)
    print("DEMONSTRATION: Creating and Inspecting a Sample .idx File")
    print("=" * 80)

    import os
    import tempfile

    from indexed_dataset_megatron import DType, IndexedDatasetBuilder

    # Create a temporary sample dataset
    temp_dir = tempfile.mkdtemp()
    sample_prefix = f"{temp_dir}/sample"

    try:
        # Create builder
        builder = IndexedDatasetBuilder(bin_path=f"{sample_prefix}.bin", dtype=np.int32)

        # Add sample data
        sample_documents = [
            [1, 2, 3, 4, 5],  # Document 1: 5 tokens
            [100, 200, 300],  # Document 2: 3 tokens
            [1000, 2000, 3000, 4000],  # Document 3: 4 tokens
            [10000, 20000, 30000, 40000, 50000, 60000],  # Document 4: 6 tokens
        ]

        print("Creating sample dataset with 4 documents:")
        for i, doc in enumerate(sample_documents):
            builder.add_document(doc, lengths=[len(doc)])
            print(f"  Document {i+1}: {doc} ({len(doc)} tokens)")

        # Finalize
        builder.finalize(f"{sample_prefix}.idx")

        print(f"\nFiles created:")
        bin_size = os.path.getsize(f"{sample_prefix}.bin")
        idx_size = os.path.getsize(f"{sample_prefix}.idx")
        print(f"  {sample_prefix}.bin: {bin_size} bytes")
        print(f"  {sample_prefix}.idx: {idx_size} bytes")

        # Now inspect the created .idx file
        print(f"\n" + "=" * 40)
        print("INSPECTION RESULTS:")
        print("=" * 40)

        info = inspect_idx_file(f"{sample_prefix}.idx")

        # Explain the structure
        print(f"\n" + "=" * 40)
        print("EXPLANATION:")
        print("=" * 40)
        print(f"The .idx file contains:")
        print(f"  📋 Metadata: Header, version, data types")
        print(f"  📊 Counts: {info['sequence_count']} sequences, {info['document_count']} documents")
        print(f"  📏 Lengths: How many tokens in each sequence")
        print(f"  📍 Pointers: Byte offsets to find each sequence in .bin file")
        print(f"  📄 Documents: Which sequences belong to which documents")
        print(f"")
        print(f"This allows:")
        print(f"  🚀 Fast random access: Jump directly to any sequence")
        print(f"  💾 Memory efficiency: Don't load entire .bin file")
        print(f"  📈 Scalability: Handle datasets larger than RAM")

    finally:
        # Cleanup
        import shutil

        shutil.rmtree(temp_dir)


def show_idx_file_layout():
    """Show the byte-level layout of .idx files."""
    print("\n" + "=" * 80)
    print("MEGATRON INDEXEDDATASET .IDX FILE FORMAT")
    print("=" * 80)

    layout = """
Byte Layout of .idx files:
┌─────────────────────────────────────────────────────────────────┐
│                           HEADER SECTION                        │
├─────────────────────────────────────────────────────────────────┤
│ Bytes 0-8:   Header Magic String: b'MMIDIDX\\x00\\x00'          │
│ Bytes 9-16:  Version (uint64): Always 1                        │
│ Byte 17:     Data Type Code (uint8): See DType enum            │
├─────────────────────────────────────────────────────────────────┤
│                           COUNT SECTION                         │
├─────────────────────────────────────────────────────────────────┤
│ Bytes 18-25: Sequence Count (uint64): Number of sequences      │
│ Bytes 26-33: Document Count (uint64): Number of documents      │
├─────────────────────────────────────────────────────────────────┤
│                          ARRAYS SECTION                         │
├─────────────────────────────────────────────────────────────────┤
│ Next N*4 bytes:    Sequence Lengths (int32 array)              │
│ Next N*8 bytes:    Sequence Pointers (int64 array)             │
│ Next D*8 bytes:    Document Indices (int64 array)              │
│ Optional bytes:    Sequence Modes (int8 array) [multimodal]    │
└─────────────────────────────────────────────────────────────────┘

Where:
- N = Number of sequences
- D = Number of documents
- Each sequence pointer is a byte offset into the .bin file
- Document indices mark which sequences end each document
- Sequence modes are only present in multimodal datasets
"""

    print(layout)

    print("Data Type Codes (DType enum):")
    print("  1 = uint8    (1 byte per token)")
    print("  2 = int8     (1 byte per token)")
    print("  3 = int16    (2 bytes per token)")
    print("  4 = int32    (4 bytes per token)  ← Most common")
    print("  5 = int64    (8 bytes per token)")
    print("  6 = float64  (8 bytes per token)")
    print("  7 = float32  (4 bytes per token)")
    print("  8 = uint16   (2 bytes per token)  ← Optimal for vocab < 65K")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Inspect a specific .idx file
        idx_file = sys.argv[1]
        try:
            inspect_idx_file(idx_file)
        except Exception as e:
            print(f"Error inspecting {idx_file}: {e}")
    else:
        # Show documentation and demo
        show_idx_file_layout()
        demonstrate_idx_structure()

        print(f"\n" + "=" * 80)
        print("USAGE")
        print("=" * 80)
        print("To inspect a specific .idx file:")
        print(f"  python {__file__} path/to/dataset.idx")
        print()
        print("To see this demo again:")
        print(f"  python {__file__}")
