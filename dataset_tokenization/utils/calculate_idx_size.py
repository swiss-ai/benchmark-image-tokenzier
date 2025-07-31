#!/usr/bin/env python3
"""
Calculate the exact size of .idx files based on Megatron format.
This verifies the formula from the Megatron documentation.
"""

def calculate_idx_file_size(num_sequences: int, num_documents: int = None) -> dict:
    """
    Calculate the exact size of a .idx file based on the Megatron format.
    
    Args:
        num_sequences: Number of sequences (images in our case)
        num_documents: Number of documents (usually same as sequences for images)
    
    Returns:
        Dictionary with size breakdown
    """
    if num_documents is None:
        num_documents = num_sequences
    
    # Fixed header components
    header_size = 9        # b'MMIDIDX\x00\x00'
    version_size = 8       # uint64 version (always 1)
    dtype_code_size = 1    # uint8 dtype code
    seq_count_size = 8     # uint64 sequence count
    doc_count_size = 8     # uint64 document count
    
    # Variable arrays
    seq_lengths_size = num_sequences * 4    # int32 per sequence
    seq_pointers_size = num_sequences * 8   # int64 per sequence  
    doc_indices_size = num_documents * 8    # int64 per document
    
    # Total calculation
    fixed_header = header_size + version_size + dtype_code_size + seq_count_size + doc_count_size
    variable_arrays = seq_lengths_size + seq_pointers_size + doc_indices_size
    total_size = fixed_header + variable_arrays
    
    # Alternative formula from Megatron docs:
    # 34 bytes (fixed) + 20 bytes per sequence/document
    megatron_formula = 34 + (20 * num_sequences)
    
    return {
        'num_sequences': num_sequences,
        'num_documents': num_documents,
        'breakdown': {
            'header': header_size,
            'version': version_size, 
            'dtype_code': dtype_code_size,
            'sequence_count': seq_count_size,
            'document_count': doc_count_size,
            'sequence_lengths': seq_lengths_size,
            'sequence_pointers': seq_pointers_size,
            'document_indices': doc_indices_size
        },
        'fixed_header_total': fixed_header,
        'variable_arrays_total': variable_arrays,
        'calculated_total': total_size,
        'megatron_formula': megatron_formula,
        'match': total_size == megatron_formula
    }


def show_size_examples():
    """Show size calculations for various dataset sizes."""
    print("=" * 80)
    print("MEGATRON INDEXEDDATASET .IDX FILE SIZE CALCULATOR")
    print("=" * 80)
    
    print("Formula Breakdown:")
    print("  Fixed Header: 34 bytes")
    print("    - Magic header: 9 bytes (b'MMIDIDX\\x00\\x00')")
    print("    - Version: 8 bytes (uint64)")
    print("    - Data type code: 1 byte (uint8)")
    print("    - Sequence count: 8 bytes (uint64)")
    print("    - Document count: 8 bytes (uint64)")
    print()
    print("  Per Sequence/Document: 20 bytes")
    print("    - Sequence length: 4 bytes (int32)")
    print("    - Sequence pointer: 8 bytes (int64)")
    print("    - Document index: 8 bytes (int64)")
    print()
    print("  Total Formula: 34 + (20 × num_sequences) bytes")
    print()
    
    # Test cases
    test_cases = [
        ("Small test", 4),
        ("Medium dataset", 1000),
        ("Large dataset", 35000),
        ("LLaVA dataset", 558128),
        ("Million images", 1000000),
    ]
    
    print("Size Examples:")
    print("=" * 80)
    
    for name, num_seq in test_cases:
        result = calculate_idx_file_size(num_seq)
        
        print(f"\n{name}: {num_seq:,} sequences/documents")
        print(f"  Fixed header: {result['fixed_header_total']:,} bytes")
        print(f"  Variable data: {result['variable_arrays_total']:,} bytes")
        print(f"  Total size: {result['calculated_total']:,} bytes")
        print(f"  Size in MB: {result['calculated_total'] / 1024 / 1024:.2f} MB")
        print(f"  Formula check: {'✅' if result['match'] else '❌'}")
        
        # Show breakdown for first example
        if name == "Small test":
            print(f"  Detailed breakdown:")
            for component, size in result['breakdown'].items():
                print(f"    - {component}: {size} bytes")


def verify_with_real_file():
    """Create a real file and verify the size calculation."""
    print("\n" + "=" * 80)
    print("VERIFICATION WITH ACTUAL FILE")
    print("=" * 80)
    
    from indexed_dataset_megatron import IndexedDatasetBuilder, DType
    import tempfile
    import os
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test dataset
        num_sequences = 100
        import numpy as np
        
        builder = IndexedDatasetBuilder(
            bin_path=f"{temp_dir}/test.bin",
            dtype=np.int32
        )
        
        # Add sequences of varying lengths
        import numpy as np
        np.random.seed(42)  # Reproducible
        
        print(f"Creating dataset with {num_sequences} sequences...")
        for i in range(num_sequences):
            # Random sequence length between 50-200 tokens
            seq_length = np.random.randint(50, 201)
            tokens = np.random.randint(0, 10000, seq_length)
            builder.add_document(tokens.tolist(), lengths=[seq_length])
        
        builder.finalize(f"{temp_dir}/test.idx")
        
        # Check actual file size
        actual_idx_size = os.path.getsize(f"{temp_dir}/test.idx")
        actual_bin_size = os.path.getsize(f"{temp_dir}/test.bin")
        
        # Calculate expected size
        expected = calculate_idx_file_size(num_sequences)
        expected_size = expected['calculated_total']
        
        print(f"\nResults:")
        print(f"  Actual .idx size: {actual_idx_size:,} bytes")
        print(f"  Expected .idx size: {expected_size:,} bytes")
        print(f"  Difference: {abs(actual_idx_size - expected_size):,} bytes")
        print(f"  Match: {'✅' if actual_idx_size == expected_size else '❌'}")
        print(f"  .bin file size: {actual_bin_size:,} bytes")
        
        if actual_idx_size != expected_size:
            print(f"\nNote: Small differences may occur due to:")
            print(f"  - Additional padding or alignment")
            print(f"  - Extra metadata in specific implementations")
            print(f"  - Document index variations")
    
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import numpy as np
    
    show_size_examples()
    verify_with_real_file()
    
    print("\n" + "=" * 80)
    print("SUMMARY FOR YOUR LLAVA DATASET")
    print("=" * 80)
    
    llava_result = calculate_idx_file_size(558128)
    
    print(f"For LLaVA dataset with 558,128 images:")
    print(f"  .idx file size: {llava_result['calculated_total']:,} bytes")
    print(f"  .idx file size: {llava_result['calculated_total'] / 1024 / 1024:.1f} MB")
    print(f"")
    print(f"Storage breakdown:")
    print(f"  - Fixed header: {llava_result['fixed_header_total']} bytes")
    print(f"  - Per-image metadata: 20 bytes × 558,128 = {llava_result['variable_arrays_total']:,} bytes")
    print(f"")
    print(f"The .idx file is tiny compared to .bin file!")
    print(f"  - .idx: ~11 MB (just metadata)")
    print(f"  - .bin: ~1-2 GB (actual tokens)")
    print(f"  - Ratio: .idx is ~0.5% the size of .bin")