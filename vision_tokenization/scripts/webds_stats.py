import glob

import webdataset as wds


def inspect_webdataset(shard_pattern):
    # Expand the shard pattern to a list of file paths
    shard_files = sorted(glob.glob(shard_pattern))
    if not shard_files:
        print(f"No files found matching pattern: {shard_pattern}")
        return

    dataset = wds.WebDataset(shard_files)

    # Efficiently count and inspect without loading all into memory
    sample_count = 0
    all_keys = set()
    first_sample = None

    print("Inspecting WebDataset...")
    print("-" * 50)

    for i, sample in enumerate(dataset):
        sample_count += 1
        all_keys.update(sample.keys())

        # Save first sample for display
        if i == 0:
            first_sample = sample
            print(f"\nFirst sample keys and types:")
            for key, value in sample.items():
                print(f"  {key}: {type(value).__name__}")

        # Progress indicator for large datasets
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1} samples...")
            print(f"All keys until now: {all_keys}")

    print(f"\n{'=' * 50}")
    print(f"Total samples: {sample_count}")
    print(f"\nAll available keys across dataset:")
    for key in sorted(all_keys):
        print(f"  - {key}")


# Example usage:
shard_pattern = "/capstor/store/cscs/swissai/infra01/vision-datasets/imageomics/TreeOfLife-10M/*.tar.gz"  # Replace with your shard pattern
inspect_webdataset(shard_pattern)
