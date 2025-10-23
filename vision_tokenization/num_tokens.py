import json
from pathlib import Path
import argparse


from typing import Tuple


def find_dataset_info_files(root: Path):
    # Recursively yield all dataset_info.json files under root
    yield from root.rglob("dataset_info.json")


def read_statistics(p: Path) -> Tuple[int, int, int, int, int]:
    # Read JSON file and extract top-level "statistics" fields
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        print("Could not read: ", p, )
        return 0, 0, 0

    stats = data.get("statistics")
    if not isinstance(stats, dict):
        print("Invalid statistics (not found or not a dict): ", p)
        return 0, 0, 0

    def get_int(d: dict, key: str) -> int:
        v = d.get(key, 0)
        try:
            return int(v)
        except (TypeError, ValueError):
            print(f"Invalid {key} (not an int or not found): ", p)
            return 0

    total = get_int(stats, "total_tokens")
    image = get_int(stats, "image_tokens")
    text = get_int(stats, "text_tokens")
    skipped_samples = get_int(stats, "samples_skipped")
    skipped_resolution = get_int(stats, "resolution_skipped")
    total_skipped = skipped_samples + skipped_resolution
    total_processed = get_int(stats, "total_samples_processed")
    return total, image, text, total_skipped, total_processed


def human_tokens(n: int) -> str:
    # Format numbers in human-readable "X B tokens" style (B=billion)
    # Uses base-10 (1B = 1_000_000_000)
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f} B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n / 1_000:.2f} K"
    return f"{n}"


def main():
    parser = argparse.ArgumentParser(
        description="Sum token statistics from dataset_info.json files. Given a root directory, recursively searches for all dataset_info.json files, reads their statistics, and sums them up. Can also read a single JSON file directly.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/capstor/store/cscs/swissai/infra01/vision-datasets/FineVision/tokenized_sft,/capstor/store/cscs/swissai/infra01/vision-datasets/LLaVA-OneVision-1.5-Instruct-Data/tokenized_sft",
        help="Root directory to search for dataset_info.json files, or path to a single JSON file. Can be a comma separated list as well",
    )

    args = parser.parse_args()
    paths = [p.strip() for p in args.path.split(",") if p.strip()]
    overall_info_paths = []
    for root in paths:
        root = Path(root).expanduser().resolve()
        if root.is_file():
            # If it's a file, add it directly
            if root.suffix == ".json":
                print(f"Reading single JSON file: {root}")
                overall_info_paths.append(root)
            else:
                print(f"Warning: {root} is not a JSON file, skipping")
        elif root.is_dir():
            # If it's a directory, recursively search for dataset_info.json files
            print(f"Scanning directory: {root}")
            overall_info_paths += list(find_dataset_info_files(root))
        else:
            print(f"Invalid path (not a file or directory): {root}")
            continue

    total_sum = 0
    image_sum = 0
    text_sum = 0
    skipped_sum = 0
    processed_sum = 0

    print(f"\n=== SUBSET info ===")
    print(f"{'Dataset':<35} {'Total Tokens':>15} {'Image Tokens':>15} {'Text Tokens':>15} {'Skipped':>10} {'Processed':>10}")
    print("-" * 110)
    for info_path in overall_info_paths:
        t, i, x, s, p = read_statistics(info_path)
        total_sum += t
        image_sum += i
        text_sum += x
        skipped_sum += s
        processed_sum += p
        print(f"{info_path.parent.name:<35} {human_tokens(t):>15} {human_tokens(i):>15} {human_tokens(x):>15} {s:>10,} {p:>10,}")

    print("\n===Raw totals===")
    print(f"  total_tokens: {total_sum}")
    print(f"  image_tokens: {image_sum}")
    print(f"  text_tokens:  {text_sum}")
    print(f"  skipped_samples: {skipped_sum}")
    print(f"  processed_samples: {processed_sum}")

    print("\n===Human-readable===")
    print(f"  total_tokens: {human_tokens(total_sum)}")
    print(f"  image_tokens: {human_tokens(image_sum)}")
    print(f"  text_tokens:  {human_tokens(text_sum)}")


if __name__ == "__main__":
    main()