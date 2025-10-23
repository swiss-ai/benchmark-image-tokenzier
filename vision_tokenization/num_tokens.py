import json
from pathlib import Path
import argparse


from typing import Tuple


def find_dataset_info_files(root: Path):
    # Recursively yield all dataset_info.json files under root
    yield from root.rglob("dataset_info.json")


def get_int(d: dict, key: str, p: Path = None) -> int:
    """Extract integer value from dictionary, with error handling."""
    v = d.get(key, 0)
    try:
        return int(v)
    except (TypeError, ValueError):
        if p:
            print(f"Invalid {key} (not an int or not found): ", p)
        return 0


def get_float(d: dict, key: str, p: Path = None) -> float:
    """Extract float value from dictionary, with error handling."""
    v = d.get(key, 0.0)
    try:
        return float(v)
    except (TypeError, ValueError):
        if p:
            print(f"Invalid {key} (not a float or not found): ", p)
        return 0.0


def read_statistics(p: Path) -> Tuple[int, int, int, int, int]:
    # Read JSON file and extract top-level "statistics" fields
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        print("Could not read: ", p, )
        return 0, 0, 0, 0, 0

    stats = data.get("statistics")
    if not isinstance(stats, dict):
        print("Invalid statistics (not found or not a dict): ", p)
        return 0, 0, 0, 0, 0

    total = get_int(stats, "total_tokens", p)
    image = get_int(stats, "image_tokens", p)
    text = get_int(stats, "text_tokens", p)
    skipped_samples = get_int(stats, "samples_skipped", p)
    skipped_resolution = get_int(stats, "resolution_skipped", p)
    total_skipped = skipped_samples + skipped_resolution
    total_processed = get_int(stats, "total_samples_processed", p)
    return total, image, text, total_skipped, total_processed


def read_merged_statistics(p: Path) -> Tuple[str, int, int, float, int, list]:
    """Read JSON file from merged dataset and extract statistics.

    Returns: (dataset_name, input_tokens, output_tokens, efficiency_percent, num_sources, sources_detail)
    Sources_detail will have 'percentage' field added to each source.
    """
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        print("Could not read: ", p)
        return "", 0, 0, 0.0, 0, []

    dataset_name = data.get("dataset_name", "unknown")

    stats = data.get("statistics")
    if not isinstance(stats, dict):
        print("Invalid statistics (not found or not a dict): ", p)
        return dataset_name, 0, 0, 0.0, 0, []

    input_tokens = get_int(stats, "input_tokens", p)
    output_tokens = get_int(stats, "output_tokens", p)
    efficiency = get_float(stats, "merge_efficiency_percent", p)

    # Get sources detail and calculate percentages
    sources_detail = stats.get("sources_detail", [])
    if not isinstance(sources_detail, list):
        sources_detail = []

    # Add percentage calculation to each source
    for source in sources_detail:
        if isinstance(source, dict):
            source_tokens = source.get("tokens", 0)
            source["percentage"] = (source_tokens / input_tokens * 100) if input_tokens > 0 else 0.0

    num_sources = len(sources_detail)

    return dataset_name, input_tokens, output_tokens, efficiency, num_sources, sources_detail


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
        description="Sum token statistics from dataset_info.json files. Given a root directory, recursively searches for all dataset_info.json files, reads their statistics, and sums them up. Can also read a single JSON file directly."
                    "Works only on json output files as part of tokenization script in this repo.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/capstor/store/cscs/swissai/infra01/vision-datasets/FineVision/tokenized_sft,/capstor/store/cscs/swissai/infra01/vision-datasets/LLaVA-OneVision-1.5-Instruct-Data/tokenized_sft",
        help="Root directory to search for dataset_info.json files, or path to a single JSON file. Can be a comma separated list as well",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Read merged dataset statistics files (different JSON structure with input_tokens, output_tokens, merge_efficiency_percent)",
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

    if args.merged:
        # Handle merged dataset statistics
        input_tokens_sum = 0
        output_tokens_sum = 0
        total_sources = 0

        print(f"\n=== MERGED DATASET INFO ===")
        print(f"{'Dataset Name':<35} {'Input Tokens':>15} {'Output Tokens':>15} {'Efficiency %':>12} {'Sources':>10}")
        print("-" * 95)
        for info_path in overall_info_paths:
            name, input_tok, output_tok, efficiency, num_sources, sources_detail = read_merged_statistics(info_path)
            input_tokens_sum += input_tok
            output_tokens_sum += output_tok
            total_sources += num_sources
            print(f"{name:<35} {human_tokens(input_tok):>15} {human_tokens(output_tok):>15} {efficiency:>11.2f}% {num_sources:>10}")

            # Print source breakdown with percentages
            if sources_detail:
                print(f"\n  Source Breakdown for '{name}':")
                print(f"  {'Source Path':<60} {'Tokens':>15} {'Percentage':>12}")
                print(f"  {'-' * 90}")
                for source in sources_detail:
                    source_path = source.get("path", "unknown")
                    source_tokens = source.get("tokens", 0)
                    percentage = source.get("percentage", 0.0)
                    # Extract just the last part of the path for cleaner display
                    source_name = Path(source_path).name if source_path else "unknown"
                    print(f"  {source_name:<60} {human_tokens(source_tokens):>15} {percentage:>11.2f}%")
                print()  # Empty line after source breakdown

        print("\n===Raw totals===")
        print(f"  input_tokens:  {input_tokens_sum}")
        print(f"  output_tokens: {output_tokens_sum}")
        print(f"  total_sources: {total_sources}")

        print("\n===Human-readable===")
        print(f"  input_tokens:  {human_tokens(input_tokens_sum)}")
        print(f"  output_tokens: {human_tokens(output_tokens_sum)}")
    else:
        # Handle regular tokenization statistics
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