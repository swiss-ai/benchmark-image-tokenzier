"""Merge per-rank tokenized shards into a single dataset.

Follows the same "last rank out" pattern as ``stats_reducer``: each rank
calls ``maybe_merge_shards`` after finishing tokenization. The call checks
whether all expected ranks have completed (checkpoint files exist). The
first rank to observe a complete set performs the merge; others return
immediately.

Can also be run standalone::

    python -m vision_tokenization.pipeline.merge \
        /path/to/output_dir --expected-ranks 80
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _find_shard_pairs(
    directory: Path,
    *,
    subdirs: Optional[List[str]] = None,
) -> list[tuple[str, str]]:
    """Find all matching .bin/.idx pairs in *directory* (and optional subdirs)."""
    search_dirs = [directory]
    if subdirs:
        search_dirs.extend(directory / s for s in subdirs if (directory / s).is_dir())

    pairs = []
    seen = set()
    for search_dir in search_dirs:
        for f in sorted(search_dir.glob("rank_*_chunk_*.bin")):
            idx = f.with_suffix(".idx")
            if idx.exists() and f.stem not in seen:
                seen.add(f.stem)
                pairs.append((str(f), str(idx)))
    return pairs


def _all_ranks_done(output_dir: Path, expected_ranks: int) -> bool:
    """Check if all ranks have written their checkpoint files."""
    for rank in range(expected_ranks):
        ckpt = output_dir / f"rank_{rank:04d}_checkpoint.pt"
        if not ckpt.exists():
            return False
    return True


def merge_shards(
    output_dir: Path,
    output_name: str = "merged",
    *,
    shuffle: bool = False,
    seed: int = 42,
) -> Optional[Path]:
    """Merge all rank shard pairs in *output_dir* into a single dataset.

    Returns the output prefix path, or None if no shards were found.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    try:
        from megatron.core.datasets.indexed_dataset import (
            IndexedDataset,
            IndexedDatasetBuilder,
            get_bin_path,
            get_idx_path,
        )
    except ImportError:
        # Try common Megatron paths
        for candidate in [
            "/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM",
            os.environ.get("MEGATRON_PATH", ""),
        ]:
            if candidate and Path(candidate).is_dir():
                sys.path.insert(0, candidate)
        from megatron.core.datasets.indexed_dataset import (
            IndexedDataset,
            IndexedDatasetBuilder,
            get_bin_path,
            get_idx_path,
        )

    output_dir = Path(output_dir)

    # Collect shard pairs from main dir and split subdirs
    pairs = _find_shard_pairs(output_dir, subdirs=["stage2", "lct"])
    if not pairs:
        logger.warning("No shard pairs found in %s", output_dir)
        return None

    # Sort for deterministic order
    import random

    prefixes = [bin_path.rsplit(".bin", 1)[0] for bin_path, _ in pairs]
    if shuffle:
        random.seed(seed)
        random.shuffle(prefixes)

    output_prefix = str(output_dir / output_name)

    logger.info("Merging %d shard pairs into %s", len(prefixes), output_prefix)

    builder = None
    for prefix in prefixes:
        if builder is None:
            dataset = IndexedDataset(prefix)
            builder = IndexedDatasetBuilder(
                get_bin_path(output_prefix), dtype=dataset.index.dtype,
            )
            del dataset
        builder.add_index(prefix)

    builder.finalize(get_idx_path(output_prefix))

    out_bin = Path(get_bin_path(output_prefix))
    out_tokens = out_bin.stat().st_size // 4
    logger.info(
        "Merge complete: %s (%d shards, %s tokens)",
        output_prefix,
        len(prefixes),
        f"{out_tokens:,}",
    )
    return Path(output_prefix)


def maybe_merge_shards(
    output_dir: Path | str,
    *,
    expected_ranks: int,
    output_name: str = "merged",
    shuffle: bool = False,
    seed: int = 42,
) -> Optional[Path]:
    """Merge shards if all ranks are done. Returns None if not ready or merge disabled."""
    output_dir = Path(output_dir)

    # Check if already merged
    merged_bin = output_dir / f"{output_name}.bin"
    if merged_bin.exists():
        logger.debug("Merged file already exists: %s", merged_bin)
        return Path(output_dir / output_name)

    if not _all_ranks_done(output_dir, expected_ranks):
        return None

    try:
        return merge_shards(
            output_dir,
            output_name=output_name,
            shuffle=shuffle,
            seed=seed,
        )
    except Exception:
        logger.warning("Failed to merge shards in %s", output_dir, exc_info=True)
        return None


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Directory containing rank shard files")
    parser.add_argument(
        "--expected-ranks", type=int, default=None,
        help="Wait for this many rank checkpoints before merging",
    )
    parser.add_argument("--output-name", default="merged", help="Output prefix name")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(args.output_dir)
    if args.expected_ranks is not None and not _all_ranks_done(output_dir, args.expected_ranks):
        print(f"Not all {args.expected_ranks} ranks have finished yet.")
        return 1

    result = merge_shards(
        output_dir,
        output_name=args.output_name,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if result is None:
        print("No shards found to merge.")
        return 1
    print(f"Merged to {result}.bin / {result}.idx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
