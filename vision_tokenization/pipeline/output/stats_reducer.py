"""Reduce per-rank stats into a single ``stats_summary.json`` file.

Used in two places:
- opportunistically by finishing workers, once all expected ranks reported
- manually as a repair tool for old runs with stale summaries
"""

import argparse
from vision_tokenization.utils.json import json_loads, json_dumps, json_dump
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_latest_rank_stats(stats_path: Path) -> List[Dict[str, Any]]:
    """Load the latest stats entry for each rank from ``stats.jsonl``."""
    if not stats_path.exists():
        return []

    by_rank: Dict[int, Dict[str, Any]] = {}
    for line in stats_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json_loads(line)
        if "rank" not in record:
            continue
        by_rank[int(record["rank"])] = record
    return [by_rank[rank] for rank in sorted(by_rank)]


def build_aggregate(rank_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a deduplicated list of per-rank stats."""
    agg = {
        "type": "aggregate",
        "num_ranks": len(rank_stats),
        "samples_processed": sum(s.get("samples_processed", 0) for s in rank_stats),
        "tokens_generated": sum(s.get("tokens_generated", 0) for s in rank_stats),
        "image_tokens": sum(s.get("image_tokens", 0) for s in rank_stats),
        "text_tokens": sum(s.get("text_tokens", 0) for s in rank_stats),
        "stage2_tokens": sum(s.get("stage2_tokens", 0) for s in rank_stats),
        "stage2_samples": sum(s.get("stage2_samples", 0) for s in rank_stats),
        "lct_tokens": sum(s.get("lct_tokens", 0) for s in rank_stats),
        "lct_samples": sum(s.get("lct_samples", 0) for s in rank_stats),
        "errors": sum(s.get("errors", 0) for s in rank_stats),
        "samples_skipped": sum(s.get("samples_skipped", 0) for s in rank_stats),
        "cuda_oom_errors": sum(s.get("cuda_oom_errors", 0) for s in rank_stats),
        "max_elapsed_s": max((s.get("elapsed_time", 0) for s in rank_stats), default=0),
        "per_rank": rank_stats,
    }
    elapsed = agg["max_elapsed_s"]
    if elapsed > 0:
        agg["tokens_per_second"] = agg["tokens_generated"] / elapsed
        agg["image_tokens_per_second"] = agg["image_tokens"] / elapsed
        agg["samples_per_second"] = agg["samples_processed"] / elapsed
    else:
        agg["tokens_per_second"] = 0
        agg["image_tokens_per_second"] = 0
        agg["samples_per_second"] = 0
    return agg


def write_stats_summary(output_dir: Path, aggregate: Dict[str, Any]) -> Path:
    """Atomically write ``stats_summary.json``."""
    summary_path = output_dir / "stats_summary.json"
    tmp_path = output_dir / f"{summary_path.name}.tmp.{os.getpid()}"
    try:
        json_dump(aggregate, tmp_path, default=str)
        os.replace(tmp_path, summary_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return summary_path


def recompute_stats_summary(
    output_dir: Path | str,
    *,
    expected_ranks: Optional[int] = None,
    require_complete: bool = True,
) -> Optional[Dict[str, Any]]:
    """Rebuild ``stats_summary.json`` from ``stats.jsonl``.

    When ``require_complete`` is true, returns ``None`` until all expected ranks
    have reported.
    """
    output_dir = Path(output_dir)
    rank_stats = load_latest_rank_stats(output_dir / "stats.jsonl")
    if not rank_stats:
        return None
    if (
        require_complete
        and expected_ranks is not None
        and len(rank_stats) < expected_ranks
    ):
        return None

    aggregate = build_aggregate(rank_stats)
    write_stats_summary(output_dir, aggregate)
    return aggregate


def load_rank_stats_files(output_dir: Path) -> List[Dict[str, Any]]:
    """Load per-rank stats from ``rank_XXXX_stats.json`` files."""
    stats = []
    for f in sorted(output_dir.glob("rank_*_stats.json")):
        try:
            record = json_loads(f.read_text())
            if "rank" in record:
                stats.append(record)
        except Exception:
            continue
    return stats


def maybe_write_stats_summary(
    output_dir: Path | str,
    *,
    expected_ranks: int,
) -> Optional[Dict[str, Any]]:
    """Try to write a complete summary, returning ``None`` if not ready."""
    output_dir = Path(output_dir)
    try:
        # Try per-rank files first, fall back to stats.jsonl
        rank_stats = load_rank_stats_files(output_dir)
        if not rank_stats:
            rank_stats = load_latest_rank_stats(output_dir / "stats.jsonl")
        if len(rank_stats) < expected_ranks:
            return None
        aggregate = build_aggregate(rank_stats)
        write_stats_summary(output_dir, aggregate)
        return aggregate
    except Exception:
        logger.debug("Failed to reduce stats in %s", output_dir, exc_info=True)
        return None


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for manual summary recomputation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Run output directory containing stats.jsonl")
    parser.add_argument(
        "--expected-ranks",
        type=int,
        default=None,
        help="Require this many unique ranks before writing the summary",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write a summary even if fewer than --expected-ranks are present",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    rank_stats = load_latest_rank_stats(output_dir / "stats.jsonl")
    if not rank_stats:
        parser.error(f"no rank stats found in {output_dir / 'stats.jsonl'}")

    if (
        args.expected_ranks is not None
        and not args.allow_partial
        and len(rank_stats) < args.expected_ranks
    ):
        parser.error(
            f"found {len(rank_stats)} rank stats but expected {args.expected_ranks}; "
            "use --allow-partial to override"
        )

    aggregate = build_aggregate(rank_stats)
    write_stats_summary(output_dir, aggregate)
    print(json_dumps(aggregate, indent=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
