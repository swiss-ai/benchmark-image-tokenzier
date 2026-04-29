"""Dry-run utilities for plan inspection without GPU execution."""

import logging
from pathlib import Path
from typing import Any, Dict

from vision_tokenization.utils.json import json_dump

logger = logging.getLogger(__name__)


def dry_run_batch_plan(plan: Any, spatial_factor: int) -> Dict[str, Any]:
    """Summarize the execution plan without instantiating the tokenizer.

    This stays intentionally small: the dry run should explain how much work the
    planner scheduled, not try to emulate runtime execution. The summary is used
    by tests and by dry-run CLI paths to sanity-check batching decisions.
    """
    if hasattr(plan, "execution"):
        batch_table = plan.execution.image_batches
        batch_token_counts = batch_table.batch_token_counts.tolist()
        summary = {
            "total_documents": int(plan.total_documents),
            "total_components": int(plan.total_components),
            "total_image_components": int(plan.total_image_components),
            "total_text_components": int(plan.total_text_components),
            "total_batches": int(plan.total_batches),
        }
    else:
        # Tests still use a small adapter object that exposes ``batches`` only.
        batch_token_counts = [int(batch.batch_token_count) for batch in plan.batches]
        summary = {
            "total_batches": len(plan.batches),
            "total_samples": int(getattr(plan, "total_samples", 0)),
            "total_filtered": int(getattr(plan, "total_filtered", 0)),
        }

    summary["max_tokens_per_batch"] = max(batch_token_counts, default=0)
    # Keep this in the summary so callers can confirm what token-space
    # assumptions the dry run was computed with.
    summary["spatial_factor"] = int(spatial_factor)
    return summary


def export_dry_run(result: Dict[str, Any], output_dir: str) -> str:
    """Save dry-run stats to a JSON file in the output directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats_path = out / "dry_run_stats.json"
    json_dump(result, stats_path)
    logger.info(f"Dry run stats saved to {stats_path}")
    return str(stats_path)
