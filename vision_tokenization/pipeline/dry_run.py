"""Dry-run utilities: export stats without GPU."""

import logging
from pathlib import Path
from typing import Any, Dict

from vision_tokenization.utils.json import json_dump

logger = logging.getLogger(__name__)


def export_dry_run(result: Dict[str, Any], output_dir: str) -> str:
    """Save dry-run stats to a JSON file in the output directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats_path = out / "dry_run_stats.json"
    json_dump(result, stats_path)
    logger.info(f"Dry run stats saved to {stats_path}")
    return str(stats_path)
