"""Token count estimation without GPU (dry-run mode).

Estimates total image tokens from manifest dimensions + spatial_factor.
For SFT, optionally loads text and tokenizes with a CPU-fast HF tokenizer
to estimate text tokens (~0.5ms/sample, no GPU needed).
"""

from vision_tokenization.utils.json import json_dump
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from vision_tokenization.indexing.planning.batch_planner import BatchAssignment, BatchPlan
from vision_tokenization.utils.image_geometry import estimate_image_tokens

logger = logging.getLogger(__name__)

def dry_run_batch_plan(
    batch_plan: BatchPlan,
    spatial_factor: int = 16,
) -> Dict[str, Any]:
    """Estimate total image tokens across all batches using resize dimensions.

    Args:
        batch_plan: Pre-computed BatchPlan with resize_height/resize_width per batch.
        spatial_factor: Spatial down-sampling factor of the vision tokenizer.

    Returns:
        Dict with summary statistics.
    """
    total_image_tokens = 0
    total_images = 0
    total_documents = 0
    batch_token_counts: List[int] = []

    for batch in batch_plan.batches:
        n_images = len(batch.sample_indices)
        n_docs = len(batch.group_slices) if batch.group_slices is not None else n_images
        if batch.batch_token_count is not None:
            batch_tokens = int(batch.batch_token_count)
        else:
            per_image = estimate_image_tokens(
                batch.resize_height,
                batch.resize_width,
                spatial_factor=spatial_factor,
            )
            batch_tokens = per_image * n_images
        total_image_tokens += batch_tokens
        total_images += n_images
        total_documents += n_docs
        batch_token_counts.append(batch_tokens)

    token_arr = np.array(batch_token_counts) if batch_token_counts else np.array([0])

    result = {
        "total_batches": len(batch_plan.batches),
        "total_documents": total_documents,
        "total_images": total_images,
        "total_filtered": batch_plan.total_filtered,
        "total_image_tokens": total_image_tokens,
        "avg_tokens_per_document": total_image_tokens / max(1, total_documents),
        "avg_tokens_per_batch": float(token_arr.mean()),
        "max_tokens_per_batch": int(token_arr.max()),
        "min_tokens_per_batch": int(token_arr.min()),
    }

    logger.info(
        f"Dry run: {total_documents:,} documents ({total_images:,} images), "
        f"{total_image_tokens:,} estimated image tokens "
        f"({result['avg_tokens_per_document']:.0f} avg/doc), "
        f"{len(batch_plan.batches)} batches"
    )
    return result


def export_dry_run(result: Dict[str, Any], output_dir: str) -> str:
    """Save dry-run stats to a JSON file in the output directory.

    Args:
        result: Dict returned by ``dry_run_batch_plan``.
        output_dir: Directory where the stats file will be written.

    Returns:
        Path to the written JSON file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats_path = out / "dry_run_stats.json"
    json_dump(result, stats_path)
    logger.info(f"Dry run stats saved to {stats_path}")
    return str(stats_path)
