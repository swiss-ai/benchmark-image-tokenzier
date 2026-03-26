"""Document-level rank assignment for the pooled tokenization pipeline.

Assigns logical documents (multi-image groups or interleave documents) to
ranks by estimated token cost.  Uses weighted contiguous splitting so that
each rank owns a contiguous range of documents — this preserves shard
locality for I/O while balancing compute across ranks.

Does NOT decide resize targets — that happens rank-locally in the encode pool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pyarrow.parquet as pq

from vision_tokenization.indexing.manifest import load_group_arrays
from vision_tokenization.utils.image_geometry import (
    estimate_image_tokens_batch,
    smart_resize_dims_batch,
)

logger = logging.getLogger(__name__)


@dataclass
class DocumentAssignment:
    """One logical document assigned to a rank."""

    document_id: int  # group_id from manifest
    manifest_rows: np.ndarray  # int64 indices into manifest for this doc's images
    estimated_tokens: int  # total estimated tokens (all images)


@dataclass
class DocumentOwnerPlan:
    """Assignment of documents to ranks."""

    documents: List[DocumentAssignment] = field(default_factory=list)
    total_filtered: int = 0

    @property
    def total_documents(self) -> int:
        return len(self.documents)

    def split_for_workers(self, num_workers: int) -> List[List[DocumentAssignment]]:
        """Split documents into weighted contiguous chunks for workers."""
        from vision_tokenization.utils.partitioning import weighted_contiguous_split
        costs = [float(doc.estimated_tokens) for doc in self.documents]
        return weighted_contiguous_split(self.documents, costs, num_workers)


def plan_document_ownership(
    manifest_path: Union[str, Path],
    *,
    spatial_factor: int = 16,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    resize_min_pixels: Optional[int] = None,
    resize_max_pixels: Optional[int] = None,
) -> DocumentOwnerPlan:
    """Plan document-level rank assignment from a manifest.

    Reads the manifest, groups rows by ``group_id`` (or treats each row as
    its own document if no ``group_id`` column), estimates per-document token
    cost, and returns a ``DocumentOwnerPlan`` ready for
    ``split_for_workers()``.

    Args:
        manifest_path: Path to manifest parquet.
        spatial_factor: Vision tokenizer spatial downsampling factor.
        min_pixels: Filter: drop images below this pixel count.
        max_pixels: Filter: drop images above this pixel count.
        resize_min_pixels: Tokenizer min_pixels for smart_resize.
        resize_max_pixels: Tokenizer max_pixels for smart_resize.

    Returns:
        ``DocumentOwnerPlan`` with documents sorted by manifest order.
    """
    widths, heights, group_ids, image_indices = load_group_arrays(manifest_path)
    total_rows = len(widths)

    # Pixel-count filter
    pixels = widths.astype(np.int64) * heights.astype(np.int64)
    valid_mask = np.ones(total_rows, dtype=bool)
    if min_pixels is not None:
        valid_mask &= pixels >= min_pixels
    if max_pixels is not None:
        valid_mask &= pixels <= max_pixels
    valid_mask &= (widths >= spatial_factor) & (heights >= spatial_factor)

    # Estimate per-image tokens using post-resize dims (only for valid images)
    per_image_tokens = np.zeros(total_rows, dtype=np.int64)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) > 0:
        per_image_tokens[valid_idx] = estimate_image_tokens_batch(
            heights[valid_idx], widths[valid_idx],
            spatial_factor=spatial_factor,
            min_pixels=resize_min_pixels,
            max_pixels=resize_max_pixels,
        )

    # Group by document_id (= group_id) — O(N log N) via argsort
    if total_rows == 0:
        return DocumentOwnerPlan(total_filtered=0)

    order = np.argsort(group_ids, kind="stable")
    unique_gids, group_starts, group_counts = np.unique(
        group_ids[order], return_index=True, return_counts=True,
    )

    documents: List[DocumentAssignment] = []
    total_filtered = 0

    for g, g_start, g_count in zip(unique_gids, group_starts, group_counts):
        rows = order[g_start : g_start + g_count]

        # Group-level filter: drop if ANY member fails
        if not valid_mask[rows].all():
            total_filtered += len(rows)
            continue

        # Sort rows by image_index within group
        idx_order = np.argsort(image_indices[rows])
        sorted_rows = rows[idx_order]

        est_tokens = int(per_image_tokens[sorted_rows].sum())

        documents.append(DocumentAssignment(
            document_id=int(g),
            manifest_rows=sorted_rows,
            estimated_tokens=max(1, est_tokens),
        ))

    logger.info(
        f"Document ownership plan: {len(documents):,} documents, "
        f"{total_filtered:,} filtered rows, {total_rows:,} total rows"
    )

    return DocumentOwnerPlan(
        documents=documents,
        total_filtered=total_filtered,
    )
