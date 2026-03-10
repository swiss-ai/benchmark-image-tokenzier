"""Global k-means clustering + batch assignment using faiss."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import faiss
import numpy as np
import pyarrow.parquet as pq

from vision_tokenization.indexing.manifest import load_group_arrays, load_resolution_arrays

logger = logging.getLogger(__name__)


@dataclass
class BatchAssignment:
    """A single batch: indices into the manifest table + target resize dims."""

    sample_indices: np.ndarray  # int64 indices into manifest
    resize_height: int
    resize_width: int
    group_slices: Optional[np.ndarray] = None  # shape (num_groups, 2): [start, end) into sample_indices


@dataclass
class BatchPlan:
    """Collection of all batches produced by the planner."""

    batches: List[BatchAssignment] = field(default_factory=list)
    total_samples: int = 0
    total_filtered: int = 0

    def split_for_workers(self, num_workers: int) -> List[List[BatchAssignment]]:
        """Split batches into contiguous chunks for *num_workers* workers.

        Contiguous assignment maximises tar file-handle cache locality.
        """
        if num_workers <= 0:
            raise ValueError("num_workers must be > 0")
        n = len(self.batches)
        if n == 0:
            return [[] for _ in range(num_workers)]
        chunk_size = max(1, (n + num_workers - 1) // num_workers)
        return [
            self.batches[i : i + chunk_size]
            for i in range(0, n, chunk_size)
        ]


def _compute_resize(
    widths: np.ndarray,
    heights: np.ndarray,
    mode: str,
) -> tuple:
    """Compute (resize_height, resize_width) for a batch."""
    if mode == "avg":
        return int(round(heights.mean())), int(round(widths.mean()))
    elif mode == "min":
        return int(heights.min()), int(widths.min())
    elif mode == "max":
        return int(heights.max()), int(widths.max())
    else:
        raise ValueError(f"Unknown resize_mode: {mode}")


def _estimate_tokens(h: np.ndarray, w: np.ndarray, spatial_factor: int) -> np.ndarray:
    """Estimate token count per sample: (h // sf) * (w // sf)."""
    return (h // spatial_factor) * (w // spatial_factor)


def _chunk_by_token_budget(
    members: np.ndarray,
    heights: np.ndarray,
    widths: np.ndarray,
    valid_indices: np.ndarray,
    max_batch_tokens: int,
    spatial_factor: int,
    resize_mode: str,
    batch_size: Optional[int] = None,
) -> List[BatchAssignment]:
    """Greedily pack sorted members into batches respecting a token budget.

    If *batch_size* is also given it acts as a hard sample cap per batch.
    """
    batches: List[BatchAssignment] = []
    start = 0
    while start < len(members):
        budget = 0
        end = start
        while end < len(members):
            if batch_size is not None and (end - start) >= batch_size:
                break
            gidx = valid_indices[members[end]]
            tok = int(heights[gidx] // spatial_factor) * int(widths[gidx] // spatial_factor)
            if budget + tok > max_batch_tokens and end > start:
                break
            budget += tok
            end += 1
        chunk = members[start:end]
        global_idx = valid_indices[chunk]
        rh, rw = _compute_resize(widths[global_idx], heights[global_idx], resize_mode)
        batches.append(BatchAssignment(sample_indices=global_idx, resize_height=rh, resize_width=rw))
        start = end
    return batches


def plan_clustered_batches(
    manifest_path: Union[str, Path],
    batch_size: int,
    max_batch_tokens: int,
    spatial_factor: int = 16,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    num_clusters: int = 2000,
    resize_mode: str = "avg",
    gpu: bool = True,
    niter: int = 10,
    multi_image: bool = False,
) -> BatchPlan:
    """Plan globally-clustered batches from a manifest file.

    Strategy:
        1. Load only width/height columns from Parquet.
        2. Filter by pixel count.
        3. Compute features ``[aspect_ratio, log_area]``, normalise to [0, 1].
        4. Coarse k-means with ``k = min(num_clusters, N // avg_batch)``.
        5. Within each cluster: sort by ``log_area``, pack into batches.
        6. Compute ``resize_size`` per batch.

    Batching: uses both constraints together:
        - ``max_batch_tokens``: token budget per batch using
          ``(h // spatial_factor) * (w // spatial_factor)`` per sample.
        - ``batch_size``: hard sample cap per batch.
        Token budget is the primary constraint; batch_size prevents
        excessive samples when images are small.

    Args:
        manifest_path: Path to a WDS or HF Parquet manifest.
        batch_size: Maximum images per batch (fixed chunking).
        max_batch_tokens: Token budget per batch (dynamic packing).
        spatial_factor: Spatial down-sampling factor for token estimation (default 16).
        min_pixels: Drop images with fewer total pixels.
        max_pixels: Drop images with more total pixels.
        num_clusters: Upper-bound on number of k-means clusters.
        resize_mode: How to pick the per-batch resize size (``avg``, ``min``, ``max``).
        gpu: If True, use GPU-accelerated faiss k-means.
        niter: Number of k-means iterations (default 10; 2D data converges fast).
        multi_image: If True, use group-aware packing where entire groups
            are kept atomic within a single batch.  Requires the manifest
            to have ``group_id`` and ``image_index`` columns.

    Returns:
        A :class:`BatchPlan`.
    """
    if batch_size is None or max_batch_tokens is None:
        raise ValueError("Both batch_size and max_batch_tokens must be set")

    # Validate multi_image against manifest schema
    has_groups = "group_id" in pq.read_schema(str(manifest_path)).names

    if multi_image and not has_groups:
        raise ValueError(
            "multi_image=True but manifest has no group_id column. "
            "Re-run the scanner with image_list_column set."
        )

    widths, heights, group_ids, image_indices = load_group_arrays(manifest_path)
    total_samples = len(widths)

    if not multi_image and has_groups:
        num_groups = int(group_ids[-1]) + 1 if total_samples > 0 else 0
        if num_groups < total_samples:
            logger.warning(
                "Manifest has group_id column but multi_image=False — "
                "treating as single-image. Set multi_image=True if this "
                "is a multi-image dataset."
            )

    if multi_image:
        num_groups = int(group_ids[-1]) + 1 if total_samples > 0 else 0
        logger.info(
            f"Multi-image planning: {total_samples:,} rows, "
            f"{num_groups:,} groups"
        )
        return _plan_grouped_batches(
            widths, heights, group_ids, image_indices,
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
            spatial_factor=spatial_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            num_clusters=num_clusters,
            resize_mode=resize_mode,
            gpu=gpu,
            niter=niter,
        )

    # --- Single-image path (original) ---
    # --- pixel-count filter ------------------------------------------------
    pixels = widths.astype(np.int64) * heights.astype(np.int64)
    mask = np.ones(total_samples, dtype=bool)
    if min_pixels is not None:
        mask &= pixels >= min_pixels
    if max_pixels is not None:
        mask &= pixels <= max_pixels

    valid_indices = np.where(mask)[0]
    total_filtered = total_samples - len(valid_indices)

    if len(valid_indices) == 0:
        logger.warning("All samples filtered out — returning empty plan.")
        return BatchPlan(total_samples=total_samples, total_filtered=total_filtered)

    w = widths[valid_indices].astype(np.float32)
    h = heights[valid_indices].astype(np.float32)

    # --- feature matrix: [aspect_ratio, log_area] --------------------------
    aspect = w / h
    log_area = np.log(w * h)
    features = np.stack([aspect, log_area], axis=1)  # (N, 2)

    # Normalise to [0, 1]
    fmin = features.min(axis=0)
    fmax = features.max(axis=0)
    frange = fmax - fmin
    frange[frange == 0] = 1.0
    features = (features - fmin) / frange
    features = np.ascontiguousarray(features, dtype=np.float32)

    N = len(features)

    # Estimate average batch size for k-means cluster count
    tok_per_sample = _estimate_tokens(
        heights[valid_indices].astype(np.float32),
        widths[valid_indices].astype(np.float32),
        spatial_factor,
    )
    mean_tok = max(1.0, float(tok_per_sample.mean()))
    avg_batch = min(max(1, int(max_batch_tokens / mean_tok)), batch_size)

    k = min(num_clusters, max(1, N // avg_batch))

    mode_desc = f"max_batch_tokens={max_batch_tokens}, batch_size={batch_size}"
    logger.info(f"Planning batches: {N:,} valid samples, k={k}, {mode_desc}")

    # --- faiss k-means -----------------------------------------------------
    kmeans = faiss.Kmeans(d=2, k=k, niter=niter, verbose=False, gpu=gpu)
    kmeans.train(features)
    _, labels = kmeans.index.search(features, 1)
    labels = labels.ravel()

    # --- within-cluster sort + chunking ------------------------------------
    batches: List[BatchAssignment] = []

    for cluster_id in range(k):
        members = np.where(labels == cluster_id)[0]
        if len(members) == 0:
            continue

        # Sort by log_area within cluster
        order = np.argsort(log_area[members])
        members = members[order]

        batches.extend(
            _chunk_by_token_budget(
                members, heights, widths, valid_indices,
                max_batch_tokens, spatial_factor, resize_mode,
                batch_size=batch_size,
            )
        )

    logger.info(
        f"Batch plan: {len(batches)} batches, "
        f"{total_samples:,} total, {total_filtered:,} filtered"
    )
    return BatchPlan(
        batches=batches,
        total_samples=total_samples,
        total_filtered=total_filtered,
    )


# ---------------------------------------------------------------------------
# Group-aware planning for multi-image manifests
# ---------------------------------------------------------------------------


def _plan_grouped_batches(
    widths: np.ndarray,
    heights: np.ndarray,
    group_ids: np.ndarray,
    image_indices: np.ndarray,
    *,
    batch_size: Optional[int],
    max_batch_tokens: Optional[int],
    spatial_factor: int,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    num_clusters: int,
    resize_mode: str,
    gpu: bool,
    niter: int,
) -> BatchPlan:
    """Group-aware batch planning: groups are atomic (all-in or all-out).

    1. Build per-group metadata: member rows, representative dims, total tokens.
    2. Filter at the group level (drop if ANY member fails pixel filter).
    3. K-means on group-level features, then greedy group-atomic packing.
    """
    total_rows = len(widths)
    num_groups = int(group_ids.max()) + 1

    # --- Build per-group metadata ------------------------------------------
    # member_rows[g] = list of manifest row indices for group g
    member_rows: List[List[int]] = [[] for _ in range(num_groups)]
    for row_idx in range(total_rows):
        member_rows[int(group_ids[row_idx])].append(row_idx)

    # Sort each group's rows by image_index to guarantee correct ordering,
    # regardless of manifest row order.
    for g in range(num_groups):
        member_rows[g].sort(key=lambda r: image_indices[r])

    # Per-group: representative (max) dims, total tokens, pixel check
    group_rep_w = np.zeros(num_groups, dtype=np.int32)
    group_rep_h = np.zeros(num_groups, dtype=np.int32)
    group_total_tokens = np.zeros(num_groups, dtype=np.int64)
    group_valid = np.ones(num_groups, dtype=bool)
    group_size = np.zeros(num_groups, dtype=np.int32)

    pixels = widths.astype(np.int64) * heights.astype(np.int64)

    for g in range(num_groups):
        rows = member_rows[g]
        if not rows:
            group_valid[g] = False
            continue
        group_size[g] = len(rows)
        row_arr = np.array(rows, dtype=np.int64)
        gw = widths[row_arr]
        gh = heights[row_arr]
        gp = pixels[row_arr]

        group_rep_w[g] = int(gw.max())
        group_rep_h[g] = int(gh.max())
        group_total_tokens[g] = int(
            ((gh // spatial_factor) * (gw // spatial_factor)).sum()
        )

        # Group-level pixel filter: drop if ANY member fails
        if min_pixels is not None and (gp < min_pixels).any():
            group_valid[g] = False
        if max_pixels is not None and (gp > max_pixels).any():
            group_valid[g] = False

    valid_groups = np.where(group_valid)[0]
    total_filtered_rows = total_rows - sum(len(member_rows[g]) for g in valid_groups)

    if len(valid_groups) == 0:
        logger.warning("All groups filtered out — returning empty plan.")
        return BatchPlan(total_samples=total_rows, total_filtered=total_filtered_rows)

    # --- Feature matrix on groups ------------------------------------------
    rw = group_rep_w[valid_groups].astype(np.float32)
    rh = group_rep_h[valid_groups].astype(np.float32)
    aspect = rw / rh
    log_area = np.log(rw * rh)
    features = np.stack([aspect, log_area], axis=1)

    fmin = features.min(axis=0)
    fmax = features.max(axis=0)
    frange = fmax - fmin
    frange[frange == 0] = 1.0
    features = (features - fmin) / frange
    features = np.ascontiguousarray(features, dtype=np.float32)

    G = len(features)

    # Estimate k
    avg_group_size = float(group_size[valid_groups].mean())
    mean_group_tok = max(1.0, float(group_total_tokens[valid_groups].mean()))
    avg_groups_by_tokens = max(1, int(max_batch_tokens / mean_group_tok))
    avg_groups_by_size = max(1, int(batch_size / avg_group_size))
    avg_groups_per_batch = min(avg_groups_by_tokens, avg_groups_by_size)

    k = min(num_clusters, max(1, G // avg_groups_per_batch))

    mode_desc = f"max_batch_tokens={max_batch_tokens}, batch_size={batch_size}"
    logger.info(
        f"Group planning: {G:,} valid groups, k={k}, {mode_desc}"
    )

    # --- K-means on groups -------------------------------------------------
    kmeans = faiss.Kmeans(d=2, k=k, niter=niter, verbose=False, gpu=gpu)
    kmeans.train(features)
    _, labels = kmeans.index.search(features, 1)
    labels = labels.ravel()

    # --- Within-cluster group-atomic packing --------------------------------
    batches: List[BatchAssignment] = []

    for cluster_id in range(k):
        cluster_members = np.where(labels == cluster_id)[0]
        if len(cluster_members) == 0:
            continue

        # Sort groups by log_area
        order = np.argsort(log_area[cluster_members])
        cluster_members = cluster_members[order]

        start = 0
        while start < len(cluster_members):
            budget = 0
            img_count = 0
            end = start
            while end < len(cluster_members):
                g = valid_groups[cluster_members[end]]
                gs = int(group_size[g])
                tok = int(group_total_tokens[g])
                if (budget + tok > max_batch_tokens or img_count + gs > batch_size) and end > start:
                    break
                budget += tok
                img_count += gs
                end += 1
            if budget > max_batch_tokens:
                logger.warning(
                    f"Group has {budget} tokens, exceeding "
                    f"max_batch_tokens={max_batch_tokens}. Batch will "
                    f"overflow (groups are atomic and never split)."
                )
            if img_count > batch_size:
                logger.warning(
                    f"Group has {img_count} images, exceeding "
                    f"batch_size={batch_size}. Batch will overflow "
                    f"(groups are atomic and never split)."
                )
            chunk_groups = cluster_members[start:end]
            batches.append(
                _build_grouped_batch(
                    chunk_groups, valid_groups, member_rows,
                    widths, heights, resize_mode,
                )
            )
            start = end

    logger.info(
        f"Group batch plan: {len(batches)} batches, "
        f"{total_rows:,} total rows, {total_filtered_rows:,} filtered"
    )
    return BatchPlan(
        batches=batches,
        total_samples=total_rows,
        total_filtered=total_filtered_rows,
    )


def _build_grouped_batch(
    chunk_group_indices: np.ndarray,
    valid_groups: np.ndarray,
    member_rows: List[List[int]],
    widths: np.ndarray,
    heights: np.ndarray,
    resize_mode: str,
) -> BatchAssignment:
    """Build a BatchAssignment from a set of groups, with group_slices."""
    all_rows: List[int] = []
    slices: List[List[int]] = []
    offset = 0
    for gi in chunk_group_indices:
        g = valid_groups[gi]
        rows = member_rows[g]
        all_rows.extend(rows)
        slices.append([offset, offset + len(rows)])
        offset += len(rows)

    sample_indices = np.array(all_rows, dtype=np.int64)
    group_slices = np.array(slices, dtype=np.int64)
    rh, rw = _compute_resize(widths[sample_indices], heights[sample_indices], resize_mode)
    return BatchAssignment(
        sample_indices=sample_indices,
        resize_height=rh,
        resize_width=rw,
        group_slices=group_slices,
    )
