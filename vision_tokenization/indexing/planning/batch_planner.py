"""Global k-means clustering + batch assignment using faiss."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import faiss
import numpy as np
import pyarrow.parquet as pq

from vision_tokenization.indexing.manifest import load_group_arrays, load_resolution_arrays
from vision_tokenization.utils.image_geometry import (
    estimate_image_tokens,
    estimate_image_tokens_batch,
    smart_resize_dims_batch,
)

logger = logging.getLogger(__name__)


@dataclass
class BatchAssignment:
    """A single batch: indices into the manifest table + final encode dims."""

    sample_indices: np.ndarray  # int64 indices into manifest
    resize_height: int
    resize_width: int
    batch_token_count: Optional[int] = None
    group_slices: Optional[np.ndarray] = None  # shape (num_groups, 2): [start, end) into sample_indices


@dataclass
class BatchPlan:
    """Collection of all batches produced by the planner."""

    batches: List[BatchAssignment] = field(default_factory=list)
    total_samples: int = 0
    total_filtered: int = 0

    @staticmethod
    def _estimate_batch_cost(batch: BatchAssignment) -> int:
        """Estimate batch cost for worker balancing while preserving locality."""
        if batch.batch_token_count is not None:
            return int(batch.batch_token_count)
        per_image_tokens = estimate_image_tokens(
            batch.resize_height,
            batch.resize_width,
        )
        return int(per_image_tokens) * int(len(batch.sample_indices))

    def split_for_workers(self, num_workers: int) -> List[List[BatchAssignment]]:
        """Split batches into weighted contiguous chunks for *num_workers* workers.

        Contiguous assignment preserves shard/tar locality, while weighting by
        final batch size reduces long-tail stragglers on skewed image-size mixes.
        """
        if num_workers <= 0:
            raise ValueError("num_workers must be > 0")
        n = len(self.batches)
        if n == 0:
            return [[] for _ in range(num_workers)]

        costs = [self._estimate_batch_cost(batch) for batch in self.batches]
        splits: List[List[BatchAssignment]] = []
        start = 0

        while start < n and len(splits) < num_workers - 1:
            remaining_workers = num_workers - len(splits)
            remaining_batches = n - start
            if remaining_batches <= remaining_workers:
                break

            target_cost = sum(costs[start:]) / remaining_workers
            running_cost = 0.0
            end = start

            while end < n:
                next_cost = running_cost + costs[end]
                can_cut_before = end > start
                must_leave_one_per_worker = (n - (end + 1)) < (remaining_workers - 1)

                if (
                    can_cut_before
                    and not must_leave_one_per_worker
                    and abs(target_cost - running_cost) < abs(target_cost - next_cost)
                ):
                    break

                running_cost = next_cost
                end += 1

                if (n - end) == (remaining_workers - 1):
                    break

            splits.append(self.batches[start:end])
            start = end

        splits.append(self.batches[start:])
        while len(splits) < num_workers:
            splits.append([])
        return splits


def _estimate_single_image_tokens(
    heights: np.ndarray,
    widths: np.ndarray,
    spatial_factor: int,
    resize_min_pixels: Optional[int],
    resize_max_pixels: Optional[int],
) -> np.ndarray:
    """Estimate final per-image token counts after tokenizer smart resize."""
    return estimate_image_tokens_batch(
        heights,
        widths,
        spatial_factor=spatial_factor,
        min_pixels=resize_min_pixels,
        max_pixels=resize_max_pixels,
    )


def _plan_exact_resize_batches(
    valid_indices: np.ndarray,
    heights: np.ndarray,
    widths: np.ndarray,
    batch_size: int,
    max_batch_tokens: int,
    spatial_factor: int,
    resize_min_pixels: int,
    resize_max_pixels: int,
    total_samples: int,
    total_filtered: int,
    num_clusters: int = 2000,
    gpu: bool = True,
    niter: int = 10,
) -> BatchPlan:
    """Plan batches by grouping on exact post-resize dimensions.

    Since ``smart_resize_dims`` snaps every image to a discrete grid
    (multiples of *spatial_factor*, clamped to pixel limits), we can group
    images by their exact final encode size.  Every image in a batch then
    has identical dimensions — zero padding waste, and the resize fast-path
    in ``preprocess_batch`` fires for images already at that size.

    For large datasets the exact grouping produces well-filled batches.
    For small datasets where the average group is too small to fill a batch,
    falls back to k-means on the post-resize dims to merge similar-sized
    groups together.
    """
    N = len(valid_indices)

    # Vectorised smart_resize for all valid images
    final_h, final_w = smart_resize_dims_batch(
        heights[valid_indices],
        widths[valid_indices],
        min_pixels=resize_min_pixels,
        max_pixels=resize_max_pixels,
        factor=spatial_factor,
    )

    # Encode (h, w) pairs as a single int64 key for grouping
    keys = final_h.astype(np.int64) * 100_000 + final_w.astype(np.int64)
    unique_keys, inverse = np.unique(keys, return_inverse=True)

    # --- Hybrid: exact for large groups, k-means merge for sparse groups ---
    # Pre-sort by group
    sorted_order = np.argsort(inverse, kind="stable")
    group_counts = np.bincount(inverse, minlength=len(unique_keys))
    group_offsets = np.empty(len(unique_keys) + 1, dtype=np.int64)
    group_offsets[0] = 0
    np.cumsum(group_counts, out=group_offsets[1:])

    # Per-group token-based batch size threshold
    batches: List[BatchAssignment] = []
    sparse_local_indices: List[int] = []  # indices into valid_indices
    n_exact = 0

    for group_idx, key in enumerate(unique_keys):
        rh = int(key // 100_000)
        rw = int(key % 100_000)
        g_start = int(group_offsets[group_idx])
        g_end = int(group_offsets[group_idx + 1])
        members = sorted_order[g_start:g_end]

        per_image_tokens = estimate_image_tokens(rh, rw, spatial_factor=spatial_factor)
        max_by_tokens = max(1, max_batch_tokens // per_image_tokens)
        chunk_size = min(batch_size, max_by_tokens)

        if len(members) >= chunk_size:
            # Large group: pack as exact-resize batches
            members = members[np.argsort(valid_indices[members])]
            for start in range(0, len(members), chunk_size):
                chunk = members[start : start + chunk_size]
                global_idx = valid_indices[chunk]
                batches.append(
                    BatchAssignment(
                        sample_indices=global_idx,
                        resize_height=rh,
                        resize_width=rw,
                        batch_token_count=per_image_tokens * len(chunk),
                    )
                )
            n_exact += len(members)
        else:
            # Sparse group: collect for k-means merge
            sparse_local_indices.extend(members.tolist())

    n_sparse = len(sparse_local_indices)
    logger.info(
        f"Hybrid planning: {n_exact:,} exact + {n_sparse:,} sparse "
        f"({len(batches):,} exact batches so far)"
    )

    # Merge sparse leftovers via k-means on their post-resize dims
    if n_sparse > 0:
        sparse_idx = np.array(sparse_local_indices, dtype=np.int64)
        sparse_batches = _plan_kmeans_on_resize_dims(
            valid_indices[sparse_idx],
            final_h[sparse_idx],
            final_w[sparse_idx],
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
            spatial_factor=spatial_factor,
            resize_min_pixels=resize_min_pixels,
            resize_max_pixels=resize_max_pixels,
            num_clusters=min(num_clusters, max(1, n_sparse // batch_size)),
            gpu=gpu,
            niter=niter,
            total_samples=0,
            total_filtered=0,
        )
        batches.extend(sparse_batches.batches)
        logger.info(
            f"Sparse k-means: {len(sparse_batches.batches):,} batches "
            f"from {n_sparse:,} images"
        )

    # Sort all batches by median sample index for shard locality
    batches.sort(key=lambda b: int(np.median(b.sample_indices)))

    logger.info(
        f"Hybrid plan: {len(batches):,} total batches, "
        f"{total_samples:,} total, {total_filtered:,} filtered"
    )
    return BatchPlan(
        batches=batches,
        total_samples=total_samples,
        total_filtered=total_filtered,
    )


def _plan_kmeans_on_resize_dims(
    valid_indices: np.ndarray,
    final_h: np.ndarray,
    final_w: np.ndarray,
    *,
    batch_size: int,
    max_batch_tokens: int,
    spatial_factor: int,
    resize_min_pixels: int,
    resize_max_pixels: int,
    num_clusters: int,
    gpu: bool,
    niter: int,
    total_samples: int,
    total_filtered: int,
) -> BatchPlan:
    """K-means on post-resize dims for small datasets with sparse groups.

    Groups images with similar (but not identical) post-resize dims so
    batches are well-filled.  The per-batch resize target is the average of
    the cluster members' post-resize dims, snapped to the grid.
    """
    fh = final_h.astype(np.float32)
    fw = final_w.astype(np.float32)
    aspect = fw / fh
    log_area = np.log(fh * fw)
    features = np.stack([aspect, log_area], axis=1)

    fmin = features.min(axis=0)
    fmax = features.max(axis=0)
    frange = fmax - fmin
    frange[frange == 0] = 1.0
    features = (features - fmin) / frange
    features = np.ascontiguousarray(features, dtype=np.float32)

    N = len(features)
    mean_tok = max(1.0, float(np.mean(
        (final_h // spatial_factor) * (final_w // spatial_factor)
    )))
    avg_batch = min(max(1, int(max_batch_tokens / mean_tok)), batch_size)
    k = min(num_clusters, max(1, N // avg_batch))

    logger.info(f"K-means on post-resize dims: {N:,} samples, k={k}")

    kmeans = faiss.Kmeans(d=2, k=k, niter=niter, verbose=False, gpu=gpu)
    kmeans.train(features)
    _, labels = kmeans.index.search(features, 1)
    labels = labels.ravel()

    batches: List[BatchAssignment] = []
    for cluster_id in range(k):
        members = np.where(labels == cluster_id)[0]
        if len(members) == 0:
            continue

        # Sort by log_area within cluster for greedy packing
        order = np.argsort(log_area[members])
        members = members[order]

        # Compute batch resize target: avg of post-resize dims, snapped to grid
        cluster_h = final_h[members]
        cluster_w = final_w[members]
        avg_h = int(round(cluster_h.mean() / spatial_factor)) * spatial_factor
        avg_w = int(round(cluster_w.mean() / spatial_factor)) * spatial_factor
        avg_h = max(avg_h, spatial_factor)
        avg_w = max(avg_w, spatial_factor)

        per_image_tokens = estimate_image_tokens(avg_h, avg_w, spatial_factor=spatial_factor)
        max_by_tokens = max(1, max_batch_tokens // per_image_tokens)
        chunk_size = min(batch_size, max_by_tokens)

        for start in range(0, len(members), chunk_size):
            chunk = members[start : start + chunk_size]
            global_idx = valid_indices[chunk]
            batches.append(
                BatchAssignment(
                    sample_indices=global_idx,
                    resize_height=avg_h,
                    resize_width=avg_w,
                    batch_token_count=per_image_tokens * len(chunk),
                )
            )

    logger.info(
        f"K-means resize plan: {len(batches):,} batches, "
        f"{total_samples:,} total, {total_filtered:,} filtered"
    )
    return BatchPlan(
        batches=batches,
        total_samples=total_samples,
        total_filtered=total_filtered,
    )


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
    resize_min_pixels: Optional[int] = None,
    resize_max_pixels: Optional[int] = None,
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
            resize_min_pixels=resize_min_pixels,
            resize_max_pixels=resize_max_pixels,
        )

    # --- Single-image path (original) ---
    # --- pixel-count filter ------------------------------------------------
    pixels = widths.astype(np.int64) * heights.astype(np.int64)
    mask = np.ones(total_samples, dtype=bool)
    if min_pixels is not None:
        mask &= pixels >= min_pixels
    if max_pixels is not None:
        mask &= pixels <= max_pixels
    # Both dimensions must be >= spatial_factor for smart_resize_dims
    mask &= (widths >= spatial_factor) & (heights >= spatial_factor)

    valid_indices = np.where(mask)[0]
    total_filtered = total_samples - len(valid_indices)

    if len(valid_indices) == 0:
        logger.warning("All samples filtered out — returning empty plan.")
        return BatchPlan(total_samples=total_samples, total_filtered=total_filtered)

    # --- Exact-resize fast path -------------------------------------------
    # When tokenizer pixel limits are known, group by exact post-resize dims
    # instead of approximate k-means.  Every batch member encodes at the
    # same size → zero padding waste, O(N) planning.
    if resize_min_pixels is None or resize_max_pixels is None:
        raise ValueError(
            "resize_min_pixels and resize_max_pixels are required. "
            "Set tokenizer.min_pixels and tokenizer.max_pixels in config."
        )

    return _plan_exact_resize_batches(
        valid_indices, heights, widths,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        spatial_factor=spatial_factor,
        resize_min_pixels=resize_min_pixels,
        resize_max_pixels=resize_max_pixels,
        total_samples=total_samples,
        total_filtered=total_filtered,
        num_clusters=num_clusters,
        gpu=gpu,
        niter=niter,
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
    resize_min_pixels: Optional[int],
    resize_max_pixels: Optional[int],
) -> BatchPlan:
    """Group-aware batch planning: groups are atomic (all-in or all-out).

    1. Build per-group metadata: member rows, representative dims, total tokens.
    2. Filter at the group level (drop if ANY member fails pixel filter).
    3. K-means on group-level post-resize features, then greedy group-atomic packing.
    """
    total_rows = len(widths)
    num_groups = int(group_ids.max()) + 1

    # --- Compute per-image post-resize dims upfront -----------------------
    # For images that fail the spatial_factor check, dims stay 0.
    valid_for_resize = (widths >= spatial_factor) & (heights >= spatial_factor)
    all_resize_h = np.zeros(total_rows, dtype=np.int32)
    all_resize_w = np.zeros(total_rows, dtype=np.int32)
    if resize_min_pixels is not None and resize_max_pixels is not None and valid_for_resize.any():
        vfr = np.where(valid_for_resize)[0]
        all_resize_h[vfr], all_resize_w[vfr] = smart_resize_dims_batch(
            heights[vfr], widths[vfr],
            min_pixels=resize_min_pixels, max_pixels=resize_max_pixels,
            factor=spatial_factor,
        )

    # --- Build per-group metadata ------------------------------------------
    # member_rows[g] = list of manifest row indices for group g
    member_rows: List[List[int]] = [[] for _ in range(num_groups)]
    for row_idx in range(total_rows):
        member_rows[int(group_ids[row_idx])].append(row_idx)

    # Sort each group's rows by image_index to guarantee correct ordering,
    # regardless of manifest row order.
    for g in range(num_groups):
        member_rows[g].sort(key=lambda r: image_indices[r])

    # Per-group: representative (max post-resize) dims, total tokens, pixel check
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

        # Use post-resize dims for group representative
        group_rep_w[g] = int(all_resize_w[row_arr].max())
        group_rep_h[g] = int(all_resize_h[row_arr].max())
        group_total_tokens[g] = int(
            _estimate_single_image_tokens(
                gh,
                gw,
                spatial_factor,
                resize_min_pixels,
                resize_max_pixels,
            ).sum()
        )

        # Group-level pixel filter: drop if ANY member fails
        if min_pixels is not None and (gp < min_pixels).any():
            group_valid[g] = False
        if max_pixels is not None and (gp > max_pixels).any():
            group_valid[g] = False
        if (gw < spatial_factor).any() or (gh < spatial_factor).any():
            group_valid[g] = False

    valid_groups = np.where(group_valid)[0]
    total_filtered_rows = total_rows - sum(len(member_rows[g]) for g in valid_groups)

    if len(valid_groups) == 0:
        logger.warning("All groups filtered out — returning empty plan.")
        return BatchPlan(total_samples=total_rows, total_filtered=total_filtered_rows)

    # --- Feature matrix on groups (post-resize dims) ----------------------
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
                    all_resize_h, all_resize_w,
                    spatial_factor,
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
    resize_heights: np.ndarray,
    resize_widths: np.ndarray,
    spatial_factor: int,
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

    # Batch resize target: avg of post-resize dims, snapped to grid
    batch_rh = resize_heights[sample_indices]
    batch_rw = resize_widths[sample_indices]
    rh = int(round(batch_rh.mean() / spatial_factor)) * spatial_factor
    rw = int(round(batch_rw.mean() / spatial_factor)) * spatial_factor
    rh = max(rh, spatial_factor)
    rw = max(rw, spatial_factor)
    per_image_tokens = estimate_image_tokens(rh, rw, spatial_factor=spatial_factor)
    batch_tokens = per_image_tokens * len(sample_indices)
    return BatchAssignment(
        sample_indices=sample_indices,
        resize_height=rh,
        resize_width=rw,
        batch_token_count=batch_tokens,
        group_slices=group_slices,
    )


# ---------------------------------------------------------------------------
# Locality-aware batch planning
# ---------------------------------------------------------------------------


def plan_locality_batches(
    manifest_path: Union[str, Path],
    batch_size: int,
    max_batch_tokens: int,
    spatial_factor: int = 16,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    resize_min_pixels: Optional[int] = None,
    resize_max_pixels: Optional[int] = None,
    window_size: int = 5000,
) -> BatchPlan:
    """Plan batches with locality awareness.

    Instead of grouping globally by resolution (which scatters row-group
    access), processes the manifest in **windows** of ``window_size``
    consecutive rows.  Within each window, groups by exact post-resize
    (h, w) and emits full batches; leftovers are area-sorted and
    greedy-packed.  Every batch's sample_indices stay within one window.
    """
    if resize_min_pixels is None or resize_max_pixels is None:
        raise ValueError(
            "resize_min_pixels and resize_max_pixels are required."
        )

    widths, heights = load_resolution_arrays(manifest_path)
    total_samples = len(widths)

    pixels = widths.astype(np.int64) * heights.astype(np.int64)
    mask = np.ones(total_samples, dtype=bool)
    if min_pixels is not None:
        mask &= pixels >= min_pixels
    if max_pixels is not None:
        mask &= pixels <= max_pixels
    mask &= (widths >= spatial_factor) & (heights >= spatial_factor)

    valid_indices = np.where(mask)[0]
    total_filtered = total_samples - len(valid_indices)

    if len(valid_indices) == 0:
        logger.warning("All samples filtered out — returning empty plan.")
        return BatchPlan(total_samples=total_samples, total_filtered=total_filtered)

    final_h, final_w = smart_resize_dims_batch(
        heights[valid_indices], widths[valid_indices],
        min_pixels=resize_min_pixels, max_pixels=resize_max_pixels,
        factor=spatial_factor,
    )
    keys = final_h.astype(np.int64) * 100_000 + final_w.astype(np.int64)

    # Pre-compute token budget per resolution key (avoids redundant calls)
    unique_keys = np.unique(keys)
    key_info = {}
    for k in unique_keys:
        k = int(k)
        rh = k // 100_000
        rw = k % 100_000
        per_tok = estimate_image_tokens(rh, rw, spatial_factor=spatial_factor)
        key_info[k] = (rh, rw, per_tok, min(batch_size, max(1, max_batch_tokens // per_tok)))

    # valid_indices is sorted (from np.where), so window_ids is non-decreasing.
    window_ids = valid_indices // window_size
    N = len(valid_indices)

    if N > 1:
        changes = np.where(np.diff(window_ids) > 0)[0] + 1
        win_starts = np.concatenate([[0], changes])
        win_ends = np.concatenate([changes, [N]])
    else:
        win_starts = np.array([0] if N else [], dtype=np.int64)
        win_ends = np.array([N] if N else [], dtype=np.int64)

    num_windows = len(win_starts)
    logger.info(
        f"Locality planning: {N:,} valid images, "
        f"window_size={window_size}, {num_windows:,} windows"
    )

    window_ranges = [
        (int(win_starts[wi]), int(win_ends[wi]))
        for wi in range(num_windows)
    ]
    all_batches = _process_window_group(
        window_ranges, keys, valid_indices, final_h, final_w,
        key_info, batch_size, max_batch_tokens, spatial_factor,
    )

    logger.info(
        f"Locality plan: {len(all_batches):,} batches, "
        f"{total_samples:,} samples, {total_filtered:,} filtered"
    )
    return BatchPlan(
        batches=all_batches,
        total_samples=total_samples,
        total_filtered=total_filtered,
    )


def _process_window_group(
    window_ranges: List[tuple],
    keys: np.ndarray,
    valid_indices: np.ndarray,
    final_h: np.ndarray,
    final_w: np.ndarray,
    key_info: dict,
    batch_size: int,
    max_batch_tokens: int,
    spatial_factor: int,
) -> List[BatchAssignment]:
    """Process a group of windows. Designed to run in a worker process."""
    batches: List[BatchAssignment] = []

    for ws, we in window_ranges:
        win_keys = keys[ws:we]
        wn = we - ws

        local_order = np.argsort(win_keys, kind="stable")
        sorted_keys = win_keys[local_order]
        sorted_global = local_order + ws

        if wn > 1:
            run_breaks = np.where(np.diff(sorted_keys) != 0)[0] + 1
            run_starts = np.concatenate([[0], run_breaks])
            run_ends = np.concatenate([run_breaks, [wn]])
        else:
            run_starts = np.array([0])
            run_ends = np.array([wn])

        spillover_parts: List[np.ndarray] = []

        for ri_idx in range(len(run_starts)):
            rs = int(run_starts[ri_idx])
            re = int(run_ends[ri_idx])
            run = sorted_global[rs:re]
            run_len = re - rs

            k = int(sorted_keys[rs])
            rh, rw, per_tok, chunk_size = key_info[k]

            n_full = (run_len // chunk_size) * chunk_size
            for start in range(0, n_full, chunk_size):
                chunk = run[start : start + chunk_size]
                batches.append(BatchAssignment(
                    sample_indices=valid_indices[chunk],
                    resize_height=rh, resize_width=rw,
                    batch_token_count=per_tok * len(chunk),
                ))

            if n_full < run_len:
                spillover_parts.append(run[n_full:])

        if spillover_parts:
            spillover = np.concatenate(spillover_parts)
            batches.extend(_pack_spillover_local(
                spillover, valid_indices, final_h, final_w,
                batch_size=batch_size, max_batch_tokens=max_batch_tokens,
                spatial_factor=spatial_factor,
            ))

    return batches


def _pack_spillover_local(
    arr: np.ndarray,
    valid_indices: np.ndarray,
    final_h: np.ndarray,
    final_w: np.ndarray,
    *,
    batch_size: int,
    max_batch_tokens: int,
    spatial_factor: int,
) -> List[BatchAssignment]:
    """K-means on aspect ratio + log area, then greedy-pack within clusters.

    Same approach as ``_plan_kmeans_on_resize_dims`` but for small per-window
    spillover sets.  Groups images with similar shape so the averaged batch
    resize target stays close to each member's natural dimensions.
    """
    if len(arr) == 0:
        return []

    sh = final_h[arr]
    sw = final_w[arr]

    # Features: [aspect_ratio, log_area] normalised to [0, 1]
    aspect = sw.astype(np.float32) / sh.astype(np.float32)
    log_area = np.log(sh.astype(np.float32) * sw.astype(np.float32))
    features = np.stack([aspect, log_area], axis=1)

    fmin = features.min(axis=0)
    fmax = features.max(axis=0)
    frange = fmax - fmin
    frange[frange == 0] = 1.0
    features = (features - fmin) / frange
    features = np.ascontiguousarray(features, dtype=np.float32)

    N = len(arr)
    mean_tok = max(1.0, float(np.mean(
        (sh // spatial_factor) * (sw // spatial_factor)
    )))
    avg_batch = min(max(1, int(max_batch_tokens / mean_tok)), batch_size)
    # Cap k so faiss has >= 39 points per centroid (avoids warnings)
    k = max(1, min(N // max(1, avg_batch), N // 39))

    if k <= 1:
        # Too few images for clustering — just pack as one batch
        avg_h = int(round(float(sh.mean()) / spatial_factor)) * spatial_factor
        avg_w = int(round(float(sw.mean()) / spatial_factor)) * spatial_factor
        avg_h = max(avg_h, spatial_factor)
        avg_w = max(avg_w, spatial_factor)
        per_tok = estimate_image_tokens(avg_h, avg_w, spatial_factor=spatial_factor)
        return [BatchAssignment(
            sample_indices=valid_indices[arr],
            resize_height=avg_h, resize_width=avg_w,
            batch_token_count=per_tok * len(arr),
        )]

    # Force single-thread for tiny per-window spillover — spawning 288
    # threads for ~4K points in 2D is dominated by thread management overhead.
    prev_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        kmeans = faiss.Kmeans(d=2, k=k, niter=5, verbose=False, gpu=False)
        kmeans.train(features)
    finally:
        faiss.omp_set_num_threads(prev_threads)
    _, labels = kmeans.index.search(features, 1)
    labels = labels.ravel()

    batches: List[BatchAssignment] = []
    for cluster_id in range(k):
        members = np.where(labels == cluster_id)[0]
        if len(members) == 0:
            continue

        c_h = sh[members]
        c_w = sw[members]
        avg_h = int(round(float(c_h.mean()) / spatial_factor)) * spatial_factor
        avg_w = int(round(float(c_w.mean()) / spatial_factor)) * spatial_factor
        avg_h = max(avg_h, spatial_factor)
        avg_w = max(avg_w, spatial_factor)

        per_tok = estimate_image_tokens(avg_h, avg_w, spatial_factor=spatial_factor)
        chunk_size = min(batch_size, max(1, max_batch_tokens // per_tok))

        # Sort by area within cluster for stable packing
        order = np.argsort(c_h * c_w)
        members = members[order]

        for start in range(0, len(members), chunk_size):
            chunk = members[start : start + chunk_size]
            batches.append(BatchAssignment(
                sample_indices=valid_indices[arr[chunk]],
                resize_height=avg_h, resize_width=avg_w,
                batch_token_count=per_tok * len(chunk),
            ))

    return batches
