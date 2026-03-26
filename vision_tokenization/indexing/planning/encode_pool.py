"""Rank-local image encode pool with hybrid exact/approximate batching.

Given a chunk of documents assigned to one rank, this module:
1. Flattens all image components into a pool
2. Computes exact post-resize dims for each image
3. Groups by exact (h, w) — most images land in well-populated buckets
4. Merges sparse leftovers via approximate similarity batching
5. Produces encode batches ready for GPU, with image→document mapping

The chunk drives planning, not memory residency.  Images are loaded
on demand per encode batch, not all at once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import faiss
import numpy as np

from vision_tokenization.utils.image_geometry import (
    estimate_image_tokens,
    smart_resize_dims_batch,
)

logger = logging.getLogger(__name__)


@dataclass
class EncodeBatch:
    """One GPU encode batch: images sharing the same (or similar) resize target."""

    pool_indices: np.ndarray  # indices into the chunk's image pool
    resize_height: int
    resize_width: int
    token_count: int  # estimated total tokens for this batch


@dataclass
class ImagePoolEntry:
    """Metadata for one image in the rank-local pool."""

    pool_index: int
    document_id: int
    component_index: int  # position within the document
    manifest_row: int  # global manifest index (for loading)
    resize_height: int  # exact post-resize height
    resize_width: int  # exact post-resize width


@dataclass
class ChunkEncodePlan:
    """Encode plan for one chunk of documents on one rank.

    Contains encode batches (for GPU) and the pool metadata needed to
    scatter results back to documents after encoding.
    """

    pool: List[ImagePoolEntry] = field(default_factory=list)
    encode_batches: List[EncodeBatch] = field(default_factory=list)
    total_images: int = 0
    exact_batched: int = 0  # images in exact-match batches
    approx_batched: int = 0  # images in approximate batches


def plan_chunk_encode(
    document_manifest_rows: List[np.ndarray],
    document_ids: List[int],
    component_indices: List[List[int]],
    heights: np.ndarray,
    widths: np.ndarray,
    *,
    spatial_factor: int = 16,
    resize_min_pixels: int,
    resize_max_pixels: int,
    batch_size: int = 128,
    max_batch_tokens: int = 32_768,
    min_bucket_size: int = 4,
    num_clusters: int = 256,
    gpu_kmeans: bool = False,
    max_ar_spread: float = 1.5,
    max_area_spread: float = 2.0,
) -> ChunkEncodePlan:
    """Plan encode batches for one chunk of documents.

    Args:
        document_manifest_rows: Per-document list of manifest row indices
            (only image rows, ordered by component_index).
        document_ids: Per-document IDs.
        component_indices: Per-document list of component indices for images.
        heights: Full manifest heights array (for indexing by manifest row).
        widths: Full manifest widths array.
        spatial_factor: Vision tokenizer spatial downsampling.
        resize_min_pixels: Tokenizer min_pixels for smart_resize.
        resize_max_pixels: Tokenizer max_pixels for smart_resize.
        batch_size: Max images per encode batch.
        max_batch_tokens: Token budget per encode batch.
        min_bucket_size: Exact-resize buckets smaller than this go to
            approximate batching.
        num_clusters: K-means clusters for approximate batching of leftovers.
        gpu_kmeans: Use GPU for k-means (leftovers are usually small).
        max_ar_spread: Distortion guard: max aspect ratio spread in a batch.
        max_area_spread: Distortion guard: max area ratio spread in a batch.

    Returns:
        ``ChunkEncodePlan`` with encode batches and pool metadata.
    """
    # --- Build image pool ---
    pool: List[ImagePoolEntry] = []
    all_manifest_rows = []

    for doc_idx, (rows, doc_id, comp_idxs) in enumerate(
        zip(document_manifest_rows, document_ids, component_indices)
    ):
        for row, comp_idx in zip(rows, comp_idxs):
            pool.append(ImagePoolEntry(
                pool_index=len(pool),
                document_id=doc_id,
                component_index=comp_idx,
                manifest_row=int(row),
                resize_height=0,
                resize_width=0,
            ))
            all_manifest_rows.append(int(row))

    if not pool:
        return ChunkEncodePlan()

    manifest_rows_arr = np.array(all_manifest_rows, dtype=np.int64)

    # --- Compute exact post-resize dims ---
    pool_h = heights[manifest_rows_arr]
    pool_w = widths[manifest_rows_arr]

    final_h, final_w = smart_resize_dims_batch(
        pool_h, pool_w,
        min_pixels=resize_min_pixels,
        max_pixels=resize_max_pixels,
        factor=spatial_factor,
    )

    for i, entry in enumerate(pool):
        entry.resize_height = int(final_h[i])
        entry.resize_width = int(final_w[i])

    # --- Phase 1: Exact-resize buckets ---
    keys = final_h.astype(np.int64) * 100_000 + final_w.astype(np.int64)
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique_keys))

    # Pre-sort by group for O(1) slicing
    sorted_order = np.argsort(inverse, kind="stable")
    offsets = np.empty(len(unique_keys) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    encode_batches: List[EncodeBatch] = []
    leftover_indices: List[int] = []  # pool indices for sparse buckets
    exact_count = 0

    for gidx, key in enumerate(unique_keys):
        rh = int(key // 100_000)
        rw = int(key % 100_000)
        g_start = int(offsets[gidx])
        g_end = int(offsets[gidx + 1])
        members = sorted_order[g_start:g_end]

        if len(members) < min_bucket_size:
            leftover_indices.extend(members.tolist())
            continue

        # Chunk this exact bucket into encode batches
        per_image_tokens = estimate_image_tokens(rh, rw, spatial_factor=spatial_factor)
        max_by_tokens = max(1, max_batch_tokens // per_image_tokens)
        chunk_size = min(batch_size, max_by_tokens)

        for start in range(0, len(members), chunk_size):
            chunk = members[start : start + chunk_size]
            encode_batches.append(EncodeBatch(
                pool_indices=chunk,
                resize_height=rh,
                resize_width=rw,
                token_count=per_image_tokens * len(chunk),
            ))
            exact_count += len(chunk)

    # --- Phase 2: Approximate batching for sparse leftovers ---
    approx_count = 0
    if leftover_indices:
        leftover_arr = np.array(leftover_indices, dtype=np.int64)
        lo_h = final_h[leftover_arr].astype(np.float32)
        lo_w = final_w[leftover_arr].astype(np.float32)

        if len(leftover_arr) <= batch_size:
            # Too few for k-means — chunk by token budget with averaged target
            avg_h = int(round(lo_h.mean() / spatial_factor)) * spatial_factor
            avg_w = int(round(lo_w.mean() / spatial_factor)) * spatial_factor
            avg_h = max(avg_h, spatial_factor)
            avg_w = max(avg_w, spatial_factor)
            per_tok = estimate_image_tokens(avg_h, avg_w, spatial_factor=spatial_factor)
            max_by_tokens = max(1, max_batch_tokens // per_tok)
            chunk_sz = min(batch_size, max_by_tokens)
            for s in range(0, len(leftover_arr), chunk_sz):
                chunk = leftover_arr[s : s + chunk_sz]
                encode_batches.append(EncodeBatch(
                    pool_indices=chunk,
                    resize_height=avg_h,
                    resize_width=avg_w,
                    token_count=per_tok * len(chunk),
                ))
                approx_count += len(chunk)
        else:
            # K-means on post-resize features
            aspect = lo_w / lo_h
            log_area = np.log(lo_h * lo_w)
            features = np.stack([aspect, log_area], axis=1)

            fmin = features.min(axis=0)
            fmax = features.max(axis=0)
            frange = fmax - fmin
            frange[frange == 0] = 1.0
            features = (features - fmin) / frange
            features = np.ascontiguousarray(features, dtype=np.float32)

            n_lo = len(features)
            mean_tok = max(1.0, float(np.mean(
                (lo_h / spatial_factor) * (lo_w / spatial_factor),
            )))
            avg_batch_sz = min(max(1, int(max_batch_tokens / mean_tok)), batch_size)
            k = min(num_clusters, max(1, n_lo // avg_batch_sz))

            kmeans = faiss.Kmeans(d=2, k=k, niter=10, verbose=False, gpu=gpu_kmeans)
            kmeans.train(features)
            _, labels = kmeans.index.search(features, 1)
            labels = labels.ravel()

            for cluster_id in range(k):
                c_members = np.where(labels == cluster_id)[0]
                if len(c_members) == 0:
                    continue

                # Sort by log_area within cluster
                order = np.argsort(log_area[c_members])
                c_members = c_members[order]
                c_pool_indices = leftover_arr[c_members]

                # Compute batch target from cluster's post-resize dims
                c_h = lo_h[c_members]
                c_w = lo_w[c_members]
                avg_h = int(round(c_h.mean() / spatial_factor)) * spatial_factor
                avg_w = int(round(c_w.mean() / spatial_factor)) * spatial_factor
                avg_h = max(avg_h, spatial_factor)
                avg_w = max(avg_w, spatial_factor)

                # Distortion guard: split cluster if spread is too wide
                if len(c_members) > 1:
                    ar = c_w / c_h
                    area = c_h * c_w
                    ar_spread = float(ar.max() / max(ar.min(), 1e-6))
                    area_spread = float(area.max() / max(area.min(), 1.0))
                    if ar_spread > max_ar_spread or area_spread > max_area_spread:
                        # Fall back to smaller chunks
                        mid = len(c_members) // 2
                        for sub_members in [c_members[:mid], c_members[mid:]]:
                            sub_pool = leftover_arr[sub_members]
                            sub_h = lo_h[sub_members]
                            sub_w = lo_w[sub_members]
                            sh = int(round(sub_h.mean() / spatial_factor)) * spatial_factor
                            sw = int(round(sub_w.mean() / spatial_factor)) * spatial_factor
                            sh = max(sh, spatial_factor)
                            sw = max(sw, spatial_factor)
                            per_tok = estimate_image_tokens(sh, sw, spatial_factor=spatial_factor)
                            max_by = max(1, max_batch_tokens // per_tok)
                            cs = min(batch_size, max_by)
                            for s in range(0, len(sub_pool), cs):
                                chunk = sub_pool[s : s + cs]
                                encode_batches.append(EncodeBatch(
                                    pool_indices=chunk,
                                    resize_height=sh,
                                    resize_width=sw,
                                    token_count=per_tok * len(chunk),
                                ))
                                approx_count += len(chunk)
                        continue

                per_tok = estimate_image_tokens(avg_h, avg_w, spatial_factor=spatial_factor)
                max_by = max(1, max_batch_tokens // per_tok)
                cs = min(batch_size, max_by)
                for s in range(0, len(c_pool_indices), cs):
                    chunk = c_pool_indices[s : s + cs]
                    encode_batches.append(EncodeBatch(
                        pool_indices=chunk,
                        resize_height=avg_h,
                        resize_width=avg_w,
                        token_count=per_tok * len(chunk),
                    ))
                    approx_count += len(chunk)

    logger.info(
        f"Chunk encode plan: {len(pool):,} images, "
        f"{len(encode_batches):,} batches "
        f"(exact={exact_count:,}, approx={approx_count:,})"
    )

    return ChunkEncodePlan(
        pool=pool,
        encode_batches=encode_batches,
        total_images=len(pool),
        exact_batched=exact_count,
        approx_batched=approx_count,
    )
