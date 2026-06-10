"""Tokenization planning: logical dataset index plus execution layout.

The persisted plan artifact intentionally separates:

- logical dataset state: documents and components
- execution state: image batches and safe split boundaries

``(document_id, component_index)`` is the only logical identity. Execution
fields (batch assignment, rank assignment) never define identity.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Union

import numpy as np
import pyarrow.parquet as pq

from vision_tokenization.indexing.manifest import load_group_arrays, load_resolution_arrays
from vision_tokenization.utils.image_geometry import (
    estimate_image_tokens,
    smart_resize_dims_batch,
)

logger = logging.getLogger(__name__)

# Component kinds
IMAGE = np.int8(0)
TEXT = np.int8(1)

# Source kinds
SOURCE_MANIFEST_ROW = np.int8(0)
SOURCE_JSONL_SEGMENT = np.int8(1)
SOURCE_SFT_TURN = np.int8(2)

@dataclass
class ImageBatch:
    """One GPU encoding batch over image components."""

    component_indices: np.ndarray  # int64 — indexes into component arrays (image rows only)
    resize_height: int
    resize_width: int
    batch_token_count: int


@dataclass
class ImageBatchTable:
    """Compact columnar storage for image batches."""

    flat_component_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    batch_offsets: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    resize_heights: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    resize_widths: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    batch_token_counts: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    @classmethod
    def from_batches(cls, batches: List[ImageBatch]) -> "ImageBatchTable":
        if not batches:
            return cls()
        offsets = np.empty(len(batches), dtype=np.int64)
        total = sum(len(b.component_indices) for b in batches)
        flat = np.empty(total, dtype=np.int64)
        heights = np.empty(len(batches), dtype=np.int32)
        widths = np.empty(len(batches), dtype=np.int32)
        counts = np.empty(len(batches), dtype=np.int64)

        cursor = 0
        for i, batch in enumerate(batches):
            indices = np.asarray(batch.component_indices, dtype=np.int64)
            offsets[i] = cursor
            flat[cursor:cursor + len(indices)] = indices
            heights[i] = int(batch.resize_height)
            widths[i] = int(batch.resize_width)
            counts[i] = int(batch.batch_token_count)
            cursor += len(indices)

        return cls(
            flat_component_indices=flat,
            batch_offsets=offsets,
            resize_heights=heights,
            resize_widths=widths,
            batch_token_counts=counts,
        )

    def __len__(self) -> int:
        return len(self.batch_offsets)

    def __iter__(self) -> Iterator[ImageBatch]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        i = int(idx)
        start = int(self.batch_offsets[i])
        end = int(self.batch_offsets[i + 1]) if i + 1 < len(self.batch_offsets) else len(self.flat_component_indices)
        return ImageBatch(
            component_indices=self.flat_component_indices[start:end].copy(),
            resize_height=int(self.resize_heights[i]),
            resize_width=int(self.resize_widths[i]),
            batch_token_count=int(self.batch_token_counts[i]),
        )


@dataclass
class DocumentIndex:
    """Logical document inventory."""

    document_id: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    output_order: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    num_images: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))


@dataclass
class ComponentIndex:
    """Logical component inventory."""

    document_id: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    component_index: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))
    kind: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))
    source_kind: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))
    source_ref: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    image_index: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))


@dataclass
class ExecutionPlan:
    """Execution-only layout for image encoding."""

    image_batches: ImageBatchTable = field(default_factory=ImageBatchTable)
    # First batch index of each safe split segment.
    split_batch_offsets: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))


@dataclass
class PlanMetadata:
    """Inputs that produced this plan — used for cache invalidation."""

    manifest_path: str
    manifest_fingerprint: str
    mode: str
    parser: Optional[str] = None
    text_column: Optional[str] = None
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    window_size: int = 2000
    batch_size: int = 128
    max_batch_tokens: int = 32768
    resize_min_pixels: int = 16384
    resize_max_pixels: int = 1960000
    spatial_factor: int = 16

    def matches(self, other: PlanMetadata) -> bool:
        """Return True if all fields match (plan is still valid)."""
        return (
            self.manifest_fingerprint == other.manifest_fingerprint
            and self.mode == other.mode
            and self.parser == other.parser
            and self.text_column == other.text_column
            and self.min_pixels == other.min_pixels
            and self.max_pixels == other.max_pixels
            and self.window_size == other.window_size
            and self.batch_size == other.batch_size
            and self.max_batch_tokens == other.max_batch_tokens
            and self.resize_min_pixels == other.resize_min_pixels
            and self.resize_max_pixels == other.resize_max_pixels
            and self.spatial_factor == other.spatial_factor
        )


@dataclass
class TokenizationPlan:
    """Single persisted plan artifact: logical index + execution layout.

    ``(document_id, component_index)`` is the stable logical identity.
    ``execution.image_batches`` is the execution layout for GPU encoding.
    """

    documents: DocumentIndex = field(default_factory=DocumentIndex)
    components: ComponentIndex = field(default_factory=ComponentIndex)
    execution: ExecutionPlan = field(default_factory=ExecutionPlan)
    metadata: PlanMetadata = field(default_factory=lambda: PlanMetadata("", "", ""))

    @property
    def total_documents(self) -> int:
        return len(self.documents.document_id)

    @property
    def total_components(self) -> int:
        return len(self.components.document_id)

    @property
    def total_image_components(self) -> int:
        return int((self.components.kind == IMAGE).sum())

    @property
    def total_text_components(self) -> int:
        return int((self.components.kind == TEXT).sum())

    @property
    def total_batches(self) -> int:
        return len(self.execution.image_batches)

    @property
    def mode(self) -> str:
        return self.metadata.mode

    def fingerprint(self) -> dict:
        """Cheap identity for resume safety: a checkpoint's batch_index is only
        valid against the exact plan it was counted on."""
        return {
            "manifest_fingerprint": self.metadata.manifest_fingerprint,
            "total_batches": int(self.total_batches),
            "total_tokens": int(
                np.asarray(self.execution.image_batches.batch_token_counts, dtype=np.int64).sum()
            ),
        }

    def split_image_batches_for_workers(
        self, num_workers: int,
    ) -> List[List[ImageBatch]]:
        """Split image_batches into contiguous chunks balanced by token cost.

        Two policies based on the structural invariant:
        - One image per document: split on batch boundaries — every
          boundary is document-safe, gives optimal balance.
        - Multiple images per document: split on window boundaries
          only — guarantees all of a document's images land on one rank.
        """
        image_batches = self.execution.image_batches
        n = len(image_batches)
        if n == 0:
            return [[] for _ in range(num_workers)]

        from vision_tokenization.utils.partitioning import weighted_contiguous_split

        costs = image_batches.batch_token_counts.astype(float).tolist()
        one_image_per_doc = bool(
            self.total_documents > 0 and np.all(self.documents.num_images == 1)
        )

        if one_image_per_doc:
            return weighted_contiguous_split(image_batches, costs, num_workers)

        # Multi-image: split on window boundaries
        offsets = self.execution.split_batch_offsets
        if len(offsets) == 0:
            raise ValueError(
                "Multi-image TokenizationPlan missing split_batch_offsets; "
                "regenerate the plan."
            )

        num_segments = len(offsets)
        segment_costs = []
        for si in range(num_segments):
            bs = int(offsets[si])
            be = int(offsets[si + 1]) if si + 1 < num_segments else n
            segment_costs.append(sum(costs[bs:be]))

        segment_indices = list(range(num_segments))
        segment_splits = weighted_contiguous_split(
            segment_indices, segment_costs, num_workers,
        )

        result = []
        for worker_segments in segment_splits:
            if not worker_segments:
                result.append([])
                continue
            first_seg = worker_segments[0]
            last_seg = worker_segments[-1]
            batch_start = int(offsets[first_seg])
            batch_end = int(offsets[last_seg + 1]) if last_seg + 1 < num_segments else n
            result.append(image_batches[batch_start:batch_end])

        return result


# ---------------------------------------------------------------------------
# Manifest fingerprint
# ---------------------------------------------------------------------------


def _manifest_fingerprint(path: str) -> str:
    """Compute a fast fingerprint from manifest file metadata."""
    p = Path(path)
    stat = p.stat()
    # Combine path, size, and mtime for a fast fingerprint
    raw = f"{p.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Pixel filter (shared across modes)
# ---------------------------------------------------------------------------


def _pixel_filter(
    widths: np.ndarray,
    heights: np.ndarray,
    spatial_factor: int,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
) -> np.ndarray:
    """Return boolean mask of valid images."""
    pixels = widths.astype(np.int64) * heights.astype(np.int64)
    mask = np.ones(len(widths), dtype=bool)
    if min_pixels is not None:
        mask &= pixels >= min_pixels
    if max_pixels is not None:
        mask &= pixels <= max_pixels
    mask &= (widths >= spatial_factor) & (heights >= spatial_factor)
    return mask


# ---------------------------------------------------------------------------
# Image batch planning (reuses locality windowing logic)
# ---------------------------------------------------------------------------


def _plan_image_batches(
    comp_indices: np.ndarray,
    comp_manifest_row: np.ndarray,
    comp_width: np.ndarray,
    comp_height: np.ndarray,
    comp_document_id: np.ndarray,
    *,
    batch_size: int,
    max_batch_tokens: int,
    spatial_factor: int,
    resize_min_pixels: int,
    resize_max_pixels: int,
    window_size: int,
) -> tuple:
    """Build locality-aware image batches from image components.

    Returns ``(batches, split_batch_offsets)`` where
    ``split_batch_offsets[i]`` is the index of the first batch in safe split
    segment *i*.

    Window boundaries are snapped to document boundaries so no document
    is split across windows.  This ensures ``split_image_batches_for_workers``
    (which cuts on window boundaries) never splits a document across ranks.
    """
    N = len(comp_indices)
    if N == 0:
        return [], np.array([], dtype=np.int64)

    # Post-resize dims
    final_h, final_w = smart_resize_dims_batch(
        comp_height, comp_width,
        min_pixels=resize_min_pixels, max_pixels=resize_max_pixels,
        factor=spatial_factor,
    )
    keys = final_h.astype(np.int64) * 100_000 + final_w.astype(np.int64)

    # Pre-compute token budget per resolution key
    unique_keys = np.unique(keys)
    key_info = {}
    for k in unique_keys:
        k = int(k)
        rh, rw = k // 100_000, k % 100_000
        per_tok = estimate_image_tokens(rh, rw, spatial_factor=spatial_factor)
        key_info[k] = (rh, rw, per_tok, min(batch_size, max(1, max_batch_tokens // per_tok)))

    # Window by manifest_row position, snapped to document boundaries.
    # Raw windows from manifest position:
    raw_window_ids = comp_manifest_row // window_size

    if N > 1:
        raw_changes = np.where(np.diff(raw_window_ids) > 0)[0] + 1
        # Snap each boundary forward to the next document boundary:
        # if doc_id at boundary == doc_id before it, shift forward.
        snapped = []
        for pos in raw_changes:
            while pos < N and comp_document_id[pos] == comp_document_id[pos - 1]:
                pos += 1
            if pos < N:
                snapped.append(pos)
        win_starts = np.array([0] + snapped, dtype=np.int64)
        win_ends = np.array(snapped + [N], dtype=np.int64)
    else:
        win_starts = np.array([0], dtype=np.int64)
        win_ends = np.array([N], dtype=np.int64)

    num_windows = len(win_starts)

    # Process each window, tracking where each window's batches start
    all_batches: List[ImageBatch] = []
    split_batch_offsets: List[int] = []

    for wi in range(num_windows):
        split_batch_offsets.append(len(all_batches))
        ws = int(win_starts[wi])
        we = int(win_ends[wi])
        win_keys = keys[ws:we]
        wn = we - ws

        local_order = np.argsort(win_keys, kind="stable")
        sorted_keys = win_keys[local_order]
        sorted_global = local_order + ws  # indices into comp_indices

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
                all_batches.append(ImageBatch(
                    component_indices=comp_indices[chunk],
                    resize_height=rh, resize_width=rw,
                    batch_token_count=per_tok * len(chunk),
                ))

            if n_full < run_len:
                spillover_parts.append(run[n_full:])

        if spillover_parts:
            spillover = np.concatenate(spillover_parts)
            all_batches.extend(_pack_spillover(
                spillover, comp_indices, final_h, final_w,
                batch_size=batch_size, max_batch_tokens=max_batch_tokens,
                spatial_factor=spatial_factor,
            ))

    return all_batches, np.array(split_batch_offsets, dtype=np.int64)


def _pack_spillover(
    arr: np.ndarray,
    comp_indices: np.ndarray,
    final_h: np.ndarray,
    final_w: np.ndarray,
    *,
    batch_size: int,
    max_batch_tokens: int,
    spatial_factor: int,
) -> List[ImageBatch]:
    """K-means on aspect ratio + log area, then greedy-pack within clusters."""
    import faiss

    if len(arr) == 0:
        return []

    sh = final_h[arr]
    sw = final_w[arr]

    aspect = sw.astype(np.float32) / sh.astype(np.float32)
    log_area = np.log(sh.astype(np.float32) * sw.astype(np.float32))
    features = np.stack([aspect, log_area], axis=1)
    fmin = features.min(axis=0)
    frange = features.max(axis=0) - fmin
    frange[frange == 0] = 1.0
    features = np.ascontiguousarray((features - fmin) / frange, dtype=np.float32)

    N = len(arr)
    mean_tok = max(1.0, float(np.mean((sh // spatial_factor) * (sw // spatial_factor))))
    avg_batch = min(max(1, int(max_batch_tokens / mean_tok)), batch_size)
    k = max(1, min(N // max(1, avg_batch), N // 39))

    def _make_batches(members_arr, rh, rw):
        per_tok = estimate_image_tokens(rh, rw, spatial_factor=spatial_factor)
        chunk_size = min(batch_size, max(1, max_batch_tokens // per_tok))
        return [
            ImageBatch(
                component_indices=comp_indices[arr[members_arr[s : s + chunk_size]]],
                resize_height=rh, resize_width=rw,
                batch_token_count=per_tok * min(chunk_size, len(members_arr) - s),
            )
            for s in range(0, len(members_arr), chunk_size)
        ]

    if k <= 1:
        avg_h = max(spatial_factor, int(round(float(sh.mean()) / spatial_factor)) * spatial_factor)
        avg_w = max(spatial_factor, int(round(float(sw.mean()) / spatial_factor)) * spatial_factor)
        return _make_batches(np.arange(N), avg_h, avg_w)

    prev_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        kmeans = faiss.Kmeans(d=2, k=k, niter=5, verbose=False, gpu=False)
        kmeans.train(features)
    finally:
        faiss.omp_set_num_threads(prev_threads)
    _, labels = kmeans.index.search(features, 1)
    labels = labels.ravel()

    batches: List[ImageBatch] = []
    for cid in range(k):
        members = np.where(labels == cid)[0]
        if len(members) == 0:
            continue
        c_h, c_w = sh[members], sw[members]
        avg_h = max(spatial_factor, int(round(float(c_h.mean()) / spatial_factor)) * spatial_factor)
        avg_w = max(spatial_factor, int(round(float(c_w.mean()) / spatial_factor)) * spatial_factor)
        members = members[np.argsort(c_h * c_w)]
        batches.extend(_make_batches(members, avg_h, avg_w))

    return batches


# ---------------------------------------------------------------------------
# Plan builders per mode
# ---------------------------------------------------------------------------


def build_plan_image_only(
    manifest_path: Union[str, Path],
    *,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    batch_size: int = 128,
    max_batch_tokens: int = 32768,
    spatial_factor: int = 16,
    resize_min_pixels: int = 16384,
    resize_max_pixels: int = 1960000,
    window_size: int = 2000,
) -> TokenizationPlan:
    """Build plan for image_only mode: one document per image, one component."""
    manifest_path = str(manifest_path)
    widths, heights = load_resolution_arrays(manifest_path)
    total = len(widths)

    valid_mask = _pixel_filter(widths, heights, spatial_factor, min_pixels, max_pixels)
    valid_idx = np.where(valid_mask)[0]
    N = len(valid_idx)

    logger.info(f"image_only plan: {N:,} valid images ({total - N:,} filtered)")

    # Documents: one per valid image
    doc_ids = np.arange(N, dtype=np.int64)

    # Components: one IMAGE per document
    comp_indices = np.arange(N, dtype=np.int64)

    plan = TokenizationPlan(
        documents=DocumentIndex(
            document_id=doc_ids,
            output_order=doc_ids.copy(),
            num_images=np.ones(N, dtype=np.int16),
        ),
        components=ComponentIndex(
            document_id=doc_ids.copy(),
            component_index=np.zeros(N, dtype=np.int16),
            kind=np.full(N, IMAGE, dtype=np.int8),
            source_kind=np.full(N, SOURCE_MANIFEST_ROW, dtype=np.int8),
            source_ref=valid_idx,
            image_index=np.zeros(N, dtype=np.int16),
        ),
    )

    # Build image batches
    image_batches, split_batch_offsets = _plan_image_batches(
        comp_indices, valid_idx, widths[valid_idx], heights[valid_idx],
        doc_ids,  # each image is its own document
        batch_size=batch_size, max_batch_tokens=max_batch_tokens,
        spatial_factor=spatial_factor, resize_min_pixels=resize_min_pixels,
        resize_max_pixels=resize_max_pixels, window_size=window_size,
    )
    plan.execution.image_batches = ImageBatchTable.from_batches(image_batches)
    plan.execution.split_batch_offsets = split_batch_offsets

    plan.metadata = PlanMetadata(
        manifest_path=manifest_path,
        manifest_fingerprint=_manifest_fingerprint(manifest_path),
        mode="image_only",
        min_pixels=min_pixels, max_pixels=max_pixels,
        window_size=window_size, batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        resize_min_pixels=resize_min_pixels,
        resize_max_pixels=resize_max_pixels,
        spatial_factor=spatial_factor,
    )

    logger.info(
        f"image_only plan: {plan.total_documents:,} documents, "
        f"{plan.total_components:,} components, "
        f"{plan.total_batches:,} image batches"
    )
    return plan


def build_plan_image2text(
    manifest_path: Union[str, Path],
    *,
    text_column: Optional[str] = None,
    parser: Optional[str] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    max_images_per_doc: Optional[int] = None,
    batch_size: int = 128,
    max_batch_tokens: int = 32768,
    spatial_factor: int = 16,
    resize_min_pixels: int = 16384,
    resize_max_pixels: int = 1960000,
    window_size: int = 2000,
    mode: str = "image2text",
) -> TokenizationPlan:
    """Build plan for image2text or text2image mode.

    Each document has image component(s) + one text component.
    For single-image manifests (no group_id), one document per row.
    For multi-image manifests, one document per group_id.
    """
    manifest_path = str(manifest_path)

    # Check if manifest has group_id (multi-image)
    schema = pq.read_schema(manifest_path)
    has_groups = "group_id" in schema.names

    if has_groups:
        widths, heights, group_ids, image_indices = load_group_arrays(manifest_path)
    else:
        widths, heights = load_resolution_arrays(manifest_path)
        total = len(widths)
        group_ids = np.arange(total, dtype=np.int64)
        image_indices = np.zeros(total, dtype=np.int16)

    total = len(widths)
    valid_mask = _pixel_filter(widths, heights, spatial_factor, min_pixels, max_pixels)

    if has_groups:
        # Group-level filter: drop entire group if ANY image fails (vectorized)
        unique_gids, inverse = np.unique(group_ids, return_inverse=True)
        # Mark groups with any invalid member
        invalid_groups = np.zeros(len(unique_gids), dtype=bool)
        np.logical_or.at(invalid_groups, inverse, ~valid_mask)
        valid_mask = ~invalid_groups[inverse]

    valid_idx = np.where(valid_mask)[0]
    N_rows = len(valid_idx)

    # Build documents from groups
    valid_gids = group_ids[valid_idx]
    valid_img_idx = image_indices[valid_idx]

    unique_docs, doc_inverse = np.unique(valid_gids, return_inverse=True)
    N_docs = len(unique_docs)

    # Count images per document (vectorized)
    images_per_doc = np.bincount(doc_inverse, minlength=N_docs).astype(np.int16)

    if max_images_per_doc is not None:
        keep_docs = images_per_doc <= max_images_per_doc
        if not keep_docs.all():
            n_dropped = int((~keep_docs).sum())
            logger.info(
                "max_images_per_doc=%d: dropping %d documents with too many images",
                max_images_per_doc,
                n_dropped,
            )
            keep_rows = keep_docs[doc_inverse]
            valid_idx = valid_idx[keep_rows]
            valid_gids = group_ids[valid_idx]
            valid_img_idx = image_indices[valid_idx]
            unique_docs, doc_inverse = np.unique(valid_gids, return_inverse=True)
            N_docs = len(unique_docs)
            N_rows = len(valid_idx)
            images_per_doc = np.bincount(doc_inverse, minlength=N_docs).astype(np.int16)

    components_per_doc = images_per_doc + 1  # images + 1 text

    # Build component arrays: image components first, then text components
    n_image_comps = N_rows
    n_text_comps = N_docs
    n_total_comps = n_image_comps + n_text_comps

    comp_document_id = np.empty(n_total_comps, dtype=np.int64)
    comp_component_index = np.empty(n_total_comps, dtype=np.int16)
    comp_kind = np.empty(n_total_comps, dtype=np.int8)
    comp_source_kind = np.full(n_total_comps, SOURCE_MANIFEST_ROW, dtype=np.int8)
    comp_source_ref = np.empty(n_total_comps, dtype=np.int64)
    comp_image_index = np.full(n_total_comps, -1, dtype=np.int16)

    # Image components
    comp_document_id[:n_image_comps] = doc_inverse
    comp_component_index[:n_image_comps] = valid_img_idx
    comp_kind[:n_image_comps] = IMAGE
    comp_source_ref[:n_image_comps] = valid_idx
    comp_image_index[:n_image_comps] = valid_img_idx

    # Text components: one per document, component_index after all images
    # First manifest row per doc (vectorized via argsort)
    order = np.argsort(doc_inverse, kind="stable")
    _, first_positions = np.unique(doc_inverse[order], return_index=True)
    doc_first_row = valid_idx[order[first_positions]]

    text_start = n_image_comps
    comp_document_id[text_start:] = np.arange(N_docs, dtype=np.int64)
    comp_component_index[text_start:] = images_per_doc
    comp_kind[text_start:] = TEXT
    comp_source_ref[text_start:] = doc_first_row

    logger.info(
        f"{mode} plan: {N_docs:,} documents, {n_image_comps:,} image + "
        f"{n_text_comps:,} text components ({total - N_rows:,} rows filtered)"
    )

    # Build image batches from image components only
    image_comp_indices = np.arange(n_image_comps, dtype=np.int64)

    plan = TokenizationPlan(
        documents=DocumentIndex(
            document_id=np.arange(N_docs, dtype=np.int64),
            output_order=np.arange(N_docs, dtype=np.int64),
            num_images=images_per_doc,
        ),
        components=ComponentIndex(
            document_id=comp_document_id,
            component_index=comp_component_index,
            kind=comp_kind,
            source_kind=comp_source_kind,
            source_ref=comp_source_ref,
            image_index=comp_image_index,
        ),
    )

    image_batches, split_batch_offsets = _plan_image_batches(
        image_comp_indices,
        comp_source_ref[:n_image_comps],
        widths[valid_idx],
        heights[valid_idx],
        comp_document_id[:n_image_comps],
        batch_size=batch_size, max_batch_tokens=max_batch_tokens,
        spatial_factor=spatial_factor, resize_min_pixels=resize_min_pixels,
        resize_max_pixels=resize_max_pixels, window_size=window_size,
    )
    plan.execution.image_batches = ImageBatchTable.from_batches(image_batches)
    plan.execution.split_batch_offsets = split_batch_offsets

    plan.metadata = PlanMetadata(
        manifest_path=manifest_path,
        manifest_fingerprint=_manifest_fingerprint(manifest_path),
        mode=mode, parser=parser, text_column=text_column,
        min_pixels=min_pixels, max_pixels=max_pixels,
        window_size=window_size, batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        resize_min_pixels=resize_min_pixels,
        resize_max_pixels=resize_max_pixels,
        spatial_factor=spatial_factor,
    )

    logger.info(
        f"{mode} plan: {plan.total_documents:,} documents, "
        f"{plan.total_components:,} components, "
        f"{plan.total_batches:,} image batches"
    )
    return plan


def build_plan_interleave(
    manifest_path: Union[str, Path],
    *,
    text_column: Optional[str] = None,
    parser: Optional[str] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    batch_size: int = 128,
    max_batch_tokens: int = 32768,
    spatial_factor: int = 16,
    resize_min_pixels: int = 16384,
    resize_max_pixels: int = 1960000,
    window_size: int = 2000,
) -> TokenizationPlan:
    """Build plan for interleave mode.

    Interleave text/image segment order is discovered at runtime by parsing the
    source JSONL document.  The plan therefore stores only image components and
    per-document image counts; runtime spill assigns the final component_index
    values from parsed segment order.
    """
    manifest_path = str(manifest_path)

    widths, heights, group_ids, image_indices = load_group_arrays(manifest_path)
    total = len(widths)
    valid_mask = _pixel_filter(widths, heights, spatial_factor, min_pixels, max_pixels)

    # Interleave documents are grouped by group_id; drop the whole document if
    # any member image fails the pixel filter.
    unique_gids, inverse = np.unique(group_ids, return_inverse=True)
    invalid_groups = np.zeros(len(unique_gids), dtype=bool)
    np.logical_or.at(invalid_groups, inverse, ~valid_mask)
    valid_mask = ~invalid_groups[inverse]

    valid_idx = np.where(valid_mask)[0]
    n_rows = len(valid_idx)
    valid_gids = group_ids[valid_idx]
    valid_img_idx = image_indices[valid_idx]

    _unique_docs, doc_inverse = np.unique(valid_gids, return_inverse=True)
    n_docs = int(doc_inverse.max()) + 1 if n_rows > 0 else 0
    images_per_doc = np.bincount(doc_inverse, minlength=n_docs).astype(np.int16)

    logger.info(
        f"interleave plan: {n_docs:,} documents, {n_rows:,} image components "
        f"({total - n_rows:,} rows filtered)"
    )

    comp_indices = np.arange(n_rows, dtype=np.int64)
    plan = TokenizationPlan(
        documents=DocumentIndex(
            document_id=np.arange(n_docs, dtype=np.int64),
            output_order=np.arange(n_docs, dtype=np.int64),
            num_images=images_per_doc,
        ),
        components=ComponentIndex(
            document_id=doc_inverse.astype(np.int64, copy=False),
            component_index=valid_img_idx,
            kind=np.full(n_rows, IMAGE, dtype=np.int8),
            source_kind=np.full(n_rows, SOURCE_MANIFEST_ROW, dtype=np.int8),
            source_ref=valid_idx,
            image_index=valid_img_idx,
        ),
    )

    image_batches, split_batch_offsets = _plan_image_batches(
        comp_indices,
        valid_idx,
        widths[valid_idx],
        heights[valid_idx],
        plan.components.document_id,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        spatial_factor=spatial_factor,
        resize_min_pixels=resize_min_pixels,
        resize_max_pixels=resize_max_pixels,
        window_size=window_size,
    )
    plan.execution.image_batches = ImageBatchTable.from_batches(image_batches)
    plan.execution.split_batch_offsets = split_batch_offsets

    plan.metadata = PlanMetadata(
        manifest_path=manifest_path,
        manifest_fingerprint=_manifest_fingerprint(manifest_path),
        mode="interleave",
        parser=parser,
        text_column=text_column,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        window_size=window_size,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        resize_min_pixels=resize_min_pixels,
        resize_max_pixels=resize_max_pixels,
        spatial_factor=spatial_factor,
    )

    logger.info(
        f"interleave plan: {plan.total_documents:,} documents, "
        f"{plan.total_components:,} image components, "
        f"{plan.total_batches:,} image batches"
    )
    return plan


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_tokenization_plan(
    manifest_path: Union[str, Path],
    mode: str,
    *,
    text_column: Optional[str] = None,
    parser: Optional[str] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    max_images_per_doc: Optional[int] = None,
    batch_size: int = 128,
    max_batch_tokens: int = 32768,
    spatial_factor: int = 16,
    resize_min_pixels: int = 16384,
    resize_max_pixels: int = 1960000,
    window_size: int = 2000,
) -> TokenizationPlan:
    """Build a TokenizationPlan for the given mode.

    Args:
        manifest_path: Path to manifest parquet.
        mode: One of image_only, image2text, text2image, sft, interleave.
        text_column: Text column/field name for text loading.
        parser: Optional dataset parser name used at text load time.
        min_pixels, max_pixels: Pixel count filter bounds.
        batch_size, max_batch_tokens: Batch constraints.
        spatial_factor: Vision tokenizer spatial downsampling factor.
        resize_min_pixels, resize_max_pixels: Tokenizer resize bounds.
        window_size: Locality window size (manifest rows).

    Returns:
        A :class:`TokenizationPlan`.
    """
    common = dict(
        min_pixels=min_pixels, max_pixels=max_pixels,
        batch_size=batch_size, max_batch_tokens=max_batch_tokens,
        spatial_factor=spatial_factor,
        resize_min_pixels=resize_min_pixels,
        resize_max_pixels=resize_max_pixels,
        window_size=window_size,
    )
    image_doc_filter = dict(max_images_per_doc=max_images_per_doc)

    if mode == "image_only":
        return build_plan_image_only(manifest_path, **common)
    elif mode in ("image2text", "text2image"):
        return build_plan_image2text(
            manifest_path, text_column=text_column, parser=parser, mode=mode,
            **common, **image_doc_filter,
        )
    elif mode == "sft":
        return build_plan_image2text(
            manifest_path, text_column=text_column, parser=parser, mode="sft",
            **common, **image_doc_filter,
        )
    elif mode == "interleave":
        return build_plan_interleave(
            manifest_path,
            text_column=text_column,
            parser=parser,
            **common,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
