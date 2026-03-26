"""TokenizationPlan: single source of truth for what to tokenize and how.

Contains both logical truth (documents, components, source refs) and execution
layout (image batches with resize targets).  The plan is built deterministically
from a manifest + mode, serialized once, and consumed by the executor and rebuild.

``(document_id, component_index)`` is the only logical identity.  Execution
fields (batch assignment, rank assignment) never define identity.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

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
    """Single source of truth: documents + components + image batches.

    ``(document_id, component_index)`` is the stable logical identity.
    ``image_batches`` is the execution layout for GPU encoding.
    """

    # --- Documents (one row per logical document/sample) ---
    doc_document_id: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    doc_output_order: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    doc_num_components: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))

    # --- Components (one row per component across all documents) ---
    comp_document_id: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    comp_component_index: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))
    comp_kind: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))
    comp_source_kind: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))
    comp_source_ref: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    comp_manifest_row: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    comp_image_index: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))
    comp_width: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    comp_height: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))

    # --- Image batches (execution layout) ---
    image_batches: List[ImageBatch] = field(default_factory=list)
    # window_batch_offsets[i] = first batch index of window i.
    # split_image_batches_for_workers cuts on window boundaries only.
    window_batch_offsets: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    # --- Metadata ---
    metadata: PlanMetadata = field(default_factory=lambda: PlanMetadata("", "", ""))

    @property
    def total_documents(self) -> int:
        return len(self.doc_document_id)

    @property
    def total_components(self) -> int:
        return len(self.comp_document_id)

    @property
    def total_image_components(self) -> int:
        return int((self.comp_kind == IMAGE).sum())

    @property
    def total_text_components(self) -> int:
        return int((self.comp_kind == TEXT).sum())

    @property
    def total_batches(self) -> int:
        return len(self.image_batches)

    @property
    def mode(self) -> str:
        return self.metadata.mode

    def split_image_batches_for_workers(
        self, num_workers: int,
    ) -> List[List[ImageBatch]]:
        """Split image_batches into contiguous chunks balanced by token cost.

        Cuts only on window boundaries so no document is split across ranks.
        Each window's batches are assigned atomically to one rank.
        """
        n = len(self.image_batches)
        if n == 0:
            return [[] for _ in range(num_workers)]

        offsets = self.window_batch_offsets
        if len(offsets) == 0:
            # Fallback: no window info, treat each batch as its own window
            offsets = np.arange(n, dtype=np.int64)

        num_windows = len(offsets)
        # Compute per-window cost
        window_costs = []
        for wi in range(num_windows):
            ws = int(offsets[wi])
            we = int(offsets[wi + 1]) if wi + 1 < num_windows else n
            cost = sum(float(self.image_batches[i].batch_token_count) for i in range(ws, we))
            window_costs.append(cost)

        # Split windows across workers
        from vision_tokenization.utils.partitioning import weighted_contiguous_split
        window_indices = list(range(num_windows))
        window_splits = weighted_contiguous_split(window_indices, window_costs, num_workers)

        # Map window splits back to batch splits
        result = []
        for worker_windows in window_splits:
            if not worker_windows:
                result.append([])
                continue
            first_win = worker_windows[0]
            last_win = worker_windows[-1]
            batch_start = int(offsets[first_win])
            batch_end = int(offsets[last_win + 1]) if last_win + 1 < num_windows else n
            result.append(self.image_batches[batch_start:batch_end])

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

    Returns ``(batches, window_batch_offsets)`` where
    ``window_batch_offsets[i]`` is the index of the first batch in window *i*.

    Window boundaries are snapped to document boundaries so no document
    is split across windows.  This ensures ``split_image_batches_for_workers``
    (which cuts on window boundaries) never splits a document across ranks.
    """
    N = len(comp_indices)
    if N == 0:
        return []

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
    window_batch_offsets: List[int] = []

    for wi in range(num_windows):
        window_batch_offsets.append(len(all_batches))
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
            from vision_tokenization.indexing.planning.batch_planner import (
                _pack_spillover_local,
            )
            spillover = np.concatenate(spillover_parts)
            spill_batches = _pack_spillover_local(
                spillover, comp_indices, final_h, final_w,
                batch_size=batch_size, max_batch_tokens=max_batch_tokens,
                spatial_factor=spatial_factor,
            )
            for sb in spill_batches:
                all_batches.append(ImageBatch(
                    component_indices=sb.sample_indices,
                    resize_height=sb.resize_height,
                    resize_width=sb.resize_width,
                    batch_token_count=sb.batch_token_count,
                ))

    return all_batches, np.array(window_batch_offsets, dtype=np.int64)


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
        doc_document_id=doc_ids,
        doc_output_order=doc_ids.copy(),
        doc_num_components=np.ones(N, dtype=np.int16),

        comp_document_id=doc_ids.copy(),
        comp_component_index=np.zeros(N, dtype=np.int16),
        comp_kind=np.full(N, IMAGE, dtype=np.int8),
        comp_source_kind=np.full(N, SOURCE_MANIFEST_ROW, dtype=np.int8),
        comp_source_ref=valid_idx,
        comp_manifest_row=valid_idx,
        comp_image_index=np.zeros(N, dtype=np.int16),
        comp_width=widths[valid_idx],
        comp_height=heights[valid_idx],
    )

    # Build image batches
    plan.image_batches, plan.window_batch_offsets = _plan_image_batches(
        comp_indices, valid_idx, widths[valid_idx], heights[valid_idx],
        doc_ids,  # each image is its own document
        batch_size=batch_size, max_batch_tokens=max_batch_tokens,
        spatial_factor=spatial_factor, resize_min_pixels=resize_min_pixels,
        resize_max_pixels=resize_max_pixels, window_size=window_size,
    )

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
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
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
    comp_manifest_row = np.full(n_total_comps, -1, dtype=np.int64)
    comp_image_index = np.full(n_total_comps, -1, dtype=np.int16)
    comp_width = np.zeros(n_total_comps, dtype=np.int32)
    comp_height = np.zeros(n_total_comps, dtype=np.int32)

    # Image components
    comp_document_id[:n_image_comps] = doc_inverse
    comp_component_index[:n_image_comps] = valid_img_idx
    comp_kind[:n_image_comps] = IMAGE
    comp_source_ref[:n_image_comps] = valid_idx
    comp_manifest_row[:n_image_comps] = valid_idx
    comp_image_index[:n_image_comps] = valid_img_idx
    comp_width[:n_image_comps] = widths[valid_idx]
    comp_height[:n_image_comps] = heights[valid_idx]

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
    comp_manifest_row[text_start:] = doc_first_row

    logger.info(
        f"{mode} plan: {N_docs:,} documents, {n_image_comps:,} image + "
        f"{n_text_comps:,} text components ({total - N_rows:,} rows filtered)"
    )

    # Build image batches from image components only
    image_comp_indices = np.arange(n_image_comps, dtype=np.int64)

    plan = TokenizationPlan(
        doc_document_id=np.arange(N_docs, dtype=np.int64),
        doc_output_order=np.arange(N_docs, dtype=np.int64),
        doc_num_components=components_per_doc,

        comp_document_id=comp_document_id,
        comp_component_index=comp_component_index,
        comp_kind=comp_kind,
        comp_source_kind=comp_source_kind,
        comp_source_ref=comp_source_ref,
        comp_manifest_row=comp_manifest_row,
        comp_image_index=comp_image_index,
        comp_width=comp_width,
        comp_height=comp_height,
    )

    plan.image_batches, plan.window_batch_offsets = _plan_image_batches(
        image_comp_indices,
        comp_manifest_row[:n_image_comps],
        comp_width[:n_image_comps],
        comp_height[:n_image_comps],
        comp_document_id[:n_image_comps],
        batch_size=batch_size, max_batch_tokens=max_batch_tokens,
        spatial_factor=spatial_factor, resize_min_pixels=resize_min_pixels,
        resize_max_pixels=resize_max_pixels, window_size=window_size,
    )

    plan.metadata = PlanMetadata(
        manifest_path=manifest_path,
        manifest_fingerprint=_manifest_fingerprint(manifest_path),
        mode=mode, text_column=text_column,
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
        doc_document_id=np.arange(n_docs, dtype=np.int64),
        doc_output_order=np.arange(n_docs, dtype=np.int64),
        # For interleave, this field tracks the number of image occurrences in
        # the document; total segment/component count is runtime-discovered.
        doc_num_components=images_per_doc,

        comp_document_id=doc_inverse.astype(np.int64, copy=False),
        comp_component_index=valid_img_idx,
        comp_kind=np.full(n_rows, IMAGE, dtype=np.int8),
        comp_source_kind=np.full(n_rows, SOURCE_MANIFEST_ROW, dtype=np.int8),
        comp_source_ref=valid_idx,
        comp_manifest_row=valid_idx,
        comp_image_index=valid_img_idx,
        comp_width=widths[valid_idx],
        comp_height=heights[valid_idx],
    )

    plan.image_batches, plan.window_batch_offsets = _plan_image_batches(
        comp_indices,
        valid_idx,
        widths[valid_idx],
        heights[valid_idx],
        plan.comp_document_id,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        spatial_factor=spatial_factor,
        resize_min_pixels=resize_min_pixels,
        resize_max_pixels=resize_max_pixels,
        window_size=window_size,
    )

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
        parser: Interleave parser name (pin200m, shizhen, medpix).
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

    if mode == "image_only":
        return build_plan_image_only(manifest_path, **common)
    elif mode in ("image2text", "text2image"):
        return build_plan_image2text(
            manifest_path, text_column=text_column, mode=mode, **common,
        )
    elif mode == "sft":
        # SFT: one text component (conversation) + N image components per doc.
        # Same structure as image2text — text component is the full conversation,
        # image components are the images from the group.
        # The rebuild handles placeholder replacement.
        return build_plan_image2text(
            manifest_path, text_column=text_column, mode="sft", **common,
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
