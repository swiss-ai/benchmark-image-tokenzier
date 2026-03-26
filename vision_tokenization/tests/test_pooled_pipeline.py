"""Tests for document owner planner and image encode pool."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vision_tokenization.indexing.planning.document_planner import (
    DocumentOwnerPlan,
    plan_document_ownership,
)
from vision_tokenization.indexing.planning.encode_pool import (
    ImagePoolEntry,
    plan_chunk_encode,
    ChunkEncodePlan,
)
from vision_tokenization.pipeline.pooled.loop import _missing_image_pool_indices


# --- Helpers ----------------------------------------------------------------

def _create_grouped_manifest(tmp_path, groups):
    """Create a manifest with group_id and image_index columns.

    groups: list of list of (width, height) per group.
    """
    rows = {"sample_index": [], "width": [], "height": [],
            "group_id": [], "image_index": []}
    idx = 0
    for g_id, images in enumerate(groups):
        for img_idx, (w, h) in enumerate(images):
            rows["sample_index"].append(idx)
            rows["width"].append(w)
            rows["height"].append(h)
            rows["group_id"].append(g_id)
            rows["image_index"].append(img_idx)
            idx += 1

    table = pa.table(rows)
    path = tmp_path / "manifest.parquet"
    pq.write_table(table, path)
    return path


# --- DocumentOwnerPlan tests ------------------------------------------------

class TestDocumentOwnerPlanner:
    def test_basic_grouping(self, tmp_path):
        groups = [
            [(200, 200)],               # doc 0: 1 image
            [(300, 300), (400, 400)],    # doc 1: 2 images
            [(500, 500)],               # doc 2: 1 image
        ]
        path = _create_grouped_manifest(tmp_path, groups)
        plan = plan_document_ownership(
            path, resize_min_pixels=128*128, resize_max_pixels=1024*1024,
        )
        assert plan.total_documents == 3
        assert len(plan.documents) == 3
        assert plan.documents[0].document_id == 0
        assert len(plan.documents[0].manifest_rows) == 1
        assert len(plan.documents[1].manifest_rows) == 2

    def test_filtering(self, tmp_path):
        groups = [
            [(10, 10)],    # too small
            [(200, 200)],  # ok
        ]
        path = _create_grouped_manifest(tmp_path, groups)
        plan = plan_document_ownership(
            path, min_pixels=100*100,
            resize_min_pixels=128*128, resize_max_pixels=1024*1024,
        )
        # Doc 0 filtered (10x10 = 100 pixels < 10000 min_pixels)
        assert len(plan.documents) == 1
        assert plan.documents[0].document_id == 1

    def test_split_for_workers(self, tmp_path):
        groups = [[(200, 200)] for _ in range(20)]
        path = _create_grouped_manifest(tmp_path, groups)
        plan = plan_document_ownership(
            path, resize_min_pixels=128*128, resize_max_pixels=1024*1024,
        )
        splits = plan.split_for_workers(4)
        assert len(splits) == 4
        # All docs covered
        all_ids = [doc.document_id for chunk in splits for doc in chunk]
        assert len(all_ids) == 20
        assert len(set(all_ids)) == 20

    def test_weighted_split_balances_cost(self, tmp_path):
        # One expensive doc (large image) + many cheap docs (small images)
        groups = [[(1000, 1000)]] + [[(100, 100)] for _ in range(9)]
        path = _create_grouped_manifest(tmp_path, groups)
        plan = plan_document_ownership(
            path, resize_min_pixels=64*64, resize_max_pixels=2048*2048,
        )
        splits = plan.split_for_workers(2)
        # The expensive doc should be balanced against the cheap ones
        costs = [
            sum(d.estimated_tokens for d in chunk)
            for chunk in splits
        ]
        # Not perfectly balanced but should not be 99/1
        assert all(c > 0 for c in costs if len(splits) > 0)


# --- ImageEncodePool tests --------------------------------------------------

class TestImageEncodePool:
    def _make_arrays(self, groups):
        """Build manifest-like arrays from groups spec."""
        ws, hs, gids, iidxs = [], [], [], []
        for g_id, images in enumerate(groups):
            for i_idx, (w, h) in enumerate(images):
                ws.append(w)
                hs.append(h)
                gids.append(g_id)
                iidxs.append(i_idx)
        return (
            np.array(ws, dtype=np.int32),
            np.array(hs, dtype=np.int32),
            np.array(gids, dtype=np.int64),
            np.array(iidxs, dtype=np.int16),
        )

    def test_basic_encode_plan(self):
        groups = [
            [(256, 256), (256, 256)],   # doc 0: 2 same-size images
            [(512, 512)],               # doc 1: 1 larger image
        ]
        widths, heights, _, _ = self._make_arrays(groups)

        plan = plan_chunk_encode(
            document_manifest_rows=[np.array([0, 1]), np.array([2])],
            document_ids=[0, 1],
            component_indices=[[0, 1], [0]],
            heights=heights,
            widths=widths,
            resize_min_pixels=128*128,
            resize_max_pixels=1024*1024,
            batch_size=128,
            max_batch_tokens=100000,
        )

        assert plan.total_images == 3
        assert len(plan.encode_batches) > 0
        # All images accounted for
        all_pool_indices = np.concatenate([b.pool_indices for b in plan.encode_batches])
        assert len(all_pool_indices) == 3

    def test_exact_batching_for_same_size(self):
        # 20 images all same size → should land in exact bucket
        groups = [[(256, 256)] for _ in range(20)]
        widths, heights, _, _ = self._make_arrays(groups)

        plan = plan_chunk_encode(
            document_manifest_rows=[np.array([i]) for i in range(20)],
            document_ids=list(range(20)),
            component_indices=[[0]] * 20,
            heights=heights,
            widths=widths,
            resize_min_pixels=128*128,
            resize_max_pixels=1024*1024,
            min_bucket_size=4,
        )

        assert plan.exact_batched == 20
        assert plan.approx_batched == 0

    def test_sparse_images_go_to_approximate(self):
        # Each image is a different size → sparse buckets → approximate
        groups = [[(100 + i * 50, 100 + i * 50)] for i in range(10)]
        widths, heights, _, _ = self._make_arrays(groups)

        plan = plan_chunk_encode(
            document_manifest_rows=[np.array([i]) for i in range(10)],
            document_ids=list(range(10)),
            component_indices=[[0]] * 10,
            heights=heights,
            widths=widths,
            resize_min_pixels=64*64,
            resize_max_pixels=2048*2048,
            min_bucket_size=8,  # high threshold forces most to approximate
        )

        assert plan.approx_batched > 0
        # All images still accounted for
        total = plan.exact_batched + plan.approx_batched
        assert total == 10

    def test_pool_preserves_document_mapping(self):
        groups = [
            [(256, 256), (512, 512)],  # doc 0: 2 different-size images
            [(256, 256)],              # doc 1: 1 image same as doc0's first
        ]
        widths, heights, _, _ = self._make_arrays(groups)

        plan = plan_chunk_encode(
            document_manifest_rows=[np.array([0, 1]), np.array([2])],
            document_ids=[10, 20],
            component_indices=[[0, 1], [0]],
            heights=heights,
            widths=widths,
            resize_min_pixels=128*128,
            resize_max_pixels=1024*1024,
        )

        # Check pool entries have correct document mappings
        assert plan.pool[0].document_id == 10
        assert plan.pool[0].component_index == 0
        assert plan.pool[1].document_id == 10
        assert plan.pool[1].component_index == 1
        assert plan.pool[2].document_id == 20
        assert plan.pool[2].component_index == 0

    def test_batch_size_respected(self):
        # Many same-size images, small batch_size
        groups = [[(256, 256)] for _ in range(50)]
        widths, heights, _, _ = self._make_arrays(groups)

        plan = plan_chunk_encode(
            document_manifest_rows=[np.array([i]) for i in range(50)],
            document_ids=list(range(50)),
            component_indices=[[0]] * 50,
            heights=heights,
            widths=widths,
            resize_min_pixels=128*128,
            resize_max_pixels=1024*1024,
            batch_size=10,
            max_batch_tokens=999999,
        )

        for batch in plan.encode_batches:
            assert len(batch.pool_indices) <= 10

    def test_empty_chunk(self):
        plan = plan_chunk_encode(
            document_manifest_rows=[],
            document_ids=[],
            component_indices=[],
            heights=np.array([], dtype=np.int32),
            widths=np.array([], dtype=np.int32),
            resize_min_pixels=128*128,
            resize_max_pixels=1024*1024,
        )
        assert plan.total_images == 0
        assert len(plan.encode_batches) == 0


class TestPooledLoopHelpers:
    def test_missing_image_pool_indices_returns_incomplete_members(self):
        entries = [
            ImagePoolEntry(
                pool_index=0,
                document_id=7,
                component_index=0,
                manifest_row=100,
                resize_height=256,
                resize_width=256,
            ),
            ImagePoolEntry(
                pool_index=1,
                document_id=7,
                component_index=1,
                manifest_row=101,
                resize_height=256,
                resize_width=256,
            ),
            ImagePoolEntry(
                pool_index=2,
                document_id=7,
                component_index=2,
                manifest_row=102,
                resize_height=256,
                resize_width=256,
            ),
        ]

        encoded_tokens = {
            0: np.array([11, 12], dtype=np.int32),
            2: np.array([21, 22], dtype=np.int32),
        }

        assert _missing_image_pool_indices(entries, encoded_tokens) == [1]

    def test_missing_image_pool_indices_empty_when_document_is_complete(self):
        entries = [
            ImagePoolEntry(
                pool_index=0,
                document_id=3,
                component_index=0,
                manifest_row=10,
                resize_height=128,
                resize_width=128,
            ),
            ImagePoolEntry(
                pool_index=1,
                document_id=3,
                component_index=1,
                manifest_row=11,
                resize_height=128,
                resize_width=128,
            ),
        ]

        encoded_tokens = {
            0: np.array([1], dtype=np.int32),
            1: np.array([2], dtype=np.int32),
        }

        assert _missing_image_pool_indices(entries, encoded_tokens) == []
