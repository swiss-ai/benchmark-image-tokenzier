"""Tests for vision_tokenization.indexing — CPU-only, no tokenizer needed."""

import io
import logging
import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from vision_tokenization.indexing.manifest import (
    load_hf_manifest,
    load_resolution_arrays,
    load_wds_manifest,
    save_wds_manifest,
)
from vision_tokenization.indexing.planning.tokenization_plan import (
    TokenizationPlan,
    build_tokenization_plan,
)
from vision_tokenization.indexing.reader import TarRandomAccessReader
from vision_tokenization.indexing.scanners._workers.hf_common import (
    build_hf_output_columns,
    scan_hf_batch_columns,
)
from vision_tokenization.indexing.scanners._workers.wds import scan_single_tar
from vision_tokenization.indexing.scanners.hf import scan_hf_dataset
from vision_tokenization.indexing.scanners.wds import scan_wds_dataset
from vision_tokenization.pipeline.runtime.data import HFImageLoader, WDSImageLoader
from vision_tokenization.pipeline.runtime.dry_run import dry_run_batch_plan
from vision_tokenization.utils.image_geometry import estimate_image_tokens, smart_resize_dims
from vision_tokenization.utils.json import json_load
from vision_tokenization.utils.partitioning import weighted_contiguous_split

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class BatchAssignment:
    """Test-side view of one planned batch in manifest-row coordinates."""

    sample_indices: np.ndarray
    resize_height: int
    resize_width: int
    batch_token_count: int


@dataclass
class BatchPlan:
    """Small test adapter over the current ``TokenizationPlan`` shape.

    The production planner no longer exposes the old ``BatchPlan`` API, but the
    behavioral assertions in this test file are still useful. This adapter lets
    the tests reason in terms of manifest rows and per-batch token cost without
    reintroducing the old product abstraction.
    """

    batches: list[BatchAssignment]
    total_samples: int = 0
    total_filtered: int = 0

    @staticmethod
    def _estimate_batch_cost(batch: BatchAssignment) -> float:
        return float(batch.batch_token_count)

    def split_for_workers(self, num_workers: int) -> list[list[BatchAssignment]]:
        costs = [self._estimate_batch_cost(batch) for batch in self.batches]
        return weighted_contiguous_split(self.batches, costs, num_workers)


def plan_clustered_batches(
    manifest_path: str,
    *,
    batch_size: int,
    max_batch_tokens: int,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    resize_min_pixels: int = 16384,
    resize_max_pixels: int = 1960000,
    spatial_factor: int = 16,
    multi_image: bool = False,
) -> BatchPlan:
    """Build a test-friendly batch plan from the current execution plan.

    The real planner persists component identities and execution batches. These
    tests mostly want the older manifest-row view, so we project each image
    batch back to its manifest rows via ``components.source_ref``.
    """
    metadata = pq.read_metadata(manifest_path)
    column_names = list(metadata.schema.names)
    total_rows = int(metadata.num_rows)

    if multi_image and "group_id" not in column_names:
        raise ValueError("multi_image=True but manifest has no group_id")

    plan = build_tokenization_plan(
        manifest_path=manifest_path,
        mode="image_only",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        resize_min_pixels=resize_min_pixels,
        resize_max_pixels=resize_max_pixels,
        spatial_factor=spatial_factor,
    )

    batches: list[BatchAssignment] = []
    for batch in plan.execution.image_batches:
        component_indices = np.asarray(batch.component_indices, dtype=np.int64)
        sample_indices = np.asarray(
            plan.components.source_ref[component_indices],
            dtype=np.int64,
        )
        batches.append(
            BatchAssignment(
                sample_indices=sample_indices,
                resize_height=int(batch.resize_height),
                resize_width=int(batch.resize_width),
                batch_token_count=int(batch.batch_token_count),
            )
        )

    return BatchPlan(
        batches=batches,
        total_samples=int(plan.total_image_components),
        total_filtered=max(0, total_rows - int(plan.total_image_components)),
    )


def _make_image(width: int, height: int, color: tuple = (255, 0, 0)) -> Image.Image:
    """Create a solid-colour RGB image."""
    return Image.new("RGB", (width, height), color)


def _image_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _create_tar(tar_path: str, samples: list):
    """Create a tar file with the given samples.

    Each sample is a dict with keys: key, ext, width, height, color.
    An optional ``text`` key adds a .txt sidecar.
    """
    with tarfile.open(tar_path, "w") as tf:
        for s in samples:
            img = _make_image(s["width"], s["height"], s.get("color", (255, 0, 0)))
            data = _image_bytes(img, "JPEG" if s["ext"] == "jpg" else s["ext"].upper())
            info = tarfile.TarInfo(name=f"{s['key']}.{s['ext']}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

            if "text" in s:
                txt = s["text"].encode()
                tinfo = tarfile.TarInfo(name=f"{s['key']}.txt")
                tinfo.size = len(txt)
                tf.addfile(tinfo, io.BytesIO(txt))


_HF_IMAGE_TYPE = pa.struct(
    [
        pa.field("bytes", pa.binary()),
        pa.field("path", pa.string()),
    ]
)


def _hf_image_cell(width: int, height: int) -> dict:
    return {
        "bytes": _image_bytes(_make_image(width, height)),
        "path": None,
    }


def _write_hf_arrow_shard(
    shard_path: str,
    rows: list,
    column_name: str = "image",
    multi_image: bool = False,
    batch_size: int | None = None,
    text_rows: list | None = None,
    text_column: str = "text",
):
    if multi_image:
        array = pa.array(
            [[_hf_image_cell(width, height) for width, height in sample] for sample in rows],
            type=pa.list_(_HF_IMAGE_TYPE),
        )
    else:
        array = pa.array(
            [_hf_image_cell(width, height) for width, height in rows],
            type=_HF_IMAGE_TYPE,
        )

    columns = {column_name: array}
    if text_rows is not None:
        columns[text_column] = pa.array(text_rows)
    table = pa.table(columns)
    with pa.OSFile(shard_path, "wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            for batch in table.to_batches(max_chunksize=batch_size):
                writer.write_batch(batch)


def _write_hf_parquet_shard(
    shard_path: str,
    rows: list,
    column_name: str = "image",
    multi_image: bool = False,
    row_group_size: int | None = None,
    text_rows: list | None = None,
    text_column: str = "text",
):
    if multi_image:
        array = pa.array(
            [[_hf_image_cell(width, height) for width, height in sample] for sample in rows],
            type=pa.list_(_HF_IMAGE_TYPE),
        )
    else:
        array = pa.array(
            [_hf_image_cell(width, height) for width, height in rows],
            type=_HF_IMAGE_TYPE,
        )

    columns = {column_name: array}
    if text_rows is not None:
        columns[text_column] = pa.array(text_rows)
    pq.write_table(pa.table(columns), shard_path, row_group_size=row_group_size)


# ======================================================================
# TestWDSScanner
# ======================================================================
class TestWDSScanner:

    def test_scan_single_tar(self, tmp_path):
        """Scan a tar with 5 images, verify correct dims and offsets."""
        samples = [{"key": f"{i:06d}", "ext": "jpg", "width": 100 + i * 10, "height": 200 + i * 10} for i in range(5)]
        tar_path = str(tmp_path / "shard_000.tar")
        _create_tar(tar_path, samples)

        records = scan_single_tar(tar_path)
        assert len(records) == 5

        for rec, s in zip(sorted(records, key=lambda r: r["sample_key"]), samples):
            assert rec["width"] == s["width"]
            assert rec["height"] == s["height"]
            assert rec["tar_path"] == tar_path
            assert rec["image_ext"] == "jpg"
            assert rec["offset_data"] > 0
            assert rec["file_size"] > 0

    def test_scan_ignores_non_images(self, tmp_path):
        """Only image files are returned; .txt sidecars are excluded."""
        samples = [
            {"key": "000001", "ext": "jpg", "width": 64, "height": 64, "text": "hello"},
        ]
        tar_path = str(tmp_path / "shard.tar")
        _create_tar(tar_path, samples)

        records = scan_single_tar(tar_path)
        assert len(records) == 1
        assert records[0]["image_ext"] == "jpg"

    def test_scan_multiple_tars(self, tmp_path):
        """Parallel scan of 3 tars should find 15 total images."""
        for shard_idx in range(3):
            samples = [{"key": f"{shard_idx:03d}_{i:03d}", "ext": "jpg", "width": 80, "height": 80} for i in range(5)]
            _create_tar(str(tmp_path / f"shard_{shard_idx:03d}.tar"), samples)

        manifest_path = str(tmp_path / "manifest.parquet")
        scan_wds_dataset(
            input_pattern=str(tmp_path / "shard_*.tar"),
            output_manifest=manifest_path,
            num_workers=2,
        )

        table = load_wds_manifest(manifest_path)
        assert len(table) == 15

    def test_manifest_parquet_schema(self, tmp_path):
        """Verify all expected columns are present in the manifest."""
        samples = [{"key": "000001", "ext": "jpg", "width": 32, "height": 32}]
        tar_path = str(tmp_path / "shard.tar")
        _create_tar(tar_path, samples)

        manifest_path = str(tmp_path / "manifest.parquet")
        scan_wds_dataset(
            input_pattern=tar_path,
            output_manifest=manifest_path,
            num_workers=1,
        )

        schema = pq.read_schema(manifest_path)
        expected = {"sample_key", "tar_path", "offset_data", "file_size", "width", "height", "image_ext"}
        assert set(schema.names) == expected

    def test_image_field_pattern_without_multi_image_keeps_single_image_manifest(self, tmp_path):
        """image_field_pattern should normalize sample keys without forcing grouped output."""
        samples = [
            {"key": "000001.img1", "ext": "jpg", "width": 32, "height": 32, "text": "caption"},
        ]
        tar_path = str(tmp_path / "shard.tar")
        _create_tar(tar_path, samples)

        manifest_path = str(tmp_path / "manifest.parquet")
        scan_wds_dataset(
            input_pattern=tar_path,
            output_manifest=manifest_path,
            text_extensions={"txt"},
            image_field_pattern="img",
            num_workers=1,
        )

        table = load_wds_manifest(manifest_path)
        schema = pq.read_schema(manifest_path)
        assert "group_id" not in schema.names
        assert "image_index" not in schema.names
        assert table.column("sample_key")[0].as_py() == "000001"
        assert table.column("offset_text")[0].as_py() >= 0

    def test_image_field_pattern_with_multi_image_writes_grouped_manifest(self, tmp_path):
        """Grouped output should require explicit multi_image=True."""
        samples = [
            {"key": "000001.img0", "ext": "jpg", "width": 32, "height": 32, "text": "caption"},
            {"key": "000001.img1", "ext": "jpg", "width": 48, "height": 48},
            {"key": "000002.img0", "ext": "jpg", "width": 64, "height": 64, "text": "other"},
        ]
        tar_path = str(tmp_path / "shard.tar")
        _create_tar(tar_path, samples)

        manifest_path = str(tmp_path / "manifest.parquet")
        scan_wds_dataset(
            input_pattern=tar_path,
            output_manifest=manifest_path,
            text_extensions={"txt"},
            image_field_pattern="img",
            multi_image=True,
            num_workers=1,
        )

        table = load_wds_manifest(manifest_path)
        schema = pq.read_schema(manifest_path)
        assert "group_id" in schema.names
        assert "image_index" in schema.names

        sample_keys = table.column("sample_key").to_pylist()
        group_ids = table.column("group_id").to_pylist()
        image_indices = table.column("image_index").to_pylist()
        assert sample_keys == ["000001", "000001", "000002"]
        assert image_indices == [0, 1, 0]
        assert group_ids[0] == group_ids[1]
        assert group_ids[2] != group_ids[0]

    def test_multi_image_scan_preserves_tar_order_and_offsets_group_ids(self, tmp_path):
        """Parallel grouped scans should emit rows in tar order with global group ids."""
        samples_a = [
            {"key": "000001.img0", "ext": "jpg", "width": 32, "height": 32, "text": "caption"},
            {"key": "000001.img1", "ext": "jpg", "width": 48, "height": 48},
            {"key": "000002.img0", "ext": "jpg", "width": 64, "height": 64},
        ]
        samples_b = [
            {"key": "000010.img0", "ext": "jpg", "width": 80, "height": 80, "text": "other"},
            {"key": "000010.img1", "ext": "jpg", "width": 96, "height": 96},
        ]
        _create_tar(str(tmp_path / "shard_000.tar"), samples_a)
        _create_tar(str(tmp_path / "shard_001.tar"), samples_b)

        manifest_path = str(tmp_path / "manifest.parquet")
        scan_wds_dataset(
            input_pattern=str(tmp_path / "shard_*.tar"),
            output_manifest=manifest_path,
            text_extensions={"txt"},
            image_field_pattern="img",
            multi_image=True,
            num_workers=2,
        )

        table = load_wds_manifest(manifest_path)
        assert table.column("sample_key").to_pylist() == [
            "000001",
            "000001",
            "000002",
            "000010",
            "000010",
        ]
        assert table.column("group_id").to_pylist() == [0, 0, 1, 2, 2]
        assert table.column("image_index").to_pylist() == [0, 1, 0, 0, 1]

    def test_single_image_validation_rejects_multiple_images_per_sample(self, tmp_path):
        """multi_image=False should fail if normalized sample keys have >1 image."""
        samples = [
            {"key": "000001.img0", "ext": "jpg", "width": 32, "height": 32, "text": "caption"},
            {"key": "000001.img1", "ext": "jpg", "width": 48, "height": 48},
        ]
        tar_path = str(tmp_path / "shard.tar")
        _create_tar(tar_path, samples)

        manifest_path = str(tmp_path / "manifest.parquet")
        with pytest.raises(ValueError, match="multiple images"):
            scan_wds_dataset(
                input_pattern=tar_path,
                output_manifest=manifest_path,
                text_extensions={"txt"},
                image_field_pattern="img",
                multi_image=False,
                num_workers=1,
            )

    def test_multi_image_validation_warns_on_singleton_groups(self, tmp_path, caplog):
        """multi_image=True should warn when all parsed groups have size 1."""
        samples = [
            {"key": "000001.img0", "ext": "jpg", "width": 32, "height": 32, "text": "caption"},
            {"key": "000002.img0", "ext": "jpg", "width": 48, "height": 48, "text": "other"},
        ]
        tar_path = str(tmp_path / "shard.tar")
        _create_tar(tar_path, samples)

        manifest_path = str(tmp_path / "manifest.parquet")
        with caplog.at_level("WARNING"):
            scan_wds_dataset(
                input_pattern=tar_path,
                output_manifest=manifest_path,
                text_extensions={"txt"},
                image_field_pattern="img",
                multi_image=True,
                num_workers=1,
            )

        assert "only singleton groups" in caplog.text

    def test_wds_key_format_matches_by_tar_and_path(self, tmp_path):
        from vision_tokenization.utils.contamination import load_contamination_index

        ids_path = tmp_path / "ids.txt"
        ids_path.write_text(
            "shard_000:000001  shard_000.tar:000003\n"  # .tar tolerated, dropped on parse
            "subA/0000:aaa subB/0000:bbb\n"  # path-qualified disambiguation of stem "0000"
        )
        index = load_contamination_index(ids_path, format="wds_key")
        assert index.rows_for_path("/x/shard_000.tar") == frozenset({"000001", "000003"})
        assert index.rows_for_path("/d/subA/0000.tar") == frozenset({"aaa"})
        assert index.rows_for_path("/d/subB/0000.tar") == frozenset({"bbb"})
        assert index.rows_for_path("/d/subC/0000.tar") == frozenset()

    def test_scan_wds_skips_excluded_samples_single_image(self, tmp_path):
        data = tmp_path / "data"
        data.mkdir()
        samples = [{"key": f"{i:06d}", "ext": "jpg", "width": 32, "height": 32} for i in range(5)]
        _create_tar(str(data / "shard_000.tar"), samples)
        ids_path = tmp_path / "ids.txt"
        ids_path.write_text("shard_000:000001, shard_000:000003\n")

        manifest_path = str(tmp_path / "manifest.parquet")
        scan_wds_dataset(
            input_pattern=str(data / "*.tar"),
            output_manifest=manifest_path,
            contamination_ids_path=ids_path,
            contamination_format="wds_key",
            num_workers=1,
        )

        table = load_wds_manifest(manifest_path)
        assert table.column("sample_key").to_pylist() == ["000000", "000002", "000004"]
        meta = json_load(Path(manifest_path).with_name("manifest_meta.json"))
        assert meta["contaminated_skipped"] == 2
        assert meta["contamination_format"] == "wds_key"
        assert meta["contamination_ids"] == 2

    def test_scan_wds_skips_excluded_samples_multi_image(self, tmp_path):
        data = tmp_path / "data"
        data.mkdir()
        samples = [
            {"key": "000001.img0", "ext": "jpg", "width": 32, "height": 32},
            {"key": "000001.img1", "ext": "jpg", "width": 32, "height": 32},
            {"key": "000002.img0", "ext": "jpg", "width": 32, "height": 32},
            {"key": "000002.img1", "ext": "jpg", "width": 32, "height": 32},
            {"key": "000003.img0", "ext": "jpg", "width": 32, "height": 32},
            {"key": "000003.img1", "ext": "jpg", "width": 32, "height": 32},
        ]
        _create_tar(str(data / "shard_000.tar"), samples)
        ids_path = tmp_path / "ids.txt"
        # Drop the middle group, which is itself multi-image: exercises both re-densification
        # (000001->0, 000003->1 with a gap) and per-sample skip counting (2 rows, 1 sample).
        ids_path.write_text("shard_000:000002\n")

        manifest_path = str(tmp_path / "manifest.parquet")
        scan_wds_dataset(
            input_pattern=str(data / "*.tar"),
            output_manifest=manifest_path,
            image_field_pattern="img",
            multi_image=True,
            contamination_ids_path=ids_path,
            contamination_format="wds_key",
            num_workers=1,
        )

        table = load_wds_manifest(manifest_path)
        assert table.column("sample_key").to_pylist() == ["000001", "000001", "000003", "000003"]
        # group_ids must be re-densified contiguous (000001->0, 000003->1), not [0,0,2,2]
        assert table.column("group_id").to_pylist() == [0, 0, 1, 1]
        assert table.column("image_index").to_pylist() == [0, 1, 0, 1]
        meta = json_load(Path(manifest_path).with_name("manifest_meta.json"))
        assert meta["contaminated_skipped"] == 1

    def test_scan_wds_tar_qualified_disambiguation(self, tmp_path):
        data = tmp_path / "data"
        (data / "subA").mkdir(parents=True)
        (data / "subB").mkdir(parents=True)
        _create_tar(
            str(data / "subA" / "0000.tar"),
            [{"key": k, "ext": "jpg", "width": 32, "height": 32} for k in ("000001", "000002")],
        )
        _create_tar(
            str(data / "subB" / "0000.tar"),
            [{"key": k, "ext": "jpg", "width": 32, "height": 32} for k in ("000001", "000009")],
        )
        ids_path = tmp_path / "ids.txt"
        ids_path.write_text("subA/0000:000001\n")  # only subA's 000001

        manifest_path = str(tmp_path / "manifest.parquet")
        scan_wds_dataset(
            input_pattern=str(data / "**" / "*.tar"),
            output_manifest=manifest_path,
            contamination_ids_path=ids_path,
            contamination_format="wds_key",
            num_workers=1,
        )

        table = load_wds_manifest(manifest_path)
        rows = list(zip(table.column("tar_path").to_pylist(), table.column("sample_key").to_pylist()))
        assert not any(sk == "000001" and tp.endswith("subA/0000.tar") for tp, sk in rows)
        assert any(sk == "000001" and tp.endswith("subB/0000.tar") for tp, sk in rows)

    def test_scan_wds_ambiguous_key_raises(self, tmp_path):
        data = tmp_path / "data"
        (data / "subA").mkdir(parents=True)
        (data / "subB").mkdir(parents=True)
        _create_tar(str(data / "subA" / "0000.tar"), [{"key": "000001", "ext": "jpg", "width": 32, "height": 32}])
        _create_tar(str(data / "subB" / "0000.tar"), [{"key": "000001", "ext": "jpg", "width": 32, "height": 32}])
        ids_path = tmp_path / "ids.txt"
        ids_path.write_text("0000:000001\n")  # bare tar stem matches both subsets

        with pytest.raises(ValueError, match="multiple shards"):
            scan_wds_dataset(
                input_pattern=str(data / "**" / "*.tar"),
                output_manifest=str(tmp_path / "manifest.parquet"),
                contamination_ids_path=ids_path,
                contamination_format="wds_key",
                num_workers=1,
            )

    def test_decontaminate_manifest_filters_wds_manifest(self, tmp_path):
        from scripts.decontaminate_manifest import decontaminate_manifest

        data = tmp_path / "data"
        data.mkdir()
        _create_tar(
            str(data / "a.tar"),
            [{"key": f"{i:06d}", "ext": "jpg", "width": 32, "height": 32} for i in (1, 2, 3)],
        )
        _create_tar(
            str(data / "b.tar"),
            [{"key": f"{i:06d}", "ext": "jpg", "width": 32, "height": 32} for i in (1, 4)],
        )
        raw_manifest = tmp_path / "raw.parquet"
        scan_wds_dataset(input_pattern=str(data / "*.tar"), output_manifest=str(raw_manifest), num_workers=1)

        ids_path = tmp_path / "ids.txt"
        ids_path.write_text("a:000002\n")
        summary = decontaminate_manifest(
            raw_manifest,
            tmp_path / "clean.parquet",
            ids_path,
            contamination_format="wds_key",
        )
        assert summary["manifest_kind"] == "wds"
        assert summary["rows_in"] == 5
        assert summary["rows_out"] == 4
        assert summary["manifest_rows_dropped"] == 1
        assert summary["source_docs_dropped"] == 1


# ======================================================================
# TestHFScanner
# ======================================================================
class TestHFScanner:

    def test_hf_batch_scan_skips_contamination_lookup_when_empty(self):
        class EmptyRows:
            def __bool__(self):
                return False

            def __contains__(self, _item):
                raise AssertionError("empty contamination index should not be checked")

        out = build_hf_output_columns(is_multi=False)
        image_col = pa.array(
            [_hf_image_cell(10, 20), _hf_image_cell(30, 40)],
            type=_HF_IMAGE_TYPE,
        )

        out, source_rows, failed_dims, contaminated_skipped = scan_hf_batch_columns(
            out,
            image_col,
            chunk_index=0,
            source_rows=0,
            failed_dims=0,
            is_multi=False,
            contaminated_rows=EmptyRows(),
        )

        assert source_rows == 2
        assert failed_dims == 0
        assert contaminated_skipped == 0

    def test_scan_hf_arrow_single_image(self, tmp_path):
        rows_a = [(32, 48), (64, 96)]
        rows_b = [(20, 30)]
        _write_hf_arrow_shard(str(tmp_path / "part_000.arrow"), rows_a)
        _write_hf_arrow_shard(str(tmp_path / "part_001.arrow"), rows_b)

        manifest_path = str(tmp_path / "manifest.parquet")
        scan_hf_dataset(
            input_pattern=str(tmp_path / "*.arrow"),
            output_manifest=manifest_path,
            num_workers=2,
        )

        table = load_hf_manifest(manifest_path)
        assert table.column("sample_index").to_pylist() == [0, 1, 2]
        assert table.column("width").to_pylist() == [32, 64, 20]
        assert table.column("height").to_pylist() == [48, 96, 30]
        assert table.column("chunk_index").to_pylist() == [0, 0, 0]
        assert table.column("row_in_chunk").to_pylist() == [0, 1, 0]
        assert table.column("shard_path").to_pylist() == [
            str(tmp_path / "part_000.arrow"),
            str(tmp_path / "part_000.arrow"),
            str(tmp_path / "part_001.arrow"),
        ]

    def test_scan_hf_parquet_single_image(self, tmp_path):
        rows_a = [(80, 40), (120, 60)]
        rows_b = [(25, 35)]
        _write_hf_parquet_shard(str(tmp_path / "part_000.parquet"), rows_a)
        _write_hf_parquet_shard(str(tmp_path / "part_001.parquet"), rows_b)

        manifest_path = str(tmp_path / "manifest.parquet")
        scan_hf_dataset(
            input_pattern=str(tmp_path / "*.parquet"),
            output_manifest=manifest_path,
            num_workers=2,
        )

        table = load_hf_manifest(manifest_path)
        assert table.column("sample_index").to_pylist() == [0, 1, 2]
        assert table.column("width").to_pylist() == [80, 120, 25]
        assert table.column("height").to_pylist() == [40, 60, 35]
        assert table.column("chunk_index").to_pylist() == [0, 0, 0]
        assert table.column("row_in_chunk").to_pylist() == [0, 1, 0]
        assert table.column("shard_path").to_pylist() == [
            str(tmp_path / "part_000.parquet"),
            str(tmp_path / "part_000.parquet"),
            str(tmp_path / "part_001.parquet"),
        ]

    def test_scan_hf_parquet_skips_shards_missing_image_column(self, tmp_path, caplog):
        (tmp_path / "good").mkdir()
        (tmp_path / "bad").mkdir()
        _write_hf_parquet_shard(
            str(tmp_path / "good" / "part_000.parquet"),
            [(80, 40), (120, 60)],
        )
        pq.write_table(
            pa.table({"caption": pa.array(["a", "b"], type=pa.string())}),
            str(tmp_path / "bad" / "part_001.parquet"),
        )

        manifest_path = str(tmp_path / "manifest.parquet")
        with caplog.at_level("WARNING"):
            scan_hf_dataset(
                input_pattern=str(tmp_path),
                output_manifest=manifest_path,
                num_workers=2,
            )

        table = load_hf_manifest(manifest_path)
        assert table.column("sample_index").to_pylist() == [0, 1]
        assert table.column("width").to_pylist() == [80, 120]
        assert table.column("height").to_pylist() == [40, 60]
        assert "Skipping HF shard" in caplog.text
        assert "missing column 'image'" in caplog.text

    def test_scan_hf_arrow_multi_image(self, tmp_path):
        rows_a = [
            [(10, 20), (30, 40)],
            [(50, 60)],
        ]
        rows_b = [
            [(70, 80), (90, 100)],
        ]
        _write_hf_arrow_shard(
            str(tmp_path / "part_000.arrow"),
            rows_a,
            column_name="images",
            multi_image=True,
        )
        _write_hf_arrow_shard(
            str(tmp_path / "part_001.arrow"),
            rows_b,
            column_name="images",
            multi_image=True,
        )

        manifest_path = str(tmp_path / "manifest.parquet")
        scan_hf_dataset(
            input_pattern=str(tmp_path / "*.arrow"),
            output_manifest=manifest_path,
            image_list_column="images",
            num_workers=2,
        )

        table = load_hf_manifest(manifest_path)
        assert table.column("sample_index").to_pylist() == [0, 0, 1, 2, 2]
        assert table.column("group_id").to_pylist() == [0, 0, 1, 2, 2]
        assert table.column("image_index").to_pylist() == [0, 1, 0, 0, 1]
        assert table.column("width").to_pylist() == [10, 30, 50, 70, 90]
        assert table.column("height").to_pylist() == [20, 40, 60, 80, 100]
        assert table.column("chunk_index").to_pylist() == [0, 0, 0, 0, 0]
        assert table.column("row_in_chunk").to_pylist() == [0, 0, 1, 0, 0]

    def test_scan_hf_parquet_multi_image(self, tmp_path):
        rows_a = [
            [(11, 21), (31, 41)],
            [(51, 61)],
        ]
        rows_b = [
            [(71, 81), (91, 101)],
        ]
        _write_hf_parquet_shard(
            str(tmp_path / "part_000.parquet"),
            rows_a,
            column_name="images",
            multi_image=True,
        )
        _write_hf_parquet_shard(
            str(tmp_path / "part_001.parquet"),
            rows_b,
            column_name="images",
            multi_image=True,
        )

        manifest_path = str(tmp_path / "manifest.parquet")
        scan_hf_dataset(
            input_pattern=str(tmp_path / "*.parquet"),
            output_manifest=manifest_path,
            image_list_column="images",
            num_workers=2,
        )

        table = load_hf_manifest(manifest_path)
        assert table.column("sample_index").to_pylist() == [0, 0, 1, 2, 2]
        assert table.column("group_id").to_pylist() == [0, 0, 1, 2, 2]
        assert table.column("image_index").to_pylist() == [0, 1, 0, 0, 1]
        assert table.column("width").to_pylist() == [11, 31, 51, 71, 91]
        assert table.column("height").to_pylist() == [21, 41, 61, 81, 101]
        assert table.column("chunk_index").to_pylist() == [0, 0, 0, 0, 0]
        assert table.column("row_in_chunk").to_pylist() == [0, 0, 1, 0, 0]

    def test_scan_hf_parquet_skips_contaminated_source_rows(self, tmp_path):
        rows = [
            [(10, 20)],
            [(30, 40), (31, 41)],
            [(50, 60)],
            [(70, 80), (71, 81)],
            [(90, 100)],
        ]
        _write_hf_parquet_shard(
            str(tmp_path / "SFT_000099.parquet"),
            rows,
            column_name="images",
            multi_image=True,
            row_group_size=2,
        )
        ids_path = tmp_path / "contaminated.txt"
        ids_path.write_text("SFT_000099_000001,SFT_000099_000003\n")

        manifest_path = str(tmp_path / "manifest.parquet")
        scan_hf_dataset(
            input_pattern=str(tmp_path / "SFT_*.parquet"),
            output_manifest=manifest_path,
            image_list_column="images",
            contamination_ids_path=ids_path,
            contamination_format="innovator_vl",
            num_workers=1,
        )

        table = load_hf_manifest(manifest_path)
        assert table.column("sample_index").to_pylist() == [0, 2, 4]
        assert table.column("group_id").to_pylist() == [0, 2, 4]
        assert table.column("chunk_index").to_pylist() == [0, 1, 2]
        assert table.column("row_in_chunk").to_pylist() == [0, 0, 0]

        meta = json_load(Path(manifest_path).with_name("manifest_meta.json"))
        assert meta["contaminated_skipped"] == 2
        assert meta["contamination_format"] == "innovator_vl"
        assert meta["contamination_ids"] == 2

    def test_decontaminate_manifest_filters_existing_hf_manifest(self, tmp_path):
        from scripts.decontaminate_manifest import decontaminate_manifest

        rows = [
            [(10, 20)],
            [(30, 40), (31, 41)],
            [(50, 60)],
            [(70, 80), (71, 81)],
            [(90, 100)],
        ]
        _write_hf_parquet_shard(
            str(tmp_path / "SFT_000100.parquet"),
            rows,
            column_name="images",
            multi_image=True,
            row_group_size=2,
        )
        ids_path = tmp_path / "contaminated.txt"
        ids_path.write_text("SFT_000100_000001 SFT_000100_000003\n")

        raw_manifest = tmp_path / "raw_manifest.parquet"
        clean_manifest = tmp_path / "clean_manifest.parquet"
        scan_hf_dataset(
            input_pattern=str(tmp_path / "SFT_*.parquet"),
            output_manifest=raw_manifest,
            image_list_column="images",
            num_workers=1,
        )

        summary = decontaminate_manifest(
            raw_manifest,
            clean_manifest,
            ids_path,
            contamination_format="innovator_vl",
        )

        table = load_hf_manifest(clean_manifest)
        assert table.column("sample_index").to_pylist() == [0, 2, 4]
        assert summary["rows_in"] == 7
        assert summary["rows_out"] == 3
        assert summary["manifest_rows_dropped"] == 4
        assert summary["source_docs_dropped"] == 2
        meta = json_load(clean_manifest.with_name("clean_manifest_decontamination_meta.json"))
        assert meta["source_docs_dropped"] == 2

    def test_rebuild_guardrail_requires_matching_plan_document_count(self, tmp_path):
        # rebuild_decontaminated maps contaminated source rows -> plan doc_ids via
        # doc_id == position in np.unique(group_ids). That positional mapping is only
        # valid when the plan's document count equals the manifest's unique-group count.
        # If a Stage-2 plan filter (e.g. a --min-pixels mismatch vs the original run)
        # dropped a group, the positions shift and it must abort instead of dropping the
        # wrong docs. A fake plan lets us test the guardrail without a real plan/tokenizer.
        from types import SimpleNamespace

        from scripts.rebuild_decontaminated import compute_rejected_doc_ids

        _write_hf_parquet_shard(
            str(tmp_path / "part_000.parquet"),
            [(10, 20), (30, 40), (50, 60), (70, 80), (90, 100)],
            column_name="image",
            row_group_size=2,  # chunk_offsets == [0, 2, 4]
        )
        ids_path = tmp_path / "ids.txt"
        ids_path.write_text("part_000:2\n")  # source row 2 -> chunk 1, row_in_chunk 0

        manifest_path = str(tmp_path / "manifest.parquet")
        scan_hf_dataset(
            input_pattern=str(tmp_path / "part_*.parquet"),
            output_manifest=manifest_path,
            num_workers=1,
        )

        # Matching plan: 5 single-image docs == manifest's 5 unique sample_index -> maps
        # the contaminated source row (sample_index 2) to plan doc_id 2.
        rejected = compute_rejected_doc_ids(
            Path(manifest_path), ids_path, "stem_row", SimpleNamespace(total_documents=5)
        )
        assert rejected == {2}

        # Mismatched plan (a filter dropped one group): positions no longer align -> abort.
        with pytest.raises(SystemExit):
            compute_rejected_doc_ids(Path(manifest_path), ids_path, "stem_row", SimpleNamespace(total_documents=4))

    def test_stem_row_format_matches_by_stem_and_path(self, tmp_path):
        from vision_tokenization.utils.contamination import load_contamination_index

        ids_path = tmp_path / "ids.txt"
        ids_path.write_text(
            "train-00000-of-00003:5, train-00000-of-00003:9\n"
            "train-00002-of-00003.parquet:120\n"  # extension tolerated, dropped on parse
            "subsetA/0000:1  subsetB/0000:2\n"  # path-qualified to disambiguate same stem
        )
        index = load_contamination_index(ids_path, format="stem_row")

        # bare stem matches regardless of directory or extension
        assert index.rows_for_path("/d/train-00000-of-00003.parquet") == frozenset({5, 9})
        assert index.rows_for_path("/d/train-00002-of-00003.arrow") == frozenset({120})
        assert index.rows_for_path("/d/train-00001-of-00003.parquet") == frozenset()
        # colliding stem "0000" disambiguated by the subset path component
        assert index.rows_for_path("/cache/subsetA/0000.parquet") == frozenset({1})
        assert index.rows_for_path("/cache/subsetB/0000.parquet") == frozenset({2})
        assert index.rows_for_path("/cache/subsetC/0000.parquet") == frozenset()

    def test_scan_hf_parquet_stem_row_skips_rows_across_files(self, tmp_path):
        _write_hf_parquet_shard(
            str(tmp_path / "train-00000-of-00002.parquet"),
            [[(10, 20)], [(30, 40)], [(50, 60)]],
            column_name="images",
            multi_image=True,
            row_group_size=2,
        )
        _write_hf_parquet_shard(
            str(tmp_path / "train-00001-of-00002.parquet"),
            [[(11, 21)], [(31, 41)], [(51, 61)]],
            column_name="images",
            multi_image=True,
            row_group_size=2,
        )
        ids_path = tmp_path / "ids.txt"
        ids_path.write_text(
            "train-00000-of-00002:1\n"  # drop row 1 of file 0
            "train-00001-of-00002:0, train-00001-of-00002:2\n"  # drop rows 0,2 of file 1
        )

        manifest_path = str(tmp_path / "manifest.parquet")
        scan_hf_dataset(
            input_pattern=str(tmp_path / "*.parquet"),
            output_manifest=manifest_path,
            image_list_column="images",
            contamination_ids_path=ids_path,
            contamination_format="stem_row",
            num_workers=1,
        )

        table = load_hf_manifest(manifest_path)
        # file0 keeps source rows 0,2 -> sample_index 0,2; file1 (offset 3) keeps row 1 -> 4
        assert table.column("sample_index").to_pylist() == [0, 2, 4]
        meta = json_load(Path(manifest_path).with_name("manifest_meta.json"))
        assert meta["contaminated_skipped"] == 3
        assert meta["contamination_format"] == "stem_row"

    def test_scan_hf_ambiguous_stem_row_id_raises(self, tmp_path):
        data = tmp_path / "data"
        (data / "configA").mkdir(parents=True)
        (data / "configB").mkdir(parents=True)
        _write_hf_parquet_shard(
            str(data / "configA" / "0000.parquet"),
            [[(10, 20)], [(30, 40)]],
            column_name="images",
            multi_image=True,
        )
        _write_hf_parquet_shard(
            str(data / "configB" / "0000.parquet"),
            [[(11, 21)], [(31, 41)]],
            column_name="images",
            multi_image=True,
        )
        ids_path = tmp_path / "ids.txt"
        ids_path.write_text("0000:1\n")  # bare stem matches both configs -> ambiguous

        with pytest.raises(ValueError, match="multiple shards"):
            scan_hf_dataset(
                input_pattern=str(data),
                output_manifest=str(tmp_path / "manifest.parquet"),
                image_list_column="images",
                contamination_ids_path=ids_path,
                contamination_format="stem_row",
                num_workers=1,
            )

    def test_decontaminate_manifest_counts_colliding_stems_per_shard(self, tmp_path):
        from scripts.decontaminate_manifest import decontaminate_manifest

        data = tmp_path / "data"
        (data / "configA").mkdir(parents=True)
        (data / "configB").mkdir(parents=True)
        _write_hf_parquet_shard(
            str(data / "configA" / "0000.parquet"),
            [[(10, 20)], [(30, 40)], [(50, 60)]],
            column_name="images",
            multi_image=True,
        )
        _write_hf_parquet_shard(
            str(data / "configB" / "0000.parquet"),
            [[(11, 21)], [(31, 41)], [(51, 61)]],
            column_name="images",
            multi_image=True,
        )
        raw_manifest = tmp_path / "raw.parquet"
        scan_hf_dataset(
            input_pattern=str(data),
            output_manifest=str(raw_manifest),
            image_list_column="images",
            num_workers=1,
        )
        ids_path = tmp_path / "ids.txt"
        ids_path.write_text("configA/0000:1\nconfigB/0000:1\n")  # same row, different subset

        summary = decontaminate_manifest(
            raw_manifest,
            tmp_path / "clean.parquet",
            ids_path,
            contamination_format="stem_row",
        )
        assert summary["manifest_rows_dropped"] == 2
        assert summary["source_docs_dropped"] == 2  # per-shard, not deduped by shared stem
        assert summary["unmatched_contamination_ids"] == 0


# ======================================================================
# TestWDSRandomAccess
# ======================================================================
class TestWDSRandomAccess:

    def _scan_and_build_refs(self, tar_path: str):
        """Helper: scan a tar and return list of (tar_path, offset, size, original_img)."""
        samples_meta = [
            {
                "key": f"{i:06d}",
                "ext": "jpg",
                "width": 50 + i * 20,
                "height": 50 + i * 20,
                "color": (i * 40 % 256, 100, 200),
            }
            for i in range(5)
        ]
        _create_tar(tar_path, samples_meta)
        records = scan_single_tar(tar_path)
        records.sort(key=lambda r: r["sample_key"])

        refs = []
        for rec, meta in zip(records, samples_meta):
            original = _make_image(meta["width"], meta["height"], meta["color"])
            refs.append((rec["tar_path"], rec["offset_data"], rec["file_size"], original))
        return refs

    def test_read_single_image(self, tmp_path):
        """Read each image by offset, verify dimensions match."""
        tar_path = str(tmp_path / "shard.tar")
        refs = self._scan_and_build_refs(tar_path)

        with TarRandomAccessReader() as reader:
            for tp, offset, size, original in refs:
                img = reader.read_image(tp, offset, size)
                assert img.size == original.size

    def test_read_batch(self, tmp_path):
        """read_batch returns images in order, all non-None."""
        tar_path = str(tmp_path / "shard.tar")
        refs = self._scan_and_build_refs(tar_path)

        batch_refs = [(tp, off, sz) for tp, off, sz, _ in refs]
        with TarRandomAccessReader() as reader:
            images = reader.read_batch(batch_refs)
        assert len(images) == len(refs)
        assert all(img is not None for img in images)
        for img, (_, _, _, orig) in zip(images, refs):
            assert img.size == orig.size

    def test_read_batch_logs_truncated_image_cleanly(self, caplog, monkeypatch):
        """Common PIL corruption errors should log one-line warnings only."""
        reader = TarRandomAccessReader()

        def _broken_read_image(_tar_path, _offset, _size):
            raise OSError("image file is truncated")

        monkeypatch.setattr(reader, "read_image", _broken_read_image)

        with caplog.at_level(logging.WARNING):
            images = reader.read_batch([("broken.tar", 123, 456)])

        assert images == [None]
        assert len(caplog.records) == 1
        assert caplog.records[0].exc_info is None
        assert "image file is truncated" in caplog.text

    def test_file_handle_caching(self, tmp_path):
        """LRU cache: 1 handle for same tar, eviction when max reached."""
        # Create 3 tars
        tar_paths = []
        for i in range(3):
            tp = str(tmp_path / f"shard_{i}.tar")
            samples = [{"key": f"{i:03d}_000", "ext": "jpg", "width": 32, "height": 32}]
            _create_tar(tp, samples)
            tar_paths.append(tp)

        reader = TarRandomAccessReader(max_open_files=2)
        try:
            # Access tar 0 and tar 1 — both should be cached
            rec0 = scan_single_tar(tar_paths[0])[0]
            rec1 = scan_single_tar(tar_paths[1])[0]
            reader.read_image(tar_paths[0], rec0["offset_data"], rec0["file_size"])
            reader.read_image(tar_paths[1], rec1["offset_data"], rec1["file_size"])
            handles = reader._get_handles()
            assert len(handles) == 2

            # Access tar 2 — should evict tar 0 (oldest)
            rec2 = scan_single_tar(tar_paths[2])[0]
            reader.read_image(tar_paths[2], rec2["offset_data"], rec2["file_size"])
            handles = reader._get_handles()
            assert len(handles) == 2
            assert tar_paths[0] not in handles
            assert tar_paths[1] in handles
            assert tar_paths[2] in handles
        finally:
            reader.close()


# ======================================================================
# TestClusteredBatchPlanner
# ======================================================================
class TestClusteredBatchPlanner:

    @staticmethod
    def _create_manifest(tmp_path, widths, heights):
        """Write a minimal WDS manifest with given widths/heights."""
        records = [
            {
                "sample_key": f"{i:06d}",
                "tar_path": "dummy.tar",
                "offset_data": 0,
                "file_size": 1000,
                "width": int(w),
                "height": int(h),
                "image_ext": "jpg",
            }
            for i, (w, h) in enumerate(zip(widths, heights))
        ]
        path = str(tmp_path / "manifest.parquet")
        save_wds_manifest(records, path)
        return path

    def test_plan_produces_batches(self, tmp_path):
        """1000 samples with 3 resolution clusters -> non-empty plan."""
        rng = np.random.RandomState(42)
        # 3 clusters: landscape, square, portrait
        w = np.concatenate([rng.randint(400, 600, 334), rng.randint(200, 300, 333), rng.randint(100, 200, 333)])
        h = np.concatenate([rng.randint(200, 300, 334), rng.randint(200, 300, 333), rng.randint(400, 600, 333)])
        path = self._create_manifest(tmp_path, w, h)

        plan = plan_clustered_batches(
            path, batch_size=32, max_batch_tokens=999999, resize_min_pixels=64 * 64, resize_max_pixels=1024 * 1024
        )
        assert isinstance(plan, BatchPlan)
        assert len(plan.batches) > 0
        assert plan.total_samples == 1000

    def test_batch_size_respected(self, tmp_path):
        """No batch should exceed batch_size."""
        rng = np.random.RandomState(7)
        w = rng.randint(100, 800, 500)
        h = rng.randint(100, 800, 500)
        path = self._create_manifest(tmp_path, w, h)

        bs = 16
        plan = plan_clustered_batches(
            path, batch_size=bs, max_batch_tokens=999999, resize_min_pixels=64 * 64, resize_max_pixels=1024 * 1024
        )
        for batch in plan.batches:
            assert len(batch.sample_indices) <= bs

    def test_all_samples_assigned(self, tmp_path):
        """Every sample appears in exactly one batch."""
        rng = np.random.RandomState(99)
        N = 300
        w = rng.randint(100, 500, N)
        h = rng.randint(100, 500, N)
        path = self._create_manifest(tmp_path, w, h)

        plan = plan_clustered_batches(
            path, batch_size=20, max_batch_tokens=999999, resize_min_pixels=64 * 64, resize_max_pixels=1024 * 1024
        )
        all_indices = np.concatenate([b.sample_indices for b in plan.batches])
        assert len(all_indices) == N
        assert len(np.unique(all_indices)) == N

    def test_clustering_groups_similar_resolutions(self, tmp_path):
        """Within-batch aspect-ratio std should be much smaller than global std."""
        rng = np.random.RandomState(123)
        # Wide spread of aspect ratios
        w = rng.randint(100, 1000, 600)
        h = rng.randint(100, 1000, 600)
        path = self._create_manifest(tmp_path, w, h)

        plan = plan_clustered_batches(
            path, batch_size=32, max_batch_tokens=999999, resize_min_pixels=64 * 64, resize_max_pixels=1024 * 1024
        )

        global_ar = w.astype(np.float64) / h.astype(np.float64)
        global_std = np.std(global_ar)

        within_stds = []
        for batch in plan.batches:
            idx = batch.sample_indices
            bw = w[idx].astype(np.float64)
            bh = h[idx].astype(np.float64)
            ar = bw / bh
            if len(ar) > 1:
                within_stds.append(np.std(ar))

        mean_within_std = np.mean(within_stds)
        assert mean_within_std < global_std, (
            f"Mean within-batch AR std ({mean_within_std:.4f}) should be < " f"global AR std ({global_std:.4f})"
        )

    def test_resolution_filtering(self, tmp_path):
        """min_pixels should filter out small images."""
        # 50 tiny (10x10=100px) + 50 normal (200x200=40000px)
        w = np.array([10] * 50 + [200] * 50)
        h = np.array([10] * 50 + [200] * 50)
        path = self._create_manifest(tmp_path, w, h)

        plan = plan_clustered_batches(
            path,
            batch_size=10,
            max_batch_tokens=999999,
            min_pixels=1000,
            resize_min_pixels=64 * 64,
            resize_max_pixels=1024 * 1024,
        )
        assert plan.total_filtered == 50
        all_idx = np.concatenate([b.sample_indices for b in plan.batches])
        assert len(all_idx) == 50
        # All assigned indices should be from the "normal" images (index >= 50)
        assert all(i >= 50 for i in all_idx)

    def test_worker_split(self, tmp_path):
        """split_for_workers(4) produces 4 worker chunks covering all batches."""
        rng = np.random.RandomState(0)
        w = rng.randint(100, 500, 200)
        h = rng.randint(100, 500, 200)
        path = self._create_manifest(tmp_path, w, h)

        plan = plan_clustered_batches(
            path, batch_size=10, max_batch_tokens=999999, resize_min_pixels=64 * 64, resize_max_pixels=1024 * 1024
        )
        chunks = plan.split_for_workers(4)
        assert len(chunks) == 4
        # Flatten and verify all batches covered
        flat = [b for chunk in chunks for b in chunk]
        assert len(flat) == len(plan.batches)

    def test_worker_split_is_weighted_but_contiguous(self):
        """Weighted splitting should improve balance without reordering batches."""
        costs = [1, 1, 1, 1, 100, 1, 1, 100]
        batches = [
            BatchAssignment(
                sample_indices=np.array([idx], dtype=np.int64),
                resize_height=cost,
                resize_width=1,
                batch_token_count=cost,
            )
            for idx, cost in enumerate(costs)
        ]
        plan = BatchPlan(batches=batches)

        chunks = plan.split_for_workers(2)

        assert [int(batch.sample_indices[0]) for batch in chunks[0]] == [0, 1, 2, 3, 4]
        assert [int(batch.sample_indices[0]) for batch in chunks[1]] == [5, 6, 7]

        flat = [int(batch.sample_indices[0]) for chunk in chunks for batch in chunk]
        assert flat == list(range(len(batches)))

        weighted_costs = [sum(plan._estimate_batch_cost(batch) for batch in chunk) for chunk in chunks]
        count_split_costs = [
            sum(costs[:4]),
            sum(costs[4:]),
        ]
        assert max(weighted_costs) < max(count_split_costs)

    def test_multi_image_requires_group_column(self, tmp_path):
        """multi_image=True on a single-image manifest should raise ValueError."""
        w = np.array([200] * 10)
        h = np.array([200] * 10)
        path = self._create_manifest(tmp_path, w, h)

        with pytest.raises(ValueError, match="multi_image=True but manifest has no group_id"):
            plan_clustered_batches(path, batch_size=4, max_batch_tokens=999999, multi_image=True)

    def test_planner_uses_smart_resize_budget(self, tmp_path):
        """Large page images should pack by post-smart-resize token counts."""
        widths = np.array([2560] * 4)
        heights = np.array([1440] * 4)
        path = self._create_manifest(tmp_path, widths, heights)

        plan = plan_clustered_batches(
            path,
            batch_size=8,
            max_batch_tokens=25600,
            resize_min_pixels=128 * 128,
            resize_max_pixels=1400 * 1400,
        )

        assert [len(batch.sample_indices) for batch in plan.batches] == [3, 1]

        expected_height, expected_width = smart_resize_dims(
            1440,
            2560,
            min_pixels=128 * 128,
            max_pixels=1400 * 1400,
            factor=16,
        )
        assert plan.batches[0].resize_height == expected_height
        assert plan.batches[0].resize_width == expected_width

        per_image_tokens = estimate_image_tokens(
            expected_height,
            expected_width,
            spatial_factor=16,
        )
        assert per_image_tokens * 3 <= 25600
        assert per_image_tokens * 4 > 25600

        dry_run = dry_run_batch_plan(plan, spatial_factor=16)
        assert dry_run["total_batches"] == 2
        assert dry_run["max_tokens_per_batch"] == per_image_tokens * 3


# ======================================================================
# TestEndToEnd
# ======================================================================
class TestEndToEnd:

    def test_scan_cluster_read_verify(self, tmp_path):
        """Full pipeline: create tars -> scan -> plan -> read -> verify pixels."""
        # Create 2 tars with 15 images each across 3 resolution clusters
        all_originals = {}  # key -> (PIL.Image, tar_path)
        for shard_idx in range(2):
            samples = []
            for i in range(15):
                cluster = i % 3
                if cluster == 0:
                    w, h = 320, 240  # landscape
                elif cluster == 1:
                    w, h = 200, 200  # square
                else:
                    w, h = 150, 400  # portrait
                # Unique colour per image
                color = ((shard_idx * 15 + i) * 17 % 256, 100, 50)
                key = f"{shard_idx:03d}_{i:03d}"
                samples.append({"key": key, "ext": "jpg", "width": w, "height": h, "color": color})
                all_originals[key] = (_make_image(w, h, color), None)  # tar_path filled later

            tar_path = str(tmp_path / f"shard_{shard_idx:03d}.tar")
            _create_tar(tar_path, samples)

        # --- Scan ---
        manifest_path = str(tmp_path / "manifest.parquet")
        scan_wds_dataset(
            input_pattern=str(tmp_path / "shard_*.tar"),
            output_manifest=manifest_path,
            num_workers=2,
        )

        table = load_wds_manifest(manifest_path)
        assert len(table) == 30

        # --- Plan batches ---
        plan = plan_clustered_batches(
            manifest_path,
            batch_size=8,
            max_batch_tokens=999999,
            resize_min_pixels=64 * 64,
            resize_max_pixels=1024 * 1024,
        )
        assert plan.total_samples == 30
        all_idx = np.concatenate([b.sample_indices for b in plan.batches])
        assert len(np.unique(all_idx)) == 30

        # --- Random-access read ---
        tar_paths_col = table.column("tar_path").to_pylist()
        offsets_col = table.column("offset_data").to_pylist()
        sizes_col = table.column("file_size").to_pylist()
        widths_col = table.column("width").to_pylist()
        heights_col = table.column("height").to_pylist()

        with TarRandomAccessReader() as reader:
            for batch in plan.batches:
                refs = [(tar_paths_col[i], offsets_col[i], sizes_col[i]) for i in batch.sample_indices]
                images = reader.read_batch(refs)
                for img, idx in zip(images, batch.sample_indices):
                    assert img is not None
                    assert img.size == (widths_col[idx], heights_col[idx])


# ======================================================================
# TestWDSLoader
# ======================================================================
class TestWDSLoader:

    def test_load_text_batch_reads_text_without_loading_images(self, tmp_path, monkeypatch):
        samples = [
            {"key": "000001", "ext": "jpg", "width": 64, "height": 64, "text": "alpha"},
            {"key": "000002", "ext": "jpg", "width": 32, "height": 48, "text": "beta"},
        ]
        tar_path = str(tmp_path / "shard.tar")
        _create_tar(tar_path, samples)

        manifest_path = str(tmp_path / "manifest.parquet")
        scan_wds_dataset(
            input_pattern=tar_path,
            output_manifest=manifest_path,
            num_workers=1,
            text_extensions=frozenset({"txt"}),
        )

        loader = WDSImageLoader(manifest_path=manifest_path, text_field="text")
        monkeypatch.setattr(
            loader._reader,
            "read_batch",
            lambda refs: pytest.fail("load_text_batch should not load images"),
        )

        texts = loader.load_text_batch(np.array([0, 1], dtype=np.int64))
        loader.close()

        assert texts == ["alpha", "beta"]


# ======================================================================
# TestHFLoader
# ======================================================================
class TestHFLoader:

    def test_parquet_loader_excludes_manifest_and_reads_row_groups(self, tmp_path):
        rows_a = [(32, 48), (64, 96), (80, 120)]
        rows_b = [(20, 30)]
        _write_hf_parquet_shard(
            str(tmp_path / "part_000.parquet"),
            rows_a,
            row_group_size=1,
        )
        _write_hf_parquet_shard(
            str(tmp_path / "part_001.parquet"),
            rows_b,
            row_group_size=1,
        )
        pq.write_table(
            pa.table(
                {
                    "sample_index": pa.array([0], type=pa.int64()),
                    "width": pa.array([1], type=pa.int32()),
                    "height": pa.array([1], type=pa.int32()),
                }
            ),
            str(tmp_path / "manifest.parquet"),
        )

        loader = HFImageLoader(input_pattern=tmp_path)
        images, _ = loader.load_batch(np.array([0, 2, 3], dtype=np.int64))
        loader.close()

        assert loader._total_rows == 4
        assert [img.size for img in images] == [(32, 48), (80, 120), (20, 30)]

    def test_arrow_loader_reads_across_record_batches(self, tmp_path):
        rows = [(10, 20), (30, 40), (50, 60)]
        _write_hf_arrow_shard(
            str(tmp_path / "part_000.arrow"),
            rows,
            batch_size=1,
        )

        loader = HFImageLoader(input_pattern=str(tmp_path / "*.arrow"))
        images, _ = loader.load_batch(np.array([0, 2], dtype=np.int64))
        loader.close()

        assert [img.size for img in images] == [(10, 20), (50, 60)]

    def test_parquet_loader_uses_physical_manifest_coordinates(self, tmp_path):
        rows_a = [(32, 48), (64, 96), (80, 120)]
        rows_b = [(20, 30)]
        _write_hf_parquet_shard(
            str(tmp_path / "part_000.parquet"),
            rows_a,
            row_group_size=1,
        )
        _write_hf_parquet_shard(
            str(tmp_path / "part_001.parquet"),
            rows_b,
            row_group_size=1,
        )

        manifest_path = str(tmp_path / "physical_manifest.parquet")
        scan_hf_dataset(
            input_pattern=str(tmp_path / "*.parquet"),
            output_manifest=manifest_path,
            num_workers=2,
        )

        loader = HFImageLoader(
            input_pattern=str(tmp_path / "*.does_not_matter"),
            manifest_path=manifest_path,
        )
        images, _ = loader.load_batch(np.array([0, 2, 3], dtype=np.int64))
        loader.close()

        assert loader._uses_physical_manifest is True
        assert [img.size for img in images] == [(32, 48), (80, 120), (20, 30)]

    def test_parquet_loader_load_text_batch_avoids_image_decode(self, tmp_path, monkeypatch):
        rows = [(32, 48), (64, 96), (80, 120)]
        texts = ["alpha", "beta", "gamma"]
        _write_hf_parquet_shard(
            str(tmp_path / "part_000.parquet"),
            rows,
            row_group_size=1,
            text_rows=texts,
        )

        loader = HFImageLoader(input_pattern=tmp_path, text_column="text")
        monkeypatch.setattr(
            loader,
            "_decode_image",
            lambda img_data: pytest.fail("load_text_batch should not decode images"),
        )

        loaded = loader.load_text_batch(np.array([0, 2], dtype=np.int64))
        loader.close()

        assert loaded == ["alpha", "gamma"]

    def test_parquet_multi_image_loader_load_text_batch_groups_docs(self, tmp_path, monkeypatch):
        rows = [
            [(11, 21), (31, 41)],
            [(51, 61)],
        ]
        texts = ["doc0", "doc1"]
        shard_path = str(tmp_path / "part_000.parquet")
        _write_hf_parquet_shard(
            shard_path,
            rows,
            column_name="images",
            multi_image=True,
            row_group_size=1,
            text_rows=texts,
        )

        manifest_path = str(tmp_path / "physical_multi_manifest.parquet")
        scan_hf_dataset(
            input_pattern=str(tmp_path / "*.parquet"),
            output_manifest=manifest_path,
            image_list_column="images",
            num_workers=1,
        )

        loader = HFImageLoader(
            input_pattern=str(tmp_path / "*.does_not_matter"),
            manifest_path=manifest_path,
            image_list_column="images",
            text_column="text",
        )
        monkeypatch.setattr(
            loader,
            "_decode_image",
            lambda img_data: pytest.fail("load_text_batch should not decode images"),
        )

        loaded = loader.load_text_batch(
            np.array([0, 1, 2], dtype=np.int64),
            group_slices=np.array([[0, 2], [2, 3]], dtype=np.int64),
        )
        loader.close()

        assert loaded == ["doc0", "doc1"]

    def test_parquet_multi_image_loader_uses_physical_manifest_coordinates(self, tmp_path):
        rows_a = [
            [(11, 21), (31, 41)],
            [(51, 61)],
        ]
        rows_b = [
            [(71, 81), (91, 101)],
        ]
        _write_hf_parquet_shard(
            str(tmp_path / "part_000.parquet"),
            rows_a,
            column_name="images",
            multi_image=True,
            row_group_size=1,
        )
        _write_hf_parquet_shard(
            str(tmp_path / "part_001.parquet"),
            rows_b,
            column_name="images",
            multi_image=True,
            row_group_size=1,
        )

        manifest_path = str(tmp_path / "physical_multi_manifest.parquet")
        scan_hf_dataset(
            input_pattern=str(tmp_path / "*.parquet"),
            output_manifest=manifest_path,
            image_list_column="images",
            num_workers=2,
        )

        loader = HFImageLoader(
            input_pattern=str(tmp_path / "*.does_not_matter"),
            manifest_path=manifest_path,
            image_list_column="images",
        )
        images, _ = loader.load_batch(np.array([0, 1, 3, 4], dtype=np.int64))
        loader.close()

        assert loader._uses_physical_manifest is True
        assert [img.size for img in images] == [(11, 21), (31, 41), (71, 81), (91, 101)]


class TestOrderedPool:
    def test_fatal_error_shuts_down_pool_without_waiting(self, monkeypatch):
        from vision_tokenization.indexing.scanners import _parallel as parallel_mod

        created_pools = []

        class FakeFuture:
            def __init__(self, *, result=None, exc=None):
                self._result = result
                self._exc = exc
                self.cancelled = False

            def result(self):
                if self._exc is not None:
                    raise self._exc
                return self._result

            def cancel(self):
                self.cancelled = True

        class FakeExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers
                self.shutdown_calls = []
                created_pools.append(self)

            def shutdown(self, wait=True, cancel_futures=False):
                self.shutdown_calls.append((wait, cancel_futures))

        futures = [
            FakeFuture(exc=ValueError("boom")),
            FakeFuture(result="ok"),
        ]

        def fake_wait(fs, return_when):
            return {fs[0]}, set(fs[1:])

        monkeypatch.setattr(parallel_mod, "ProcessPoolExecutor", FakeExecutor)
        monkeypatch.setattr(parallel_mod, "wait", fake_wait)

        with pytest.raises(ValueError, match="boom"):
            parallel_mod.run_ordered_pool(
                n_items=2,
                submit_fn=lambda _pool, idx: futures[idx],
                emit_fn=lambda _idx, _result: None,
                num_workers=2,
            )

        assert len(created_pools) == 1
        assert created_pools[0].shutdown_calls == [(False, True)]
        assert futures[1].cancelled is True


class TestMergeShards:
    """Tests for the post-tokenization shard merger."""

    def _create_shard(self, path, sequences):
        """Create a .bin/.idx shard with the given sequences."""
        try:
            from vision_tokenization.formats.megatron import (
                IndexedDatasetBuilder,
            )
        except ImportError:
            pytest.skip("indexed_dataset_megatron not available")

        builder = IndexedDatasetBuilder(str(path) + ".bin", dtype=np.int32)
        for seq in sequences:
            builder.add_item(torch.tensor(seq, dtype=torch.int32))
            builder.end_document()
        builder.finalize(str(path) + ".idx")

    def _read_shard(self, path):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from megatron.core.datasets.indexed_dataset import IndexedDataset

        ds = IndexedDataset(str(path))
        return [ds[i].tolist() for i in range(len(ds))]

    def test_merge_combines_shards(self, tmp_path):
        """Merge multiple rank shards into a single file."""
        try:
            from megatron.core.datasets.indexed_dataset import IndexedDataset
        except ImportError:
            pytest.skip("megatron not available")

        from vision_tokenization.pipeline.output.merge import merge_shards

        # Create fake rank shards
        self._create_shard(tmp_path / "rank_0000_chunk_0000", [[1, 2, 3], [4, 5]])
        self._create_shard(tmp_path / "rank_0001_chunk_0000", [[6, 7], [8, 9, 10, 11]])

        result = merge_shards(tmp_path, output_name="merged")
        assert result is not None

        seqs = self._read_shard(tmp_path / "merged")
        assert len(seqs) == 4
        assert seqs[0] == [1, 2, 3]
        assert seqs[1] == [4, 5]
        assert seqs[2] == [6, 7]
        assert seqs[3] == [8, 9, 10, 11]

    def test_maybe_merge_waits_for_all_ranks(self, tmp_path):
        """maybe_merge_shards returns None until all checkpoints exist."""
        try:
            from megatron.core.datasets.indexed_dataset import IndexedDataset
        except ImportError:
            pytest.skip("megatron not available")

        from vision_tokenization.pipeline.output.merge import maybe_merge_shards

        self._create_shard(tmp_path / "rank_0000_chunk_0000", [[1, 2]])
        self._create_shard(tmp_path / "rank_0001_chunk_0000", [[3, 4]])

        # Only rank 0 checkpoint exists
        torch.save({}, tmp_path / "rank_0000_checkpoint.pt")
        result = maybe_merge_shards(tmp_path, expected_ranks=2)
        assert result is None
        assert not (tmp_path / "merged.bin").exists()

        # Now rank 1 finishes
        torch.save({}, tmp_path / "rank_0001_checkpoint.pt")
        result = maybe_merge_shards(tmp_path, expected_ranks=2)
        assert result is not None
        assert (tmp_path / "merged.bin").exists()

    def test_maybe_merge_is_idempotent(self, tmp_path):
        """Calling maybe_merge_shards again skips if merged file exists."""
        try:
            from megatron.core.datasets.indexed_dataset import IndexedDataset
        except ImportError:
            pytest.skip("megatron not available")

        from vision_tokenization.pipeline.output.merge import maybe_merge_shards

        self._create_shard(tmp_path / "rank_0000_chunk_0000", [[1, 2]])
        torch.save({}, tmp_path / "rank_0000_checkpoint.pt")

        # First call merges
        result1 = maybe_merge_shards(tmp_path, expected_ranks=1)
        assert result1 is not None
        mtime1 = (tmp_path / "merged.bin").stat().st_mtime

        # Second call skips (file already exists)
        result2 = maybe_merge_shards(tmp_path, expected_ranks=1)
        assert result2 is not None
        mtime2 = (tmp_path / "merged.bin").stat().st_mtime
        assert mtime1 == mtime2

    def test_merge_empty_dir_returns_none(self, tmp_path):
        """No shards → returns None."""
        from vision_tokenization.pipeline.output.merge import merge_shards

        assert merge_shards(tmp_path) is None
