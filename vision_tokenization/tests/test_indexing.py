"""Tests for vision_tokenization.indexing — CPU-only, no tokenizer needed."""

import io
import os
import tarfile
import tempfile

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from vision_tokenization.indexing._scan_worker import scan_single_tar
from vision_tokenization.indexing.clustered_batch_planner import (
    BatchPlan,
    plan_clustered_batches,
)
from vision_tokenization.indexing.manifest import (
    load_resolution_arrays,
    load_wds_manifest,
    save_wds_manifest,
)
from vision_tokenization.indexing.reader import TarRandomAccessReader
from vision_tokenization.indexing.scanner_wds import scan_wds_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ======================================================================
# TestWDSScanner
# ======================================================================
class TestWDSScanner:

    def test_scan_single_tar(self, tmp_path):
        """Scan a tar with 5 images, verify correct dims and offsets."""
        samples = [
            {"key": f"{i:06d}", "ext": "jpg", "width": 100 + i * 10, "height": 200 + i * 10}
            for i in range(5)
        ]
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
            samples = [
                {"key": f"{shard_idx:03d}_{i:03d}", "ext": "jpg", "width": 80, "height": 80}
                for i in range(5)
            ]
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


# ======================================================================
# TestWDSRandomAccess
# ======================================================================
class TestWDSRandomAccess:

    def _scan_and_build_refs(self, tar_path: str):
        """Helper: scan a tar and return list of (tar_path, offset, size, original_img)."""
        samples_meta = [
            {"key": f"{i:06d}", "ext": "jpg", "width": 50 + i * 20, "height": 50 + i * 20,
             "color": (i * 40 % 256, 100, 200)}
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

        plan = plan_clustered_batches(path, batch_size=32, max_batch_tokens=999999)
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
        plan = plan_clustered_batches(path, batch_size=bs, max_batch_tokens=999999)
        for batch in plan.batches:
            assert len(batch.sample_indices) <= bs

    def test_all_samples_assigned(self, tmp_path):
        """Every sample appears in exactly one batch."""
        rng = np.random.RandomState(99)
        N = 300
        w = rng.randint(100, 500, N)
        h = rng.randint(100, 500, N)
        path = self._create_manifest(tmp_path, w, h)

        plan = plan_clustered_batches(path, batch_size=20, max_batch_tokens=999999)
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

        plan = plan_clustered_batches(path, batch_size=32, max_batch_tokens=999999)

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
            f"Mean within-batch AR std ({mean_within_std:.4f}) should be < "
            f"global AR std ({global_std:.4f})"
        )

    def test_resolution_filtering(self, tmp_path):
        """min_pixels should filter out small images."""
        # 50 tiny (10x10=100px) + 50 normal (200x200=40000px)
        w = np.array([10] * 50 + [200] * 50)
        h = np.array([10] * 50 + [200] * 50)
        path = self._create_manifest(tmp_path, w, h)

        plan = plan_clustered_batches(path, batch_size=10, max_batch_tokens=999999, min_pixels=1000)
        assert plan.total_filtered == 50
        all_idx = np.concatenate([b.sample_indices for b in plan.batches])
        assert len(all_idx) == 50
        # All assigned indices should be from the "normal" images (index >= 50)
        assert all(i >= 50 for i in all_idx)

    def test_worker_split(self, tmp_path):
        """split_for_workers(4) produces up to 4 chunks covering all batches."""
        rng = np.random.RandomState(0)
        w = rng.randint(100, 500, 200)
        h = rng.randint(100, 500, 200)
        path = self._create_manifest(tmp_path, w, h)

        plan = plan_clustered_batches(path, batch_size=10, max_batch_tokens=999999)
        chunks = plan.split_for_workers(4)
        assert len(chunks) <= 4
        # Flatten and verify all batches covered
        flat = [b for chunk in chunks for b in chunk]
        assert len(flat) == len(plan.batches)

    def test_multi_image_requires_group_column(self, tmp_path):
        """multi_image=True on a single-image manifest should raise ValueError."""
        w = np.array([200] * 10)
        h = np.array([200] * 10)
        path = self._create_manifest(tmp_path, w, h)

        with pytest.raises(ValueError, match="multi_image=True but manifest has no group_id"):
            plan_clustered_batches(path, batch_size=4, max_batch_tokens=999999, multi_image=True)


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
                    w, h = 320, 240   # landscape
                elif cluster == 1:
                    w, h = 200, 200   # square
                else:
                    w, h = 150, 400   # portrait
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
        plan = plan_clustered_batches(manifest_path, batch_size=8, max_batch_tokens=999999)
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
                refs = [
                    (tar_paths_col[i], offsets_col[i], sizes_col[i])
                    for i in batch.sample_indices
                ]
                images = reader.read_batch(refs)
                for img, idx in zip(images, batch.sample_indices):
                    assert img is not None
                    assert img.size == (widths_col[idx], heights_col[idx])
