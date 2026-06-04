"""Tests for the provenance sidecar feature (source-row ↔ output-position map).

Covers the per-chunk writer buffer, _filter_none lockstep filtering, the
stage2/lct split routing, the plan-derived per-document source rows, and the
merge-time concatenation / parquet resolution / two-run group map.

Merge round-trip tests that need ``megatron.core`` are skipped when it isn't
importable (it ships only inside the training container).
"""

import types

import numpy as np
import pytest
import torch

from vision_tokenization.formats.megatron import IndexedDatasetBuilder
from vision_tokenization.pipeline.output import provenance as prov
from vision_tokenization.pipeline.output.direct.handler import TokenizationHandler
from vision_tokenization.pipeline.output.direct.writer import (
    MicroShardWriter,
    SplitMicroShardWriter,
)
from vision_tokenization.pipeline.runtime.checkpoint import WorkerStats


class _FakeTokenizer:
    """Minimal tokenizer surface used by MicroShardWriter.setup_writer."""

    text_tokenizer = list(range(256))
    vision_token_offset = None


def _seq(n, val=1):
    return torch.full((n,), val, dtype=torch.int64)


# ---------------------------------------------------------------------------
# Low-level sidecar I/O
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_exact_filename(tmp_path):
    path = str(tmp_path / "shard.src.npy")
    prov.save_source_ids(path, [3, 1, 4, 1, 5])
    # No ``.npy`` munging — the exact path exists.
    assert (tmp_path / "shard.src.npy").exists()
    assert not (tmp_path / "shard.src.npy.npy").exists()
    np.testing.assert_array_equal(prov.load_source_ids(path), [3, 1, 4, 1, 5])
    assert prov.load_source_ids(path).dtype == np.int64


def test_read_seq_count_matches_builder(tmp_path):
    prefix = str(tmp_path / "rank_0000_chunk_0000")
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=np.int32)
    for i in range(7):
        builder.add_item(_seq(3 + i))
        builder.end_document()
    builder.finalize(prefix + ".idx")
    assert prov.read_seq_count(prefix) == 7


# ---------------------------------------------------------------------------
# Direct writer: per-chunk provenance buffer
# ---------------------------------------------------------------------------


def test_microshard_writer_buffer_resets_per_chunk(tmp_path):
    w = MicroShardWriter()
    w.setup_writer(str(tmp_path), rank=0, chunk_id=0, tokenizer=_FakeTokenizer(), emit_prov=True)
    stats = WorkerStats()

    src_ids = [100, 101, 102, 103, 104]
    for i, sid in enumerate(src_ids[:3]):
        w.write_sequence(_seq(4), stats, source_id=sid)
    w.checkpoint_writer()  # finalizes chunk 0, opens chunk 1 (buffer reset)
    for sid in src_ids[3:]:
        w.write_sequence(_seq(4), stats, source_id=sid)
    w.finalize_writer()

    c0 = str(tmp_path / "rank_0000_chunk_0000")
    c1 = str(tmp_path / "rank_0000_chunk_0001")
    np.testing.assert_array_equal(prov.load_source_ids(prov.sidecar_path(c0)), [100, 101, 102])
    np.testing.assert_array_equal(prov.load_source_ids(prov.sidecar_path(c1)), [103, 104])
    # Sidecar length matches each shard's sequence count.
    assert len(prov.load_source_ids(prov.sidecar_path(c0))) == prov.read_seq_count(c0)
    assert len(prov.load_source_ids(prov.sidecar_path(c1))) == prov.read_seq_count(c1)
    # Atomic finalize leaves no temporary files behind (.bin/.idx/.src.npy .tmp).
    assert list(tmp_path.rglob("*.tmp")) == []


def test_microshard_writer_disabled_writes_no_sidecar(tmp_path):
    w = MicroShardWriter()
    w.setup_writer(str(tmp_path), rank=0, chunk_id=0, tokenizer=_FakeTokenizer(), emit_prov=False)
    stats = WorkerStats()
    for _ in range(3):
        w.write_sequence(_seq(4), stats)  # no source_id needed
    w.finalize_writer()
    assert list(tmp_path.glob("*.src.npy")) == []


def test_split_writer_routes_sidecar_by_length(tmp_path):
    w = SplitMicroShardWriter(seqlen_threshold=5)
    w.setup_writer(str(tmp_path), rank=0, stage2_chunk_id=0, lct_chunk_id=0, tokenizer=_FakeTokenizer(), emit_prov=True)
    stats = WorkerStats()
    # short -> stage2, long -> lct
    w.write_sequence(_seq(3), stats, source_id=10)  # stage2
    w.write_sequence(_seq(9), stats, source_id=20)  # lct
    w.write_sequence(_seq(4), stats, source_id=30)  # stage2
    w.finalize_writer()

    s2 = str(tmp_path / "stage2" / "rank_0000_chunk_0000")
    lct = str(tmp_path / "lct" / "rank_0000_chunk_0000")
    np.testing.assert_array_equal(prov.load_source_ids(prov.sidecar_path(s2)), [10, 30])
    np.testing.assert_array_equal(prov.load_source_ids(prov.sidecar_path(lct)), [20])


# ---------------------------------------------------------------------------
# _filter_none lockstep
# ---------------------------------------------------------------------------


def test_filter_none_single_image_lockstep():
    images = ["a", None, "c", "d"]
    texts = ["ta", "tb", None, "td"]
    source_ids = np.array([10, 11, 12, 13])
    stats = WorkerStats()
    vi, vt, vs, vsrc = TokenizationHandler._filter_none(images, texts, None, stats, source_ids)
    assert vi == ["a", "d"]
    assert vt == ["ta", "td"]
    assert vs is None
    assert vsrc == [10, 13]  # dropped rows 11 (None img) and 12 (None text)


def test_filter_none_image_only_lockstep():
    images = ["a", None, "c"]
    source_ids = np.array([10, 11, 12])
    stats = WorkerStats()
    vi, vt, vs, vsrc = TokenizationHandler._filter_none(images, None, None, stats, source_ids)
    assert vi == ["a", "c"]
    assert vt is None
    assert vsrc == [10, 12]


def test_filter_none_multi_image_group_one_id_per_group():
    # two groups: [img0,img1] and [img2]; second group's text is None -> dropped
    images = ["i0", "i1", "i2"]
    texts = ["t0", None]
    group_slices = np.array([[0, 2], [2, 3]])
    source_ids = np.array([100, 101, 200])  # flat per-image rows
    stats = WorkerStats()
    vi, vt, vs, vsrc = TokenizationHandler._filter_none(
        images,
        texts,
        group_slices,
        stats,
        source_ids,
    )
    assert vi == ["i0", "i1"]
    assert vt == ["t0"]
    np.testing.assert_array_equal(vs, [[0, 2]])
    assert vsrc == [100]  # group's first image row only


def test_filter_none_without_source_ids_returns_none():
    images = ["a", None]
    stats = WorkerStats()
    vi, vt, vs, vsrc = TokenizationHandler._filter_none(images, None, None, stats, None)
    assert vsrc is None


# ---------------------------------------------------------------------------
# Plan-derived per-document source rows
# ---------------------------------------------------------------------------


def _fake_plan(document_id, component_index, source_ref):
    comp = types.SimpleNamespace(
        document_id=np.asarray(document_id, dtype=np.int64),
        component_index=np.asarray(component_index, dtype=np.int16),
        source_ref=np.asarray(source_ref, dtype=np.int64),
    )
    return types.SimpleNamespace(components=comp)


def test_doc_first_source_ref_picks_lowest_component():
    # doc 0: comps at rows 5 (ci0), 6 (ci1); doc 1: row 9 (ci0)
    plan = _fake_plan(
        document_id=[0, 0, 1],
        component_index=[1, 0, 0],
        source_ref=[6, 5, 9],
    )
    docs, src = prov.doc_first_source_ref(plan)
    np.testing.assert_array_equal(docs, [0, 1])
    np.testing.assert_array_equal(src, [5, 9])  # doc 0 -> first component (ci0) at row 5


def test_doc_source_ids_for_ordered():
    plan = _fake_plan(
        document_id=[0, 0, 1, 2],
        component_index=[0, 1, 0, 0],
        source_ref=[5, 6, 9, 12],
    )
    out = prov.doc_source_ids_for(plan, np.array([2, 0, 1]))
    np.testing.assert_array_equal(out, [12, 5, 9])


# ---------------------------------------------------------------------------
# Merge-time concatenation
# ---------------------------------------------------------------------------


def _make_shard(tmp_path, name, src_ids, *, with_sidecar=True):
    prefix = str(tmp_path / name)
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=np.int32)
    for _ in src_ids:
        builder.add_item(_seq(3))
        builder.end_document()
    builder.finalize(prefix + ".idx")
    if with_sidecar:
        prov.save_source_ids(prov.sidecar_path(prefix), src_ids)
    return prefix


def test_concat_shard_sidecars_orders_and_validates(tmp_path):
    p0 = _make_shard(tmp_path, "rank_0000_chunk_0000", [1, 2, 3])
    p1 = _make_shard(tmp_path, "rank_0001_chunk_0000", [4, 5])
    out = prov.concat_shard_sidecars([p0, p1])
    np.testing.assert_array_equal(out, [1, 2, 3, 4, 5])


def test_concat_shard_sidecars_auto_skips_when_absent(tmp_path):
    p0 = _make_shard(tmp_path, "rank_0000_chunk_0000", [1, 2], with_sidecar=False)
    assert prov.concat_shard_sidecars([p0], require=None) is None


def test_concat_shard_sidecars_partial_raises_when_required(tmp_path):
    p0 = _make_shard(tmp_path, "rank_0000_chunk_0000", [1, 2])
    p1 = _make_shard(tmp_path, "rank_0001_chunk_0000", [3], with_sidecar=False)
    with pytest.raises(FileNotFoundError):
        prov.concat_shard_sidecars([p0, p1], require=True)


def test_concat_shard_sidecars_length_mismatch_raises(tmp_path):
    p0 = _make_shard(tmp_path, "rank_0000_chunk_0000", [1, 2, 3])
    # Corrupt the sidecar to a wrong length.
    prov.save_source_ids(prov.sidecar_path(p0), [1, 2])
    with pytest.raises(ValueError):
        prov.concat_shard_sidecars([p0])


# ---------------------------------------------------------------------------
# Manifest resolution + group map
# ---------------------------------------------------------------------------


def _write_manifest(path, *, kind):
    import pyarrow as pa
    import pyarrow.parquet as pq

    if kind == "hf":
        table = pa.table(
            {
                "sample_index": pa.array([1000, 1001, 1002, 1003], pa.int64()),
                "width": pa.array([1, 1, 1, 1], pa.int32()),
                "height": pa.array([1, 1, 1, 1], pa.int32()),
            }
        )
    else:  # wds
        table = pa.table(
            {
                "sample_key": pa.array(["k0", "k1", "k2", "k3"]),
                "width": pa.array([1, 1, 1, 1], pa.int32()),
                "height": pa.array([1, 1, 1, 1], pa.int32()),
            }
        )
    pq.write_table(table, path)


def _make_merged_with_sidecar(tmp_path, name, src_rows):
    prefix = str(tmp_path / name)
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=np.int32)
    for _ in src_rows:
        builder.add_item(_seq(3))
        builder.end_document()
    builder.finalize(prefix + ".idx")
    prov.save_source_ids(prov.sidecar_path(prefix), src_rows)
    return prefix


def test_write_provenance_parquet_hf(tmp_path):
    import pyarrow.parquet as pq

    manifest = str(tmp_path / "manifest.parquet")
    _write_manifest(manifest, kind="hf")
    prefix = _make_merged_with_sidecar(tmp_path, "merged", [2, 0, 3])
    out = prov.write_provenance_parquet(prefix, manifest)
    t = pq.read_table(out)
    np.testing.assert_array_equal(t.column("output_index").to_numpy(), [0, 1, 2])
    np.testing.assert_array_equal(t.column("manifest_row").to_numpy(), [2, 0, 3])
    assert t.column("source_id").to_pylist() == [1002, 1000, 1003]


def test_write_provenance_parquet_wds_string_ids(tmp_path):
    import pyarrow.parquet as pq

    manifest = str(tmp_path / "manifest.parquet")
    _write_manifest(manifest, kind="wds")
    prefix = _make_merged_with_sidecar(tmp_path, "merged", [1, 3])
    out = prov.write_provenance_parquet(prefix, manifest)
    t = pq.read_table(out)
    assert t.column("source_id").to_pylist() == ["k1", "k3"]


def test_build_group_map_intersection(tmp_path):
    manifest = str(tmp_path / "manifest.parquet")
    _write_manifest(manifest, kind="hf")
    # Run A covers source rows [0,1,2]; run B covers [1,2,3].
    pa_prefix = _make_merged_with_sidecar(tmp_path, "merged_a", [0, 1, 2])
    pb_prefix = _make_merged_with_sidecar(tmp_path, "merged_b", [1, 2, 3])
    prov_a = prov.write_provenance_parquet(pa_prefix, manifest, str(tmp_path / "a.provenance.parquet"))
    prov_b = prov.write_provenance_parquet(pb_prefix, manifest, str(tmp_path / "b.provenance.parquet"))

    import pyarrow.parquet as pq

    out = prov.build_group_map(prov_a, prov_b, str(tmp_path / "group_map.parquet"))
    t = pq.read_table(out)
    rows = {
        sid: (pa_, pb_)
        for sid, pa_, pb_ in zip(
            t.column("source_id").to_pylist(),
            t.column("pos_in_A").to_pylist(),
            t.column("pos_in_B").to_pylist(),
        )
    }
    # shared source ids are 1001 (row1) and 1002 (row2)
    assert set(rows) == {1001, 1002}
    # row 1 -> pos 1 in A, pos 0 in B; row 2 -> pos 2 in A, pos 1 in B
    assert rows[1001] == (1, 0)
    assert rows[1002] == (2, 1)


def test_build_group_map_rejects_duplicate_rows(tmp_path):
    manifest = str(tmp_path / "manifest.parquet")
    _write_manifest(manifest, kind="hf")
    pa_prefix = _make_merged_with_sidecar(tmp_path, "merged_a", [0, 0, 1])  # dup row 0
    pb_prefix = _make_merged_with_sidecar(tmp_path, "merged_b", [0, 1, 2])
    prov_a = prov.write_provenance_parquet(pa_prefix, manifest, str(tmp_path / "a.provenance.parquet"))
    prov_b = prov.write_provenance_parquet(pb_prefix, manifest, str(tmp_path / "b.provenance.parquet"))
    with pytest.raises(ValueError):
        prov.build_group_map(prov_a, prov_b, str(tmp_path / "group_map.parquet"))


# ---------------------------------------------------------------------------
# Merge round-trip (needs megatron.core)
# ---------------------------------------------------------------------------


def test_merge_shards_emits_aligned_sidecar(tmp_path):
    pytest.importorskip("megatron.core")
    from vision_tokenization.pipeline.output.merge import merge_shards

    _make_shard(tmp_path, "rank_0000_chunk_0000", [10, 11, 12])
    _make_shard(tmp_path, "rank_0001_chunk_0000", [13, 14])
    result = merge_shards(tmp_path, output_name="merged")
    assert result is not None
    merged_src = prov.load_source_ids(prov.sidecar_path(str(result)))
    np.testing.assert_array_equal(merged_src, [10, 11, 12, 13, 14])
    assert len(merged_src) == prov.read_seq_count(str(result))


def test_rewrite_dataset_no_cot_src_lockstep(tmp_path):
    pytest.importorskip("megatron.core")
    from vision_tokenization.pipeline.output.merge import rewrite_dataset

    prefix = _make_merged_with_sidecar(tmp_path, "merged", [10, 11, 12, 13])
    src_in = prov.load_source_ids(prov.sidecar_path(prefix))

    # Drop the 2nd sequence (index 1) via the transform.
    def transform(seq):
        return None if int(seq[0]) == 99 else seq

    # Rebuild merged so that one sequence triggers the drop.
    b = IndexedDatasetBuilder(str(tmp_path / "merged2.bin"), dtype=np.int32)
    for i, v in enumerate([1, 99, 1, 1]):
        b.add_item(_seq(3, v))
        b.end_document()
    b.finalize(str(tmp_path / "merged2.idx"))

    out_prefix = str(tmp_path / "merged_no_cot")
    rewrite_dataset(
        str(tmp_path / "merged2"),
        out_prefix,
        transform,
        src_in=src_in,
        src_out_path=prov.sidecar_path(out_prefix),
    )
    out_src = prov.load_source_ids(prov.sidecar_path(out_prefix))
    np.testing.assert_array_equal(out_src, [10, 12, 13])  # row 11 dropped in lockstep
