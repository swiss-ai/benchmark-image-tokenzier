"""Tests for SHAR-like spill writer and reader."""

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from vision_tokenization.pipeline.pooled.document import (
    AtomicDocument,
    Component,
)
from vision_tokenization.pipeline.pooled.spill import (
    recover_shard_progress,
    SpillReader,
    SpillWriter,
    recover_worker_shards,
    summarize_worker_shards,
    write_shard_progress,
)


def _make_doc(doc_id, mode, components_spec):
    """Helper: create AtomicDocument + component token arrays.

    components_spec: list of (kind, tokens_array) tuples.
    """
    comps = []
    tokens = []
    total = 0
    img_tok = 0
    for i, (kind, tok_arr) in enumerate(components_spec):
        tok = np.asarray(tok_arr, dtype=np.uint16)
        comps.append(Component(
            component_index=i,
            kind=kind,
            resize_height=32 if kind == "image" else 0,
            resize_width=32 if kind == "image" else 0,
            manifest_row=doc_id * 10 + i,
        ))
        tokens.append(tok)
        total += len(tok)
        if kind == "image":
            img_tok += len(tok)

    doc = AtomicDocument(
        document_id=doc_id,
        mode=mode,
        components=comps,
        total_tokens=total,
        image_tokens=img_tok,
        text_tokens=total - img_tok,
        manifest_group_id=doc_id,
    )
    return doc, tokens


class TestSpillWriter:
    def test_write_single_document(self, tmp_path):
        writer = SpillWriter(str(tmp_path), rank=0)
        writer.open()

        doc, tokens = _make_doc(0, "image2text", [
            ("image", [100, 101, 102]),
            ("text", [200, 201]),
        ])
        writer.add_document(doc, tokens)
        writer.finalize()

        # Check files exist
        worker_dir = tmp_path / "worker_00"
        assert (worker_dir / "_SUCCESS").exists()
        assert (worker_dir / "worker_stats.json").exists()
        assert (worker_dir / "documents.000000.parquet").exists()
        assert (worker_dir / "components.000000.parquet").exists()
        assert (worker_dir / "tokens.000000.bin").exists()

    def test_documents_parquet_content(self, tmp_path):
        writer = SpillWriter(str(tmp_path), rank=0)
        writer.open()

        doc, tokens = _make_doc(42, "interleave", [
            ("text", [10, 11]),
            ("image", [20, 21, 22]),
            ("text", [30]),
        ])
        writer.add_document(doc, tokens)
        writer.finalize()

        table = pq.read_table(tmp_path / "worker_00" / "documents.000000.parquet")
        assert len(table) == 1
        row = table.to_pydict()
        assert row["document_id"] == [42]
        assert row["mode"] == ["interleave"]
        assert row["num_components"] == [3]
        assert row["total_tokens"] == [6]
        assert row["image_tokens"] == [3]
        assert row["text_tokens"] == [3]

    def test_components_parquet_content(self, tmp_path):
        writer = SpillWriter(str(tmp_path), rank=0)
        writer.open()

        doc, tokens = _make_doc(7, "image2text", [
            ("image", [100, 101]),
            ("text", [200]),
        ])
        writer.add_document(doc, tokens)
        writer.finalize()

        table = pq.read_table(tmp_path / "worker_00" / "components.000000.parquet")
        assert len(table) == 2
        rows = table.to_pydict()
        assert rows["document_id"] == [7, 7]
        assert rows["component_index"] == [0, 1]
        assert rows["kind"] == ["image", "text"]
        assert rows["token_length"] == [2, 1]

    def test_tokens_bin_roundtrip(self, tmp_path):
        writer = SpillWriter(str(tmp_path), rank=0)
        writer.open()

        doc, tokens = _make_doc(0, "image_only", [
            ("image", [100, 200, 300]),
        ])
        writer.add_document(doc, tokens)
        writer.finalize()

        # Read back via SpillReader
        comp_table = pq.read_table(tmp_path / "worker_00" / "components.000000.parquet")
        row = comp_table.to_pydict()
        offset = row["token_offset"][0]
        length = row["token_length"][0]

        recovered = SpillReader.load_component_tokens(
            tmp_path / "worker_00", shard_id=0,
            token_offset=offset, token_length=length,
        )
        np.testing.assert_array_equal(recovered, [100, 200, 300])

    def test_multiple_documents(self, tmp_path):
        writer = SpillWriter(str(tmp_path), rank=0)
        writer.open()

        for i in range(5):
            doc, tokens = _make_doc(i, "image_only", [
                ("image", list(range(i * 10, i * 10 + 3))),
            ])
            writer.add_document(doc, tokens)

        writer.finalize()

        doc_table = pq.read_table(tmp_path / "worker_00" / "documents.000000.parquet")
        assert len(doc_table) == 5

        comp_table = pq.read_table(tmp_path / "worker_00" / "components.000000.parquet")
        assert len(comp_table) == 5

    def test_checkpoint_creates_new_shard(self, tmp_path):
        writer = SpillWriter(str(tmp_path), rank=0)
        writer.open()

        doc0, tokens0 = _make_doc(0, "image_only", [("image", [100])])
        writer.add_document(doc0, tokens0)

        done_shard = writer.checkpoint()
        assert done_shard == 0

        doc1, tokens1 = _make_doc(1, "image_only", [("image", [200])])
        writer.add_document(doc1, tokens1)

        writer.finalize()

        worker_dir = tmp_path / "worker_00"
        assert (worker_dir / "documents.000000.parquet").exists()
        assert (worker_dir / "documents.000001.parquet").exists()
        assert (worker_dir / "tokens.000000.bin").exists()
        assert (worker_dir / "tokens.000001.bin").exists()

    def test_worker_stats(self, tmp_path):
        writer = SpillWriter(str(tmp_path), rank=3)
        writer.open()

        for i in range(3):
            doc, tokens = _make_doc(i, "image_only", [("image", [100, 101])])
            writer.add_document(doc, tokens)

        writer.finalize()

        with open(tmp_path / "worker_03" / "worker_stats.json") as f:
            stats = json.load(f)

        assert stats["rank"] == 3
        assert stats["documents_written"] == 3
        assert stats["total_tokens"] == 6

    def test_recover_worker_shards_removes_incomplete_tail(self, tmp_path):
        writer = SpillWriter(str(tmp_path), rank=0)
        writer.open()

        doc0, tokens0 = _make_doc(0, "image_only", [("image", [100])])
        writer.add_document(doc0, tokens0)
        writer.checkpoint()

        worker_dir = tmp_path / "worker_00"
        (worker_dir / "documents.000001.parquet").write_bytes(b"stale")
        (worker_dir / "tokens.000001.bin").write_bytes(b"stale")

        next_shard_id = recover_worker_shards(worker_dir)

        assert next_shard_id == 1
        assert (worker_dir / "documents.000000.parquet").exists()
        assert not (worker_dir / "documents.000001.parquet").exists()
        assert not (worker_dir / "tokens.000001.bin").exists()

    def test_summarize_worker_shards_reads_durable_progress(self, tmp_path):
        writer = SpillWriter(str(tmp_path), rank=0)
        writer.open()

        doc0, tokens0 = _make_doc(0, "image2text", [
            ("image", [100, 101]),
            ("text", [200]),
        ])
        writer.add_document(doc0, tokens0)
        writer.checkpoint()

        doc1, tokens1 = _make_doc(1, "image_only", [
            ("image", [300, 301, 302]),
        ])
        writer.add_document(doc1, tokens1)
        writer.checkpoint()

        totals = summarize_worker_shards(tmp_path / "worker_00", 2)

        assert totals["documents_written"] == 2
        assert totals["total_tokens"] == 6
        assert totals["image_tokens"] == 5
        assert totals["text_tokens"] == 1

    def test_recover_shard_progress_trims_shards_without_progress_sidecar(self, tmp_path):
        writer = SpillWriter(str(tmp_path), rank=0)
        writer.open()
        worker_dir = tmp_path / "worker_00"

        doc0, tokens0 = _make_doc(0, "image_only", [("image", [100])])
        writer.add_document(doc0, tokens0)
        done0 = writer.checkpoint()
        write_shard_progress(
            worker_dir,
            done0,
            next_document_window_index=1,
            stats={"samples_processed": 1, "tokens_generated": 1},
        )

        doc1, tokens1 = _make_doc(1, "image_only", [("image", [200])])
        writer.add_document(doc1, tokens1)
        writer.checkpoint()
        writer.finalize()

        next_shard_id = recover_worker_shards(worker_dir)
        progress = recover_shard_progress(worker_dir, next_shard_id)

        assert progress["next_shard_id"] == 1
        assert progress["next_document_window_index"] == 1
        assert (worker_dir / "documents.000000.parquet").exists()
        assert not (worker_dir / "documents.000001.parquet").exists()
        assert not (worker_dir / "progress.000001.json").exists()


class TestSpillReader:
    def test_read_worker(self, tmp_path):
        writer = SpillWriter(str(tmp_path), rank=0)
        writer.open()
        doc, tokens = _make_doc(0, "image_only", [("image", [10, 20, 30])])
        writer.add_document(doc, tokens)
        writer.finalize()

        docs, comps, path = SpillReader.read_worker(tmp_path / "worker_00")
        assert len(docs) == 1
        assert len(comps) == 1
        assert path == tmp_path / "worker_00"

    def test_read_all_workers(self, tmp_path):
        for rank in range(3):
            writer = SpillWriter(str(tmp_path), rank=rank)
            writer.open()
            doc, tokens = _make_doc(rank, "image_only", [("image", [rank * 10])])
            writer.add_document(doc, tokens)
            writer.finalize()

        docs, comps, dirs = SpillReader.read_all_workers(tmp_path)
        assert len(docs) == 3
        assert len(comps) == 3
        assert len(dirs) == 3

    def test_read_skips_incomplete_worker(self, tmp_path):
        # Complete worker
        writer = SpillWriter(str(tmp_path), rank=0)
        writer.open()
        doc, tokens = _make_doc(0, "image_only", [("image", [10])])
        writer.add_document(doc, tokens)
        writer.finalize()

        # Incomplete worker (no _SUCCESS)
        incomplete = tmp_path / "worker_01"
        incomplete.mkdir()
        (incomplete / "documents.000000.parquet").touch()

        docs, comps, dirs = SpillReader.read_all_workers(tmp_path)
        assert len(docs) == 1
        assert len(dirs) == 1

    def test_load_component_tokens_multiple_components(self, tmp_path):
        writer = SpillWriter(str(tmp_path), rank=0)
        writer.open()

        doc, tokens = _make_doc(0, "interleave", [
            ("text", [10, 11, 12]),
            ("image", [100, 101]),
            ("text", [20]),
        ])
        writer.add_document(doc, tokens)
        writer.finalize()

        comp_table = pq.read_table(tmp_path / "worker_00" / "components.000000.parquet")
        rows = comp_table.to_pydict()

        for i, expected in enumerate([[10, 11, 12], [100, 101], [20]]):
            recovered = SpillReader.load_component_tokens(
                tmp_path / "worker_00", shard_id=0,
                token_offset=rows["token_offset"][i],
                token_length=rows["token_length"][i],
            )
            np.testing.assert_array_equal(recovered, expected)
