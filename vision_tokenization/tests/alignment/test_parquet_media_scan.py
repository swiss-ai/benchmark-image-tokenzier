import hashlib
import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

import vision_tokenization.indexing.scanners.parquet_media_scan as scanmod
from vision_tokenization.indexing.alignment.ingest import (
    build_alignment_views_from_row_refs,
)
from vision_tokenization.indexing.scanners.parquet_media_scan import (
    dedup_media_scan,
    load_media_inventory,
    scan_parquet_media_refs,
    scan_parquet_media_refs_many,
)


def _png(w, h, color=(123, 7, 89)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def _row(sid, img_bytes):
    return {
        "source-id": sid,
        "image": {"bytes": img_bytes, "path": f"{sid}.png"},
        "prompt": [{"role": "user", "content": "<image>\nWhich color?"}],
        "accepted": [{"role": "assistant", "content": "good"}],
        "rejected": [{"role": "assistant", "content": "bad"}],
    }


def _write_parquet(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)


def test_parquet_media_scan_emits_only_media_refs_and_view_builder_materializes_views(tmp_path):
    img_a = _png(32, 32)
    img_b = _png(48, 32, color=(1, 2, 3))
    src = tmp_path / "input.parquet"
    _write_parquet(src, [
        _row("a0", img_a),
        _row("b0", img_b),
        _row("b1", img_b),
        _row("a1", img_a),
    ])

    build = tmp_path / "_build"
    scan = scan_parquet_media_refs(
        src,
        build,
        workers=2,
        batch_size=1,
    )
    dedup = dedup_media_scan(build, tmp_path / "published")

    assert scan.n_row_refs == 4
    assert scan.n_media_candidates == 4
    assert dedup.n_unique_media == 2
    assert dedup.n_valid_rows == 4
    assert not (build / "examples_raw").exists()

    row_refs = pq.read_table(tmp_path / "published" / "row_media_refs.parquet").to_pylist()
    assert [sorted(row) for row in row_refs] == [
        ["media_refs", "row_group", "row_index", "source", "source_path"],
    ] * 4

    views_path = tmp_path / "published" / "views" / "data.parquet"
    n_views = build_alignment_views_from_row_refs(
        src,
        Path(dedup.row_refs_path),
        views_path,
        task="preference",
        batch_size=1,
    )
    assert n_views == 4
    views = pq.read_table(views_path).to_pylist()
    assert sorted(row["prompt_id"] for row in views) == ["a0", "a1", "b0", "b1"]

    inventory = load_media_inventory(tmp_path / "published" / "media_unique.parquet")
    by_id = {media.media_id: media for media in inventory}
    assert by_id[hashlib.sha256(img_a).hexdigest()].raw == img_a
    assert by_id[hashlib.sha256(img_b).hexdigest()].raw == img_b

    # scan.parquet and the inventory must share row order (source_ref is positional).
    scan_ids = [
        r["media_id"]
        for r in pq.read_table(tmp_path / "published" / "scan.parquet").to_pylist()
    ]
    assert scan_ids == [media.media_id for media in inventory]


def test_parquet_media_scan_many_builds_views_from_multiple_source_parts(tmp_path):
    img_a = _png(32, 32)
    img_b = _png(48, 32, color=(1, 2, 3))
    part_a = tmp_path / "part-a.parquet"
    part_b = tmp_path / "part-b.parquet"
    _write_parquet(part_a, [_row("a0", img_a), _row("a1", img_a)])
    _write_parquet(part_b, [_row("b0", img_b)])

    build = tmp_path / "_build"
    scan = scan_parquet_media_refs_many(
        [part_a, part_b],
        build,
        workers=2,
        batch_size=1,
    )
    dedup = dedup_media_scan(build, tmp_path / "published")

    assert scan.n_source_rows == 3
    assert scan.n_row_refs == 3
    assert dedup.n_unique_media == 2
    assert dedup.n_valid_rows == 3

    views_path = tmp_path / "published" / "views" / "data.parquet"
    n_views = build_alignment_views_from_row_refs(
        [part_a, part_b],
        Path(dedup.row_refs_path),
        views_path,
        task="preference",
        batch_size=1,
    )
    assert n_views == 3
    views = pq.read_table(views_path).to_pylist()
    assert sorted(row["prompt_id"] for row in views) == ["a0", "a1", "b0"]


def test_parquet_media_scan_many_caps_process_pool_to_requested_workers(tmp_path, monkeypatch):
    img = _png(32, 32)
    paths = []
    for part_idx in range(3):
        path = tmp_path / f"part-{part_idx}.parquet"
        _write_parquet(path, [_row(f"{part_idx}-{row_idx}", img) for row_idx in range(3)])
        paths.append(path)

    captured = {}

    def fake_run_ordered_pool(n_items, submit_fn, emit_fn, num_workers, **kwargs):
        captured["n_items"] = n_items
        captured["num_workers"] = num_workers
        for idx in range(n_items):
            emit_fn(idx, {
                "n_source_rows": 0,
                "n_row_refs": 0,
                "n_media_candidates": 0,
            })

    monkeypatch.setattr(scanmod, "run_ordered_pool", fake_run_ordered_pool)

    scan_parquet_media_refs_many(
        paths,
        tmp_path / "_build",
        workers=2,
        batch_size=1,
    )

    assert captured["n_items"] > 2
    assert captured["num_workers"] == 2


def test_deduper_filters_raw_examples_that_reference_invalid_media(tmp_path):
    good = _png(32, 32)
    tiny = _png(8, 8)
    src = tmp_path / "input.parquet"
    _write_parquet(src, [_row("good", good), _row("tiny", tiny)])

    build = tmp_path / "_build"
    scan = scan_parquet_media_refs(
        src,
        build,
        workers=2,
        batch_size=1,
    )
    dedup = dedup_media_scan(build, tmp_path / "published")

    assert scan.n_row_refs == 2
    assert scan.n_media_candidates == 2
    assert dedup.n_unique_media == 1
    assert dedup.n_valid_rows == 1
    assert dedup.n_filtered_rows == 1

    row_refs = pq.read_table(tmp_path / "published" / "row_media_refs.parquet").to_pylist()
    assert [row["source"] for row in row_refs] == ["good"]
    scan_rows = pq.read_table(tmp_path / "published" / "scan.parquet").to_pylist()
    assert [row["media_id"] for row in scan_rows] == [hashlib.sha256(good).hexdigest()]


def test_null_arrow_image_bytes_use_empty_media_sentinel_and_are_filtered(tmp_path):
    src = tmp_path / "input.parquet"
    _write_parquet(src, [_row("missing", None)])

    build = tmp_path / "_build"
    scan = scan_parquet_media_refs(
        src,
        build,
        workers=1,
        batch_size=1,
    )
    dedup = dedup_media_scan(build, tmp_path / "published")

    assert scan.n_row_refs == 1
    assert scan.n_media_candidates == 1
    assert dedup.n_unique_media == 0
    assert dedup.n_invalid_media == 1
    assert dedup.n_valid_rows == 0
    assert dedup.n_filtered_rows == 1
    assert pq.read_table(tmp_path / "published" / "row_media_refs.parquet").num_rows == 0


def test_dedup_raises_when_occurrences_exceed_in_memory_ceiling(tmp_path):
    """An occurrence set over the in-memory ceiling fails loud, never silently
    produces an empty inventory."""
    src = tmp_path / "input.parquet"
    _write_parquet(src, [_row("a0", _png(32, 32))])

    build = tmp_path / "_build"
    scan_parquet_media_refs(src, build, workers=1, batch_size=1)
    with pytest.raises(RuntimeError, match="in-memory dedup ceiling"):
        dedup_media_scan(build, tmp_path / "published", max_in_memory_bytes=0)
