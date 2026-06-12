import hashlib
import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from vision_tokenization.indexing.alignment.ingest import (
    MARKER,
    MarkerMismatch,
    ingest_parquet,
    write_scan_parquet,
)


def _png(w, h, color=(123, 7, 89)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def _mk_parquet(tmp_path, rows):
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "in.parquet")
    return tmp_path / "in.parquet"


def _row(sid, img_bytes, n_markers=1, prompt_text="what is this?"):
    markers = "\n".join(["<image>"] * n_markers)
    return {
        "source-id": sid,
        "image": {"bytes": img_bytes, "path": f"{sid}.jpg"},
        "prompt": [{"role": "user", "content": f"{markers}\n{prompt_text}"}],
        "accepted": [{"role": "assistant", "content": "good answer"}],
        "rejected": [{"role": "assistant", "content": "bad answer"}],
    }


def test_dedup_identical_bytes(tmp_path):
    img = _png(32, 32)
    p = _mk_parquet(tmp_path, [_row("a", img), _row("b", img)])
    out = ingest_parquet(p, task="preference")
    assert len(out.unique_media) == 1
    mid = hashlib.sha256(img).hexdigest()
    assert out.unique_media[0].media_id == mid
    assert out.view_rows[0]["prompt_media_refs"] == [mid]
    assert out.view_rows[1]["prompt_media_refs"] == [mid]


def test_marker_normalized_and_counted(tmp_path):
    p = _mk_parquet(tmp_path, [_row("a", _png(32, 32), n_markers=1)])
    out = ingest_parquet(p, task="preference")
    content = out.view_rows[0]["prompt"][0]["content"]
    assert "<image>" not in content and content.count(MARKER) == 1


def test_marker_count_mismatch_raises(tmp_path):
    p = _mk_parquet(tmp_path, [_row("a", _png(32, 32), n_markers=2)])  # 2 markers, 1 image
    with pytest.raises(MarkerMismatch):
        ingest_parquet(p, task="preference")


def test_accidental_marker_in_response_rejected(tmp_path):
    row = _row("a", _png(32, 32))
    row["accepted"][0]["content"] = f"sneaky {MARKER} text"
    p = _mk_parquet(tmp_path, [row])
    with pytest.raises(MarkerMismatch, match="accidental"):
        ingest_parquet(p, task="preference")


def test_system_role_rejected(tmp_path):
    row = _row("a", _png(32, 32))
    row["prompt"].insert(0, {"role": "system", "content": "You are helpful."})
    p = _mk_parquet(tmp_path, [row])
    with pytest.raises(MarkerMismatch, match="system"):
        ingest_parquet(p, task="preference")


def test_unknown_task_fails_loud(tmp_path):
    p = _mk_parquet(tmp_path, [_row("a", _png(32, 32))])
    with pytest.raises(ValueError, match="rl_prompt"):
        ingest_parquet(p, task="rl_prompt")


def test_scan_artifact_one_row_per_unique_media_with_pil_geometry(tmp_path):
    img_a, img_b = _png(48, 32), _png(64, 16, color=(9, 200, 41))
    p = _mk_parquet(tmp_path, [_row("a", img_a), _row("b", img_b), _row("c", img_a)])
    out = ingest_parquet(p, task="preference")
    size = write_scan_parquet(tmp_path / "scan.parquet", out.unique_media)

    assert (tmp_path / "scan.parquet").exists()
    assert size == (tmp_path / "scan.parquet").stat().st_size
    scan = {r["media_id"]: r for r in
            pq.read_table(tmp_path / "scan.parquet").to_pylist()}
    assert len(scan) == len(out.unique_media) == 2  # img_a deduped across a/c
    for raw in (img_a, img_b):
        row = scan[hashlib.sha256(raw).hexdigest()]
        with Image.open(io.BytesIO(raw)) as im:
            assert (row["width"], row["height"]) == im.size
        assert row["raw_length_bytes"] == len(raw)


def test_corrupt_image_skipped_at_scan_with_pair_dropped(tmp_path):
    good = _png(32, 32)
    p = _mk_parquet(tmp_path, [_row("good", good), _row("bad", b"\xff\xd8not-an-image")])
    out = ingest_parquet(p, task="preference")
    assert out.n_skipped_media == 1
    assert [m.media_id for m in out.unique_media] == [hashlib.sha256(good).hexdigest()]
    assert [r["prompt_id"] for r in out.view_rows] == ["good"]
    write_scan_parquet(tmp_path / "scan.parquet", out.unique_media)
    assert pq.read_table(tmp_path / "scan.parquet").num_rows == 1


def test_sub16px_image_skipped_at_scan(tmp_path):
    p = _mk_parquet(tmp_path, [_row("tiny", _png(8, 8))])
    out = ingest_parquet(p, task="preference")
    assert out.n_skipped_media == 1
    assert out.unique_media == [] and out.view_rows == []
