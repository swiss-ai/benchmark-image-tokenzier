import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vision_tokenization.indexing.alignment.ingest import (
    MARKER,
    MarkerMismatch,
    ingest_parquet,
)


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
    img = b"\xff\xd8same-image"
    p = _mk_parquet(tmp_path, [_row("a", img), _row("b", img)])
    out = ingest_parquet(p, task="preference")
    assert len(out.unique_media) == 1
    mid = hashlib.sha256(img).hexdigest()
    assert out.unique_media[0].media_id == mid
    assert out.view_rows[0]["prompt_media_refs"] == [mid]
    assert out.view_rows[1]["prompt_media_refs"] == [mid]


def test_marker_normalized_and_counted(tmp_path):
    p = _mk_parquet(tmp_path, [_row("a", b"i1", n_markers=1)])
    out = ingest_parquet(p, task="preference")
    content = out.view_rows[0]["prompt"][0]["content"]
    assert "<image>" not in content and content.count(MARKER) == 1


def test_marker_count_mismatch_raises(tmp_path):
    p = _mk_parquet(tmp_path, [_row("a", b"i1", n_markers=2)])  # 2 markers, 1 image
    with pytest.raises(MarkerMismatch):
        ingest_parquet(p, task="preference")


def test_accidental_marker_in_response_rejected(tmp_path):
    row = _row("a", b"i1")
    row["accepted"][0]["content"] = f"sneaky {MARKER} text"
    p = _mk_parquet(tmp_path, [row])
    with pytest.raises(MarkerMismatch, match="accidental"):
        ingest_parquet(p, task="preference")


def test_system_role_rejected(tmp_path):
    row = _row("a", b"i1")
    row["prompt"].insert(0, {"role": "system", "content": "You are helpful."})
    p = _mk_parquet(tmp_path, [row])
    with pytest.raises(MarkerMismatch, match="system"):
        ingest_parquet(p, task="preference")


def test_unknown_task_fails_loud(tmp_path):
    p = _mk_parquet(tmp_path, [_row("a", b"i1")])
    with pytest.raises(ValueError, match="rl_prompt"):
        ingest_parquet(p, task="rl_prompt")
