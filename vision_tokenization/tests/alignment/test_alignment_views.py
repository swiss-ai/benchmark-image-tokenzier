import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from vision_tokenization.indexing.alignment.ingest import (
    MARKER,
    MarkerMismatch,
    build_alignment_views_from_row_refs,
)


def _png(w, h, color=(123, 7, 89)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def _row(sid, n_markers=1):
    markers = "\n".join(["<image>"] * n_markers)
    return {
        "source-id": sid,
        "image": {"bytes": _png(32, 32), "path": f"{sid}.jpg"},
        "prompt": [{"role": "user", "content": f"{markers}\nwhat is this?"}],
        "accepted": [{"role": "assistant", "content": "good answer"}],
        "rejected": [{"role": "assistant", "content": "bad answer"}],
    }


def _write_source(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)


def _write_row_refs(path, source_path, refs_by_row):
    rows = [
        {
            "source_path": str(source_path),
            "row_group": row_idx,
            "row_index": 0,
            "source": f"r{row_idx}",
            "media_refs": refs,
        }
        for row_idx, refs in enumerate(refs_by_row)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_view_builder_normalizes_marker_and_uses_filtered_refs(tmp_path):
    src = tmp_path / "source.parquet"
    _write_source(src, [_row("a"), _row("b")])
    refs = tmp_path / "row_media_refs.parquet"
    _write_row_refs(refs, src, [["media-a"], ["media-b"]])

    out = tmp_path / "views" / "data.parquet"
    n_rows = build_alignment_views_from_row_refs(
        src,
        refs,
        out,
        task="preference",
        batch_size=1,
    )

    rows = pq.read_table(out).to_pylist()
    assert n_rows == 2
    assert [row["prompt_id"] for row in rows] == ["a", "b"]
    assert rows[0]["prompt_media_refs"] == ["media-a"]
    assert "<image>" not in rows[0]["prompt"][0]["content"]
    assert rows[0]["prompt"][0]["content"].count(MARKER) == 1


def test_view_builder_marker_count_mismatch_raises(tmp_path):
    src = tmp_path / "source.parquet"
    _write_source(src, [_row("a", n_markers=2)])
    refs = tmp_path / "row_media_refs.parquet"
    _write_row_refs(refs, src, [["media-a"]])

    with pytest.raises(MarkerMismatch, match="markers vs"):
        build_alignment_views_from_row_refs(src, refs, tmp_path / "views.parquet", task="preference")


def test_view_builder_rejects_accidental_marker_in_response(tmp_path):
    src = tmp_path / "source.parquet"
    row = _row("a")
    row["accepted"][0]["content"] = f"sneaky {MARKER} text"
    _write_source(src, [row])
    refs = tmp_path / "row_media_refs.parquet"
    _write_row_refs(refs, src, [["media-a"]])

    with pytest.raises(MarkerMismatch, match="accidental"):
        build_alignment_views_from_row_refs(src, refs, tmp_path / "views.parquet", task="preference")


def test_view_builder_rejects_empty_accepted_or_rejected(tmp_path):
    src = tmp_path / "source.parquet"
    row = _row("a")
    row["accepted"] = []
    _write_source(src, [row])
    refs = tmp_path / "row_media_refs.parquet"
    _write_row_refs(refs, src, [["media-a"]])

    with pytest.raises(MarkerMismatch, match="empty accepted/rejected"):
        build_alignment_views_from_row_refs(src, refs, tmp_path / "views.parquet", task="preference")


def test_view_builder_rejects_system_role(tmp_path):
    src = tmp_path / "source.parquet"
    row = _row("a")
    row["prompt"].insert(0, {"role": "system", "content": "You are helpful."})
    _write_source(src, [row])
    refs = tmp_path / "row_media_refs.parquet"
    _write_row_refs(refs, src, [["media-a"]])

    with pytest.raises(MarkerMismatch, match="system"):
        build_alignment_views_from_row_refs(src, refs, tmp_path / "views.parquet", task="preference")


def _rl_row(sid, n_markers=1, *, system=None, answer="a cat", answer_variants=None):
    markers = "\n".join(["<image>"] * n_markers)
    prompt = [{"role": "user", "content": f"{markers}\nwhat is this?"}]
    if system is not None:
        prompt.insert(0, {"role": "system", "content": system})
    row = {
        "source-id": sid,
        "image": {"bytes": _png(32, 32), "path": f"{sid}.jpg"},
        "prompt": prompt,
        "answer": answer,
    }
    if answer_variants is not None:
        row["answer_variants"] = answer_variants
    return row


def test_view_builder_rl_prompt_keeps_answer_and_allows_system(tmp_path):
    src = tmp_path / "source.parquet"
    _write_source(src, [
        _rl_row("a", answer_variants=["a cat", "cat"]),
        _rl_row("b", system="Answer in \\boxed{}.", answer="42"),
    ])
    refs = tmp_path / "row_media_refs.parquet"
    _write_row_refs(refs, src, [["media-a"], ["media-b"]])

    out = tmp_path / "views" / "data.parquet"
    n_rows = build_alignment_views_from_row_refs(src, refs, out, task="rl_prompt", batch_size=1)

    rows = pq.read_table(out).to_pylist()
    assert n_rows == 2
    assert [row["prompt_id"] for row in rows] == ["a", "b"]
    assert rows[0]["answer"] == "a cat"
    assert rows[0]["answer_variants"] == ["a cat", "cat"]
    assert rows[0]["prompt_media_refs"] == ["media-a"]
    assert rows[0]["chosen_media_refs"] == [] and rows[0]["rejected_media_refs"] == []
    assert "<image>" not in rows[0]["prompt"][0]["content"]
    assert rows[0]["prompt"][0]["content"].count(MARKER) == 1
    # real RL data keeps a system instruction; the rl path must let it through
    assert [m["role"] for m in rows[1]["prompt"]] == ["system", "user"]
    assert rows[1]["prompt"][0]["content"] == "Answer in \\boxed{}."


def test_view_builder_unknown_task_fails_loud(tmp_path):
    src = tmp_path / "source.parquet"
    _write_source(src, [_row("a")])
    refs = tmp_path / "row_media_refs.parquet"
    _write_row_refs(refs, src, [["media-a"]])

    with pytest.raises(ValueError, match="unsupported alignment task"):
        build_alignment_views_from_row_refs(src, refs, tmp_path / "views.parquet", task="bandit")
