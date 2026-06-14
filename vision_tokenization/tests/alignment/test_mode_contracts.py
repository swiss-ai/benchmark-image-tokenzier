"""Alignment mode contracts: entry validation and scan-backed dry runs."""

import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from vision_tokenization.pipeline import run_distributed_pipeline
from vision_tokenization.indexing.alignment.payload import split_payload_rows


def test_alignment_rejects_multi_image():
    with pytest.raises(ValueError, match="multi_image is meaningless"):
        run_distributed_pipeline({"mode": "alignment", "multi_image": True})


def test_alignment_rejects_high_error_tolerance():
    from vision_tokenization.pipeline.runtime.alignment import run_alignment

    with pytest.raises(ValueError, match="max_consecutive_errors"):
        run_alignment({"resume": False, "max_consecutive_errors": 50})


def _png(color):
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buf, format="PNG")
    return buf.getvalue()


def test_alignment_dry_run_runs_scan_and_reports_image_tokens(tmp_path):
    """Dry run = the real scan stage (CPU) + plan-derived token counts, no GPU.

    The persisted scan.parquet is the same artifact the GPU job's scan stage
    writes (single writer: run_scan_stage), so dry run doubles as pre-flight.
    """
    rows = [{
        "source-id": f"t-{i}",
        "image": [{"bytes": _png(c), "path": f"{i}.png"}],
        "prompt": [{"role": "user", "content": "<image>\nWhich color?"}],
        "accepted": [{"role": "assistant", "content": c}],
        "rejected": [{"role": "assistant", "content": "mauve"}],
    } for i, c in enumerate(["red", "blue"])]
    src = tmp_path / "in.parquet"
    pq.write_table(pa.Table.from_pylist(rows), src)

    result = run_distributed_pipeline({
        "mode": "alignment", "dry_run": True, "task": "preference",
        "output_name": "dry_smoke", "output_dir": str(tmp_path / "out"),
        "input_parquet": str(src),
        "batch_size": 32, "max_batch_tokens": 32_768, "spatial_factor": 16,
        "tokenizer_min_pixels": 128 * 128, "tokenizer_max_pixels": 1400 * 1400,
        "window_size": 2000,
    })

    out = tmp_path / "out" / "alignment" / "dry_smoke"
    # Dry run leaves only dry_run_stats.json public; scan artifacts stay in _work.
    assert (out / "dry_run_stats.json").exists()
    assert not (out / "scan.parquet").exists()
    assert not (out / "_work").exists()
    assert result["total_documents"] == 2  # one doc per unique media (2 scanned)
    assert result["total_image_components"] == 2
    assert result["total_text_components"] == 0
    assert result["total_image_tokens"] > 0


def _view_row(prompt_id: str, tokens: int = 10):
    return {
        "prompt": [{"role": "user", "content": "<|image|>\nWhich color?"}],
        "chosen": "good",
        "rejected": "bad",
        "prompt_media_refs": [f"media-{prompt_id}"],
        "chosen_media_refs": [],
        "rejected_media_refs": [],
        "prompt_id": prompt_id,
        "media_tokens_total": tokens,
        "text_chars": 24,
    }


def test_split_payload_rows_is_deterministic_and_keeps_prompt_groups_together():
    rows = [
        _view_row("shared", 10),
        _view_row("a", 11),
        _view_row("shared", 12),
        _view_row("b", 13),
        _view_row("c", 14),
    ]

    train_a, val_a = split_payload_rows(rows, requested_validation_rows=2)
    train_b, val_b = split_payload_rows(rows, requested_validation_rows=2)

    assert [r["prompt_id"] for r in train_a] == [r["prompt_id"] for r in train_b]
    assert [r["prompt_id"] for r in val_a] == [r["prompt_id"] for r in val_b]

    shared_locations = {
        "train": [r["prompt_id"] for r in train_a].count("shared"),
        "validation": [r["prompt_id"] for r in val_a].count("shared"),
    }
    assert shared_locations in (
        {"train": 2, "validation": 0},
        {"train": 0, "validation": 2},
    )
    assert len(train_a) + len(val_a) == len(rows)
