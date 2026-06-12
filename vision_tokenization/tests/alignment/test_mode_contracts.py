"""Posttraining mode contracts: entry validation + the publish completeness gate."""

import io
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from vision_tokenization.pipeline import run_distributed_pipeline
from vision_tokenization.pipeline.runtime.posttraining import (
    EncodeIncompleteError,
    check_encode_complete,
)


def test_posttraining_rejects_multi_image():
    with pytest.raises(ValueError, match="multi_image is meaningless"):
        run_distributed_pipeline({"mode": "posttraining", "multi_image": True})


def _png(color):
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buf, format="PNG")
    return buf.getvalue()


def test_posttraining_dry_run_runs_scan_and_reports_image_tokens(tmp_path):
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
        "mode": "posttraining", "dry_run": True, "task": "preference",
        "output_name": "dry_smoke", "output_dir": str(tmp_path / "out"),
        "input_parquet": str(src),
        "batch_size": 32, "max_batch_tokens": 32_768, "spatial_factor": 16,
        "tokenizer_min_pixels": 128 * 128, "tokenizer_max_pixels": 1400 * 1400,
        "window_size": 2000,
    })

    out = tmp_path / "out" / "preference" / "dry_smoke"  # task-keyed namespace
    assert (out / "scan.parquet").exists()
    assert (out / "dry_run_stats.json").exists()
    assert result["total_documents"] == 2  # one doc per unique media
    assert result["total_image_components"] == 2
    assert result["total_text_components"] == 0
    assert result["total_image_tokens"] > 0
    assert len(pq.read_table(out / "scan.parquet")) == 2


def _media(*ids):
    return [SimpleNamespace(media_id=i) for i in ids]


def test_encode_complete_passes_when_all_media_stored():
    check_encode_complete(_media("a", "b"), {"a": 10, "b": 7})


def test_encode_incomplete_raises_named_contract_error():
    with pytest.raises(EncodeIncompleteError, match=r"1 of 2 scanned media"):
        check_encode_complete(_media("a", "b"), {"a": 10})
