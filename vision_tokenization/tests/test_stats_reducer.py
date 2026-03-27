import json

from vision_tokenization.pipeline.output import stats_reducer


def _append_stats(stats_path, payload):
    with open(stats_path, "a") as f:
        f.write(json.dumps(payload) + "\n")


def test_maybe_write_stats_summary_waits_for_all_ranks(tmp_path):
    stats_path = tmp_path / "stats.jsonl"
    _append_stats(
        stats_path,
        {"rank": 0, "samples_processed": 10, "tokens_generated": 100, "image_tokens": 60, "elapsed_time": 5.0},
    )

    aggregate = stats_reducer.maybe_write_stats_summary(tmp_path, expected_ranks=2)

    assert aggregate is None
    assert not (tmp_path / "stats_summary.json").exists()


def test_recompute_stats_summary_uses_latest_entry_per_rank(tmp_path):
    stats_path = tmp_path / "stats.jsonl"
    _append_stats(
        stats_path,
        {"rank": 0, "samples_processed": 10, "tokens_generated": 100, "image_tokens": 60, "elapsed_time": 5.0},
    )
    _append_stats(
        stats_path,
        {"rank": 1, "samples_processed": 7, "tokens_generated": 70, "image_tokens": 35, "elapsed_time": 4.0},
    )
    _append_stats(
        stats_path,
        {"rank": 0, "samples_processed": 12, "tokens_generated": 120, "image_tokens": 72, "elapsed_time": 6.0},
    )

    aggregate = stats_reducer.recompute_stats_summary(
        tmp_path,
        expected_ranks=2,
        require_complete=True,
    )

    assert aggregate is not None
    assert aggregate["num_ranks"] == 2
    assert aggregate["samples_processed"] == 19
    assert aggregate["tokens_generated"] == 190
    assert aggregate["image_tokens"] == 107
    assert aggregate["max_elapsed_s"] == 6.0
    assert aggregate["samples_per_second"] == 19 / 6.0
    assert aggregate["image_tokens_per_second"] == 107 / 6.0
    assert aggregate["per_rank"][0]["samples_processed"] == 12

    summary = json.loads((tmp_path / "stats_summary.json").read_text())
    assert summary["tokens_per_second"] == 190 / 6.0
    assert summary["image_tokens_per_second"] == 107 / 6.0
