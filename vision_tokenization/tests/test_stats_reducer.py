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


def test_stats_summary_uses_loop_elapsed_for_throughput_and_preserves_total_time():
    aggregate = stats_reducer.build_aggregate([
        {
            "rank": 0,
            "samples_processed": 10,
            "tokens_generated": 100,
            "image_tokens": 80,
            "elapsed_time": 2.0,
            "setup_elapsed_time": 8.0,
            "total_elapsed_time": 10.0,
            "tokenizer_load_time": 7.0,
            "model_load_time": 6.0,
            "text_tokenizer_load_time": 0.5,
        },
        {
            "rank": 1,
            "samples_processed": 20,
            "tokens_generated": 200,
            "image_tokens": 160,
            "elapsed_time": 4.0,
            "setup_elapsed_time": 6.0,
            "total_elapsed_time": 10.0,
            "tokenizer_load_time": 8.0,
            "model_load_time": 5.0,
            "text_tokenizer_load_time": 0.7,
        },
    ])

    assert aggregate["max_elapsed_s"] == 4.0
    assert aggregate["max_setup_elapsed_s"] == 8.0
    assert aggregate["max_total_elapsed_s"] == 10.0
    assert aggregate["max_tokenizer_load_time_s"] == 8.0
    assert aggregate["max_model_load_time_s"] == 6.0
    assert aggregate["max_text_tokenizer_load_time_s"] == 0.7
    assert aggregate["tokens_per_second"] == 300 / 4.0
    assert aggregate["image_tokens_per_second"] == 240 / 4.0
    assert aggregate["samples_per_second"] == 30 / 4.0
