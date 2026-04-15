"""Tests for retry-on-low-CLIP behavior in CaptioningBenchmark."""

import json
from pathlib import Path
from typing import Iterable, List, Optional

import pytest

from vision_tokenization.qualitative_benchmark.benchmarks.captioning import CaptioningBenchmark


class _FakeVLM:
    """Returns prepared captions in order; records the seeds it was called with."""

    def __init__(self, captions: Iterable[str]):
        self._captions: List[str] = list(captions)
        self.seen_seeds: List[Optional[int]] = []
        self.preprocess_init_phrases: List[Optional[str]] = []
        self.preprocess_calls = 0

    def preprocess(self, image_path, init_phrase):
        self.preprocess_calls += 1
        self.preprocess_init_phrases.append(init_phrase)
        return f"<prompt for {image_path} init={init_phrase!r}>"

    def generate(self, prompt, debug=False, seed=None):
        self.seen_seeds.append(seed)
        if not self._captions:
            return ""
        return self._captions.pop(0)


def _build_benchmark(
    tmp_path: Path,
    captions: Iterable[str],
    clip_scores: Iterable[Optional[float]],
    *,
    threshold: Optional[float],
    max_attempts: int,
    base_seed: Optional[int] = None,
) -> CaptioningBenchmark:
    """Wire a CaptioningBenchmark with a fake VLM and stubbed CLIP scoring."""
    images_path = tmp_path / "images.json"
    images_path.write_text(json.dumps([{"path": "fake.png", "tags": ["test"]}]))

    fake_vlm = _FakeVLM(captions)
    # metrics=[] keeps construction hermetic: skips real CLIP model load
    # (which would download openai/clip-vit-base-patch32 on a fresh machine).
    bench = CaptioningBenchmark(
        images_config_path=str(images_path),
        vlm=fake_vlm,
        results_dir=str(tmp_path / "out"),
        debug=False,
        metrics=[],
        retry_clip_threshold=threshold,
        retry_max_attempts=max_attempts,
        retry_base_seed=base_seed,
    )
    # Inject a placeholder metric entry so retry-enabled gating sees clip_score.
    bench.metrics = {CaptioningBenchmark.CLIP_METRIC: object()}
    bench._retry_enabled = (
        threshold is not None
        and max_attempts > 1
        and CaptioningBenchmark.CLIP_METRIC in bench.metrics
    )

    score_iter = iter(clip_scores)

    def fake_compute_metrics(_data):
        try:
            score = next(score_iter)
        except StopIteration:
            score = None
        return {} if score is None else {CaptioningBenchmark.CLIP_METRIC: score}

    bench._compute_metrics = fake_compute_metrics  # type: ignore[assignment]
    bench._fake_vlm = fake_vlm                      # type: ignore[attr-defined]
    return bench


class TestRetryDisabled:
    def test_default_no_retry_means_one_attempt(self, tmp_path):
        bench = _build_benchmark(
            tmp_path,
            captions=["cat"],
            clip_scores=[0.05],
            threshold=None,
            max_attempts=1,
        )
        out = bench._generate_with_retries("fake.png", pil_image=object(), debug_this_sample=False)
        assert out["caption"] == "cat"
        assert out["init_phrase"] == ""
        assert out["metrics"] == {"clip_score": 0.05}
        assert out["retry_stats"] is None
        assert bench._fake_vlm.seen_seeds == [None]

    def test_threshold_set_but_max_attempts_one(self, tmp_path):
        # threshold alone with max_attempts=1 -> retries effectively disabled
        bench = _build_benchmark(
            tmp_path,
            captions=["cat"],
            clip_scores=[0.05],
            threshold=0.30,
            max_attempts=1,
        )
        out = bench._generate_with_retries("fake.png", pil_image=object(), debug_this_sample=False)
        assert out["retry_stats"] is None
        assert len(bench._fake_vlm.seen_seeds) == 1


class TestRetryEnabled:
    def test_first_attempt_passes_threshold_no_retry(self, tmp_path):
        bench = _build_benchmark(
            tmp_path,
            captions=["good"],
            clip_scores=[0.50],
            threshold=0.30,
            max_attempts=4,
            base_seed=42,
        )
        out = bench._generate_with_retries("fake.png", pil_image=object(), debug_this_sample=False)
        assert out["caption"] == "good"
        assert out["init_phrase"] == ""
        assert out["retry_stats"]["num_attempts"] == 1
        assert out["retry_stats"]["attempts"] == [
            {"attempt": 0, "seed": 42, "clip_score": 0.50, "init_phrase": ""},
        ]
        assert bench._fake_vlm.seen_seeds == [42]

    def test_retry_until_threshold_met(self, tmp_path):
        bench = _build_benchmark(
            tmp_path,
            captions=["bad", "meh", "fine"],
            clip_scores=[0.10, 0.15, 0.32],
            threshold=0.30,
            max_attempts=4,
            base_seed=42,
        )
        out = bench._generate_with_retries("fake.png", pil_image=object(), debug_this_sample=False)
        assert out["caption"] == "fine"
        assert out["init_phrase"] == ""
        assert out["metrics"] == {"clip_score": 0.32}
        stats = out["retry_stats"]
        assert stats["num_attempts"] == 3
        assert [a["seed"] for a in stats["attempts"]] == [42, 43, 44]
        assert [a["clip_score"] for a in stats["attempts"]] == [0.10, 0.15, 0.32]
        assert [a["init_phrase"] for a in stats["attempts"]] == ["", "", ""]

    def test_budget_exhausted_keeps_last(self, tmp_path):
        bench = _build_benchmark(
            tmp_path,
            captions=["a", "b", "c"],
            clip_scores=[0.10, 0.12, 0.14],
            threshold=0.30,
            max_attempts=3,
        )
        out = bench._generate_with_retries("fake.png", pil_image=object(), debug_this_sample=False)
        assert out["caption"] == "c"  # last attempt kept; no best-of-N
        assert out["init_phrase"] == "The image shows "
        assert out["metrics"] == {"clip_score": 0.14}
        assert out["retry_stats"]["num_attempts"] == 3
        assert [a["init_phrase"] for a in out["retry_stats"]["attempts"]] == [
            "",
            "",
            "The image shows ",
        ]

    def test_clip_returns_none_stops_retries(self, tmp_path):
        bench = _build_benchmark(
            tmp_path,
            captions=["x", "y"],
            clip_scores=[None, 0.50],
            threshold=0.30,
            max_attempts=4,
        )
        out = bench._generate_with_retries("fake.png", pil_image=object(), debug_this_sample=False)
        # First attempt scored None -> we stop; never reaches the second caption.
        assert out["caption"] == "x"
        assert out["init_phrase"] == ""
        assert out["retry_stats"]["num_attempts"] == 1
        assert bench._fake_vlm.seen_seeds == [None]

    def test_seeds_default_to_none_when_base_seed_unset(self, tmp_path):
        bench = _build_benchmark(
            tmp_path,
            captions=["a", "b"],
            clip_scores=[0.10, 0.40],
            threshold=0.30,
            max_attempts=3,
            base_seed=None,
        )
        out = bench._generate_with_retries("fake.png", pil_image=object(), debug_this_sample=False)
        assert out["caption"] == "b"
        assert out["init_phrase"] == ""
        assert bench._fake_vlm.seen_seeds == [None, None]

    def test_preprocess_called_once_per_image(self, tmp_path):
        bench = _build_benchmark(
            tmp_path,
            captions=["a", "b", "c"],
            clip_scores=[0.10, 0.12, 0.40],
            threshold=0.30,
            max_attempts=4,
            base_seed=7,
        )
        bench._generate_with_retries("fake.png", pil_image=object(), debug_this_sample=False)
        # Vision tokenization is the expensive step — it must not be repeated.
        assert bench._fake_vlm.preprocess_calls == 1

    def test_last_retry_uses_fallback_init_phrase(self, tmp_path):
        bench = _build_benchmark(
            tmp_path,
            captions=["bad", "still bad", "rescued"],
            clip_scores=[0.10, 0.12, 0.35],
            threshold=0.30,
            max_attempts=3,
            base_seed=9,
        )
        out = bench._generate_with_retries("fake.png", pil_image=object(), debug_this_sample=False)
        assert out["caption"] == "rescued"
        assert out["init_phrase"] == "The image shows "
        assert bench._fake_vlm.preprocess_calls == 2
        assert bench._fake_vlm.preprocess_init_phrases == ["", "The image shows "]
        assert [a["init_phrase"] for a in out["retry_stats"]["attempts"]] == [
            "",
            "",
            "The image shows ",
        ]
