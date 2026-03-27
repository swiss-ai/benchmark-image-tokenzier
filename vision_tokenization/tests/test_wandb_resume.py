import logging
import sys
import types

import pytest

from vision_tokenization.pipeline.runtime import checkpoint as checkpoint_mod
from vision_tokenization.pipeline.runtime import wandb_logger as wandb_mod


def _install_fake_wandb(monkeypatch):
    fake = types.ModuleType("wandb")
    fake.init_calls = []
    fake.log_calls = []
    fake.finish_calls = 0

    class FakeSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def init(**kwargs):
        fake.init_calls.append(kwargs)
        run_id = kwargs.get("id") or "generated-run"
        return types.SimpleNamespace(id=run_id)

    def log(payload, step=None):
        fake.log_calls.append((payload, step))

    def finish():
        fake.finish_calls += 1

    fake.Settings = FakeSettings
    fake.init = init
    fake.log = log
    fake.finish = finish
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def test_worker_stats_resume_preserves_elapsed_time(monkeypatch):
    current_time = {"value": 100.0}
    monkeypatch.setattr(checkpoint_mod.time, "time", lambda: current_time["value"])

    stats = checkpoint_mod.WorkerStats()
    stats.start_time = current_time["value"]
    stats.samples_processed = 12
    stats.tokens_generated = 120
    stats.image_tokens = 72

    current_time["value"] = 112.0
    snapshot = stats.to_dict()
    assert snapshot["elapsed_time"] == pytest.approx(12.0)
    assert snapshot["throughput"] == pytest.approx(10.0)
    assert snapshot["image_tokens_per_second"] == pytest.approx(6.0)

    current_time["value"] = 200.0
    resumed = checkpoint_mod.WorkerStats()
    resumed.start_time = current_time["value"]
    resumed.load_from_dict(snapshot)

    current_time["value"] = 208.0
    resumed_snapshot = resumed.to_dict()
    assert resumed.current_elapsed_time() == pytest.approx(20.0)
    assert resumed_snapshot["elapsed_time"] == pytest.approx(20.0)
    assert resumed_snapshot["throughput"] == pytest.approx(6.0)
    assert resumed_snapshot["image_tokens_per_second"] == pytest.approx(3.6)


def test_simple_wandb_logger_restores_step_and_uses_elapsed_seconds(monkeypatch):
    fake_wandb = _install_fake_wandb(monkeypatch)
    current_time = {"value": 50.0}
    monkeypatch.setattr(wandb_mod.time, "time", lambda: current_time["value"])

    logger = wandb_mod.SimpleWandbLogger(
        project="resume-test",
        run_id="resume-123",
        start_step=7,
        log_interval_seconds=1.0,
    )

    logger.log(
        samples=120,
        tokens=240,
        image_tokens=200,
        text_tokens=40,
        timing={"load_ms": 1.5},
        metrics={"batch/index": 42},
        elapsed_seconds=30.0,
        force=True,
    )
    logger.finish()

    assert fake_wandb.init_calls[0]["id"] == "resume-123"
    assert fake_wandb.log_calls[0][1] == 7
    payload = fake_wandb.log_calls[0][0]
    assert payload["elapsed_seconds"] == pytest.approx(30.0)
    assert payload["samples_per_second"] == pytest.approx(4.0)
    assert payload["tokens_per_second"] == pytest.approx(8.0)
    assert payload["image_tokens_per_second"] == pytest.approx(200 / 30.0)
    assert payload["timing/load_ms"] == pytest.approx(1.5)
    assert payload["batch/index"] == 42
    assert logger.state_dict() == {"run_id": "resume-123", "step": 8}
    assert fake_wandb.finish_calls == 1


def test_simple_wandb_logger_should_log_now_respects_interval(monkeypatch):
    fake_wandb = _install_fake_wandb(monkeypatch)
    current_time = {"value": 10.0}
    monkeypatch.setattr(wandb_mod.time, "time", lambda: current_time["value"])

    logger = wandb_mod.SimpleWandbLogger(
        project="resume-test",
        log_interval_seconds=5.0,
    )

    assert logger.should_log_now() is False

    current_time["value"] = 15.1
    assert logger.should_log_now() is True

    logger.log(samples=1, tokens=2, elapsed_seconds=1.0)
    assert fake_wandb.log_calls

    current_time["value"] = 16.0
    assert logger.should_log_now() is False


def test_load_wandb_resume_state_requires_checkpoint_metadata(caplog):
    assert wandb_mod.load_wandb_resume_state(False, {"wandb": {"run_id": "r", "step": 1}}) is None
    assert wandb_mod.load_wandb_resume_state(True, None) is None
    assert wandb_mod.load_wandb_resume_state(
        True,
        {"wandb": {"run_id": "resume-123", "step": "9"}},
    ) == {"run_id": "resume-123", "step": 9}

    with caplog.at_level(logging.WARNING):
        missing_step = wandb_mod.load_wandb_resume_state(
            True,
            {"wandb": {"run_id": "resume-123"}},
        )
    assert missing_step is None
    assert "missing W&B resume metadata" in caplog.text
