import pytest

from vision_tokenization.pipeline import _build_output_subdir


def test_preference_task_namespaces_under_preference():
    cfg = {"mode": "posttraining", "task": "preference", "output_name": "mllm_dpo_smoke"}
    assert _build_output_subdir(cfg) == "preference/mllm_dpo_smoke"


def test_rl_prompt_task_namespaces_under_rl():
    cfg = {"mode": "posttraining", "task": "rl_prompt", "output_name": "some_rl_set"}
    assert _build_output_subdir(cfg) == "rl/some_rl_set"


def test_unknown_task_fails_loud():
    cfg = {"mode": "posttraining", "task": "sft", "output_name": "x"}
    with pytest.raises(ValueError, match="unknown task"):
        _build_output_subdir(cfg)


def test_other_modes_stay_mode_keyed():
    assert _build_output_subdir({"mode": "sft", "output_name": "x"}) == "sft/x"
