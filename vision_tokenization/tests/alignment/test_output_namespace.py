from vision_tokenization.pipeline import _build_output_subdir


def test_alignment_mode_namespaces_under_alignment():
    cfg = {"mode": "alignment", "task": "preference", "output_name": "mllm_dpo_smoke"}
    assert _build_output_subdir(cfg) == "alignment/mllm_dpo_smoke"


def test_alignment_namespace_does_not_depend_on_task_schema():
    cfg = {"mode": "alignment", "task": "rl_prompt", "output_name": "some_rl_set"}
    assert _build_output_subdir(cfg) == "alignment/some_rl_set"


def test_other_modes_stay_mode_keyed():
    assert _build_output_subdir({"mode": "sft", "output_name": "x"}) == "sft/x"
