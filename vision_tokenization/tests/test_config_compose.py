from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_DATASET_DIR = _CONFIG_DIR / "dataset"


def _compose_dataset_cfg(dataset: str, mode: str, extra_overrides: list[str] | None = None):
    overrides = [f"dataset={dataset}", f"mode={mode}", "num_gpus=1"]
    if extra_overrides:
        overrides.extend(extra_overrides)

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=overrides)
    GlobalHydra.instance().clear()
    return cfg


def _assert_resolves(cfg) -> None:
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def _all_dataset_choices() -> list[str]:
    return sorted(
        str(path.relative_to(_DATASET_DIR)).replace("\\", "/")[:-5]
        for path in _DATASET_DIR.rglob("*.yaml")
        if not any(part.startswith("_") for part in path.relative_to(_DATASET_DIR).parts)
        # `alignment` mode reads `input_parquet` directly (no storage backend),
        # so it has no `dataset_type`; it is covered by tests/alignment/test_config.py.
        and path.relative_to(_DATASET_DIR).parts[0] != "alignment"
    )


@pytest.mark.parametrize("dataset", _all_dataset_choices())
def test_all_dataset_configs_compose_with_storage_only_dataset_types(dataset: str):
    mode = dataset.split("/", 1)[0]

    cfg = _compose_dataset_cfg(dataset, mode)

    assert cfg.dataset.dataset_type in {"hf", "jsonl_tar", "wds"}
    assert OmegaConf.select(cfg, "dataset._storage") is None
    assert OmegaConf.select(cfg, "dataset._task") is None


@pytest.mark.parametrize(
    ("dataset", "mode", "expected_dataset_type"),
    [
        ("sft/path_vqa", "sft", "hf"),
        ("sft/tcm_shizhen_vision", "sft", "jsonl_tar"),
        ("interleave/molmo_syn_multiimage", "interleave", "hf"),
        ("interleave/shizhen_web_vision", "interleave", "jsonl_tar"),
        ("image2text/commoncatalog_recap", "image2text", "wds"),
    ],
)
def test_representative_dataset_configs_compose_and_resolve(
    dataset: str,
    mode: str,
    expected_dataset_type: str,
):
    cfg = _compose_dataset_cfg(dataset, mode)

    _assert_resolves(cfg)
    assert cfg.dataset.dataset_type == expected_dataset_type


def test_jsonl_tar_sft_config_does_not_require_input_pattern():
    cfg = _compose_dataset_cfg("sft/tcm_shizhen_vision", "sft")

    _assert_resolves(cfg)
    assert cfg.dataset.dataset_type == "jsonl_tar"
    assert OmegaConf.select(cfg, "dataset.input_pattern") is None
    assert cfg.dataset.image_field == "image"
    assert cfg.dataset.text_column == "conversations"


def test_sft_datasets_keep_instruct_tokenizer_override():
    for dataset in ("sft/path_vqa", "sft/tcm_shizhen_vision"):
        cfg = _compose_dataset_cfg(dataset, "sft")
        _assert_resolves(cfg)
        assert str(cfg.dataset.tokenizer_path).endswith(
            "apertus_emu3.5_wavtok_instruct_thinking_token_fixed"
        )


def test_non_sft_dataset_does_not_define_dataset_tokenizer_override():
    cfg = _compose_dataset_cfg(
        "image_only/llava85m_midtrain",
        "image_only",
        extra_overrides=[
            "dataset.output_dir=/tmp/out",
            "dataset.manifest_path=/tmp/manifest.parquet",
            "dataset.input_pattern=/tmp/input",
        ],
    )

    _assert_resolves(cfg)
    assert cfg.dataset.dataset_type == "hf"
    assert OmegaConf.select(cfg, "dataset.tokenizer_path") is None


def test_wds_interleave_config_keeps_text_and_parser_fields():
    cfg = _compose_dataset_cfg("interleave/medpix", "interleave")

    _assert_resolves(cfg)
    assert cfg.dataset.dataset_type == "wds"
    assert cfg.dataset.text_column == "txt"
    assert cfg.dataset.parser == "medpix"
    assert cfg.dataset.document_field == "txt"
    assert cfg.dataset.image_field_pattern == "img"
