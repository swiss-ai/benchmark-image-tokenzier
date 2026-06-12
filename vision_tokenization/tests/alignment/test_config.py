from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _compose_alignment_cfg(dataset: str, extra_overrides: list[str] | None = None):
    overrides = [f"dataset={dataset}", "mode=alignment", "num_gpus=1"]
    if extra_overrides:
        overrides.extend(extra_overrides)
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=overrides)
    GlobalHydra.instance().clear()
    return cfg


def test_alignment_smoke_config_composes_and_resolves():
    cfg = _compose_alignment_cfg(
        "alignment/mllm_dpo_smoke",
        extra_overrides=[
            "dataset.output_dir=/tmp/out",
            "dataset.input_parquet=/tmp/in.parquet",
        ],
    )
    # manifest_path/plan keys from the _pipeline fragment stay ??? by design
    # (never read once the alignment branch short-circuits), so resolve without
    # throwing on those unused mandatory values.
    OmegaConf.to_container(cfg, resolve=True)

    assert cfg.dataset.task == "preference"
    assert cfg.dataset.encode_batch_size == 32
    assert cfg.dataset.val_rows == 256
    assert cfg.dataset.min_pixels == "128*128"
    assert cfg.dataset.max_pixels == "1400*1400"
    assert str(cfg.dataset.tokenizer_path).endswith(
        "apertus_emu3.5_wavtok_instruct_thinking_token_fixed"
    )
    # Alignment is not storage-backed: no dataset_type, no _storage fragment.
    assert OmegaConf.select(cfg, "dataset.dataset_type") is None


def test_alignment_real_config_carries_capstor_paths():
    cfg = _compose_alignment_cfg("alignment/mllm_dpo")
    # manifest_path/plan keys from the _pipeline fragment stay ??? by design
    # (never read once the alignment branch short-circuits), so resolve without
    # throwing on those unused mandatory values.
    OmegaConf.to_container(cfg, resolve=True)

    assert cfg.dataset.output_name == "mllm_dpo"
    assert str(cfg.dataset.input_parquet).endswith("alignment-processed/mllm-dpo.parquet")
    assert cfg.dataset.task == "preference"
