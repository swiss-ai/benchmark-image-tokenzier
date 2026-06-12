from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _compose_alignment_cfg(dataset: str, extra_overrides: list[str] | None = None):
    overrides = [f"dataset={dataset}", "mode=posttraining", "num_gpus=1"]
    if extra_overrides:
        overrides.extend(extra_overrides)
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=overrides)
    GlobalHydra.instance().clear()
    return cfg


def test_alignment_smoke_config_composes_and_resolves():
    cfg = _compose_alignment_cfg(
        "posttraining/mllm_dpo_smoke",
        extra_overrides=[
            "dataset.output_dir=/tmp/out",
            "dataset.input_parquet=/tmp/in.parquet",
        ],
    )
    # manifest_path stays ??? by design (the scan stage injects scan.parquet
    # at runtime), so resolve without throwing on missing mandatory values.
    OmegaConf.to_container(cfg, resolve=True)

    assert cfg.dataset.task == "preference"
    # The unified executor's batch-size cap (max_batch_tokens also applies).
    assert cfg.dataset.batch_size == 32
    assert cfg.dataset.val_rows == 256
    # The recorded resize band is the tokenizer's (config.yaml); the task
    # fragment carries no dataset pixel filter (build_plan ignores it).
    assert cfg.tokenizer.min_pixels == "128*128"
    assert cfg.tokenizer.max_pixels == "1400*1400"
    assert str(cfg.dataset.tokenizer_path).endswith(
        "apertus_emu3.5_wavtok_instruct_thinking_token_fixed"
    )
    # Alignment is not storage-backed: no dataset_type, no _storage fragment.
    assert OmegaConf.select(cfg, "dataset.dataset_type") is None


def test_alignment_real_config_carries_capstor_paths():
    cfg = _compose_alignment_cfg("posttraining/mllm_dpo")
    # manifest_path stays ??? by design (the scan stage injects scan.parquet
    # at runtime), so resolve without throwing on missing mandatory values.
    OmegaConf.to_container(cfg, resolve=True)

    assert cfg.dataset.output_name == "mllm_dpo"
    assert str(cfg.dataset.output_dir).endswith("vision-datasets/tokenized")
    assert str(cfg.dataset.input_parquet).endswith("alignment-processed/mllm-dpo.parquet")
    assert cfg.dataset.task == "preference"
