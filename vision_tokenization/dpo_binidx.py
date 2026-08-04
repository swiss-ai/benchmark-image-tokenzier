#!/usr/bin/env python3
"""DPO binidx phase entry (post-merge, CPU).

Usage::

    python -m vision_tokenization.dpo_binidx \
        mode=alignment dataset=alignment/mmpr_v1_2

Reads the published store (after scan -> encode -> merge), builds the single-sequence
``[prompt|chosen|rejected]`` ``.bin/.idx`` + per-pair ``index``, registers them in
``manifest.json`` (schema 4), and retires the now-redundant deduped ``tokens/``.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    from vision_tokenization.pipeline import _build_output_subdir
    from vision_tokenization.pipeline.runtime.alignment import run_dpo_binidx

    if cfg.get("mode") != "alignment":
        raise ValueError(f"dpo_binidx requires mode=alignment, got {cfg.get('mode')!r}")

    resolved = OmegaConf.to_container(cfg, resolve=True)
    dataset_cfg = resolved.pop("dataset", {})
    pipeline_cfg = {**resolved, **dataset_cfg}
    pipeline_cfg.pop("tokenizer", None)
    pipeline_cfg["output_dir"] = str(
        Path(pipeline_cfg["output_dir"]) / _build_output_subdir(pipeline_cfg))

    section = run_dpo_binidx(pipeline_cfg)
    n = sum(s.get("n_pairs", s.get("n_samples", 0)) for s in section.get("splits", {}).values())
    print(f"alignment binidx: {n:,} docs -> {pipeline_cfg['output_dir']}")
    return section


if __name__ == "__main__":
    from vision_tokenization.tokenize import _preprocess_dataset_override

    _preprocess_dataset_override()
    main()
