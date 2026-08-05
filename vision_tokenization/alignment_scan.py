#!/usr/bin/env python3
"""Alignment SCAN phase entry (inline, CPU, torch-free).

Usage::

    python -m vision_tokenization.alignment_scan \
        mode=alignment dataset=alignment/mmpr_v1_2

Builds the pre-encode scan artifacts (``scan.parquet``, ``media_unique.parquet``,
``views.raw.parquet``, ``publish_meta.json``) that the GPU encode
(``tokenize mode=alignment``) and the inline merge
(``python -m vision_tokenization.pipeline.output.alignment_merge``) consume.
Runs on the head node — no torch, no GPU. Mirrors the sft/interleave contract of
a pre-built scan the GPU job reads; for alignment the global content-dedup must
precede encode, so the scan is its own explicit step.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    from vision_tokenization.pipeline import _build_output_subdir
    from vision_tokenization.pipeline.runtime.alignment import run_alignment_scan
    from vision_tokenization.utils.parse_utils import parse_resolution

    if cfg.get("mode") != "alignment":
        raise ValueError(f"alignment_scan requires mode=alignment, got {cfg.get('mode')!r}")

    tok = cfg.tokenizer
    resolved = OmegaConf.to_container(cfg, resolve=True)
    dataset_cfg = resolved.pop("dataset", {})
    pipeline_cfg = {**resolved, **dataset_cfg}
    pipeline_cfg.pop("tokenizer", None)
    pipeline_cfg["tokenizer_path"] = pipeline_cfg.get("tokenizer_path", tok.path)
    pipeline_cfg["tokenizer_min_pixels"] = parse_resolution(str(tok.min_pixels))["pixels"]
    pipeline_cfg["tokenizer_max_pixels"] = parse_resolution(str(tok.max_pixels))["pixels"]
    pipeline_cfg["output_dir"] = str(
        Path(pipeline_cfg["output_dir"]) / _build_output_subdir(pipeline_cfg))

    result = run_alignment_scan(pipeline_cfg)
    print(f"alignment scan: {result['n_pairs']:,} pairs, "
          f"{result['n_unique_media']:,} unique media "
          f"({result['n_skipped_media']} skipped) -> {result['output_dir']}")
    return result


if __name__ == "__main__":
    from vision_tokenization.tokenize import _preprocess_dataset_override

    _preprocess_dataset_override()
    main()
