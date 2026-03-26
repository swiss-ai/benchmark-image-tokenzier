#!/usr/bin/env python3
"""Main entry point for vision tokenization pipeline.

Usage::

    python -m vision_tokenization.tokenize \
        mode=image2text dataset=pmc_oa num_gpus=4

    python -m vision_tokenization.tokenize \
        mode=interleave dataset=pin_subset num_gpus=4
"""

# Avoid thread oversubscription with many dataloader workers
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Reduce CUDA memory fragmentation with expandable virtual-memory segments.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import logging
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

_VALID_MODES = {"image_only", "sft", "image2text", "text2image", "interleave"}


def _preprocess_dataset_override():
    """Allow ``mode=X dataset=Y`` shorthand for ``dataset=X/Y``.

    Rewrites sys.argv before Hydra parses it so that the Hydra config group
    resolves to ``configs/dataset/{mode}/{dataset}.yaml``.
    """
    mode_val = None
    dataset_val = None
    dataset_idx = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith("mode="):
            mode_val = arg.split("=", 1)[1]
        elif arg.startswith("dataset="):
            dataset_val = arg.split("=", 1)[1]
            dataset_idx = i
    if dataset_idx is not None and dataset_val and mode_val:
        if "/" not in dataset_val:
            sys.argv[dataset_idx] = f"dataset={mode_val}/{dataset_val}"
        elif not dataset_val.startswith(f"{mode_val}/"):
            raise ValueError(
                f"mode={mode_val} conflicts with dataset={dataset_val}. "
                f"Use: mode={mode_val} dataset={dataset_val.split('/', 1)[1]}"
            )


def _resolve_mode(cfg: DictConfig) -> None:
    """Set cfg.mode from the resolved dataset path.

    After ``_preprocess_dataset_override`` rewrites ``dataset=X/Y`` →
    ``dataset={mode}/Y``, the Hydra runtime choice for dataset contains
    the mode as the first path segment.  This function reads it and sets
    ``cfg.mode`` accordingly.
    """
    mode = cfg.get("mode")
    if mode is None:
        raise ValueError(
            "mode is required. Use: mode=image_only | sft | image2text | text2image | interleave"
        )
    if mode not in _VALID_MODES:
        raise ValueError(
            f"mode={mode!r} is not valid. Expected one of: {sorted(_VALID_MODES)}"
        )


def _resolve_output_format(mode: str, dataset_cfg) -> str:
    """Resolve output format, forcing pooled for interleave and multi-image jobs."""
    requested = dataset_cfg.get("output_format")
    multi_image = dataset_cfg.get("multi_image")
    inferred_multi_image = (
        bool(multi_image)
        if multi_image is not None
        else (
            dataset_cfg.get("image_list_column") is not None
            or dataset_cfg.get("dataset_type") == "jsonl_tar_interleave"
        )
    )

    if mode == "interleave" or inferred_multi_image:
        if requested not in (None, "pooled"):
            logger.warning(
                "Forcing output_format=pooled for mode=%s multi_image=%s "
                "(requested %r)",
                mode,
                inferred_multi_image,
                requested,
            )
        return "pooled"

    return requested or "direct"


def _as_dict(section) -> dict:
    """Normalize a possibly-missing OmegaConf/container section to a plain dict."""
    return dict(section or {})


def _merge_pipeline_sections(
    root_cfg: dict,
    dataset_cfg: dict,
    *,
    output_format: str,
) -> dict:
    """Flatten active direct/pooled subsections into one pipeline config.
    """
    pipeline_cfg = dict(root_cfg)

    root_direct = _as_dict(pipeline_cfg.pop("direct", None))
    root_pooled = _as_dict(pipeline_cfg.pop("pooled", None))

    dataset_cfg = dict(dataset_cfg)
    dataset_direct = _as_dict(dataset_cfg.pop("direct", None))
    dataset_pooled = _as_dict(dataset_cfg.pop("pooled", None))

    if output_format == "pooled":
        pipeline_cfg.update(root_pooled)
    else:
        pipeline_cfg.update(root_direct)

    pipeline_cfg.update(dataset_cfg)

    if output_format == "pooled":
        pipeline_cfg.update(dataset_pooled)
    else:
        pipeline_cfg.update(dataset_direct)

    return pipeline_cfg


def _validate_pipeline_cfg(pipeline_cfg: dict, *, output_format: str) -> None:
    """Reject incompatible config surfaces for the active pipeline."""
    if output_format != "pooled":
        return

    legacy_pooled_keys = [
        key
        for key in (
            "chunk_docs",
            "checkpoint_every_chunks",
            "shard_rollover_chunks",
            "checkpoint_interval_docs",
            "checkpoint_interval_batches",
        )
        if pipeline_cfg.get(key) is not None
    ]
    if legacy_pooled_keys:
        raise ValueError(
            "Pooled mode no longer accepts legacy chunk-based keys: "
            + ", ".join(sorted(legacy_pooled_keys))
        )

    if pipeline_cfg.get("batch_plan") is not None:
        raise ValueError(
            "Pooled mode uses pooled.document_plan_path, not batch_plan."
        )

    missing_keys = [
        key
        for key in (
            "document_window_docs",
            "checkpoint_every_windows",
            "spill_shard_rollover_windows",
        )
        if pipeline_cfg.get(key) is None
    ]
    if missing_keys:
        raise ValueError(
            "Pooled mode is missing required config keys: "
            + ", ".join(sorted(missing_keys))
        )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Print config only on rank 0
    if int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0))) == 0:
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    _resolve_mode(cfg)
    output_format = _resolve_output_format(cfg.mode, cfg.dataset)

    from vision_tokenization.utils.parse_utils import parse_resolution

    tokenizer_cfg = cfg.tokenizer
    tokenizer_path = tokenizer_cfg.path
    tokenizer_min_pixels = parse_resolution(str(tokenizer_cfg.min_pixels))["pixels"]
    tokenizer_max_pixels = parse_resolution(str(tokenizer_cfg.max_pixels))["pixels"]
    tokenizer_kwargs = {
        "torch_compile": tokenizer_cfg.get("torch_compile", False),
        "torch_compile_mode": tokenizer_cfg.get("torch_compile_mode", "reduce-overhead"),
    }
    max_sequence_tokens = cfg.dataset.get("max_sequence_tokens")
    if max_sequence_tokens is not None:
        tokenizer_kwargs["max_sequence_tokens"] = int(max_sequence_tokens)

    # Dataset-level pixel bounds for batch-planner filtering.
    # Format: "H*W" string (e.g. "64*128") or plain integer.
    filter_min_pixels = parse_resolution(str(cfg.dataset.min_pixels))["pixels"]
    filter_max_pixels = parse_resolution(str(cfg.dataset.max_pixels))["pixels"]

    from vision_tokenization.pipeline import run_distributed_pipeline

    # Flatten Hydra config into a plain dict. Pipeline-specific sections are
    # resolved after output_format is known so that only the active one applies.
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    dataset_cfg = resolved_cfg.pop("dataset", {})
    pipeline_cfg = _merge_pipeline_sections(
        resolved_cfg,
        dataset_cfg,
        output_format=output_format,
    )
    _validate_pipeline_cfg(pipeline_cfg, output_format=output_format)

    # Resolve tokenizer fields
    tokenizer_cfg_resolved = pipeline_cfg.pop("tokenizer", {})
    pipeline_cfg["tokenizer_path"] = tokenizer_path
    pipeline_cfg["tokenizer_min_pixels"] = tokenizer_min_pixels
    pipeline_cfg["tokenizer_max_pixels"] = tokenizer_max_pixels
    pipeline_cfg["max_encode_pixels"] = tokenizer_cfg_resolved.get("max_encode_pixels")
    pipeline_cfg["filter_min_pixels"] = filter_min_pixels
    pipeline_cfg["filter_max_pixels"] = filter_max_pixels
    pipeline_cfg["output_format"] = output_format

    # Bridge parser -> document_format for direct pipeline backward compat
    if not pipeline_cfg.get("document_format"):
        _parser_to_format = {
            "pin200m": "pin_markdown", "shizhen": "content_array",
            "medpix": "medpix",
        }
        pipeline_cfg["document_format"] = _parser_to_format.get(
            pipeline_cfg.get("parser", "auto")
        )
    pipeline_cfg["tokenizer_kwargs"] = tokenizer_kwargs

    # Conversation policy for SFT mode
    conv_policy = pipeline_cfg.get("conversation_policy")
    if conv_policy is not None:
        from vision_tokenization.discrete.conversation import ConversationPolicy

        pipeline_cfg["tokenizer_kwargs"]["conversation_policy"] = ConversationPolicy(
            **(conv_policy if isinstance(conv_policy, dict) else {})
        )

    result = run_distributed_pipeline(pipeline_cfg)

    logger.info("Pipeline completed!")
    logger.info(f"Total processed: {result.get('samples_processed', 0)}")
    logger.info(f"Total tokens: {result.get('tokens_generated', 0)}")
    logger.info(f"Output directory: {result.get('output_dir', cfg.dataset.output_dir)}")

    return result


if __name__ == "__main__":
    _preprocess_dataset_override()
    main()
