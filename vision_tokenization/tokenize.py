#!/usr/bin/env python3
"""Main entry point for vision tokenization pipeline.

Usage::

    python -m vision_tokenization.tokenize \
        dataset=image_only/llava85m_midtrain num_gpus=4

    python -m vision_tokenization.tokenize \
        dataset=sft/llava_onevision_sft num_gpus=16 mode=sft
"""

# Avoid thread oversubscription with many dataloader workers
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _validate_mode_from_dataset_folder(cfg: DictConfig) -> None:
    """Fail fast if dataset folder semantics conflict with selected mode.

    If dataset is chosen from a nested Hydra group like ``dataset=sft/foo``,
    we treat the first segment as the expected mode and verify it matches
    ``cfg.mode``.
    """
    mode = cfg.get("mode", "image_only")

    dataset_choice = None
    try:
        dataset_choice = HydraConfig.get().runtime.choices.get("dataset")
    except Exception:
        return

    if not dataset_choice or "/" not in dataset_choice:
        return

    folder_mode = dataset_choice.split("/", 1)[0]
    valid_modes = {"image_only", "sft", "image2text", "text2image"}
    if folder_mode not in valid_modes:
        return

    if mode != folder_mode:
        raise ValueError(
            f"Mode mismatch: dataset={dataset_choice!r} implies mode={folder_mode!r} "
            f"from folder name, but cfg.mode={mode!r}. "
            f"Set mode={folder_mode} or pick a dataset under {mode}/."
        )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Print config only on rank 0
    if int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0))) == 0:
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    _validate_mode_from_dataset_folder(cfg)

    from vision_tokenization.utils.parse_utils import parse_resolution

    tokenizer_cfg = cfg.tokenizer
    tokenizer_path = tokenizer_cfg.path
    tokenizer_min_pixels = parse_resolution(str(tokenizer_cfg.min_pixels))["pixels"]
    tokenizer_max_pixels = parse_resolution(str(tokenizer_cfg.max_pixels))["pixels"]

    # Dataset-level pixel bounds for batch-planner filtering.
    # Format: "H*W" string (e.g. "64*128") or plain integer.
    filter_min_pixels = parse_resolution(str(cfg.dataset.min_pixels))["pixels"]
    filter_max_pixels = parse_resolution(str(cfg.dataset.max_pixels))["pixels"]

    from vision_tokenization.pipelines.distributed import run_distributed_pipeline

    # Build flat config dict for the pipeline
    pipeline_cfg = {
        "tokenizer_path": tokenizer_path,
        "tokenizer_min_pixels": tokenizer_min_pixels,
        "tokenizer_max_pixels": tokenizer_max_pixels,
        "max_images_per_encode": tokenizer_cfg.get("max_images_per_encode"),
        "filter_min_pixels": filter_min_pixels,
        "filter_max_pixels": filter_max_pixels,
        "output_dir": cfg.output_dir,
        "num_gpus": cfg.get("num_gpus"),
        "mode": cfg.get("mode", "image_only"),
        "resume": cfg.get("resume", False),
        "dry_run": cfg.get("dry_run", False),
        # Dataset config (flattened from dataset group)
        "dataset_type": cfg.dataset.get("dataset_type", "hf"),
        "output_name": cfg.dataset.get("output_name"),
        "manifest_path": cfg.dataset.get("manifest_path"),
        "arrow_dir": cfg.dataset.get("arrow_dir"),
        "image_column": cfg.dataset.get("image_column", "image"),
        "text_column": cfg.dataset.get("text_column"),
        # Batch planning
        "batch_plan_path": cfg.dataset.get("batch_plan_path"),
        "batch_size": cfg.dataset.get("batch_size"),
        "max_batch_tokens": cfg.dataset.get("max_batch_tokens"),
        "spatial_factor": cfg.dataset.get("spatial_factor", 16),
        "num_clusters": cfg.dataset.get("num_clusters", 2000),
        "niter": cfg.dataset.get("niter", 10),
        "gpu_kmeans": cfg.dataset.get("gpu_kmeans", True),
        "resize_mode": cfg.dataset.get("resize_mode", "avg"),
        # Augmentation
        "augmentation": OmegaConf.to_container(cfg.dataset.augmentation, resolve=True)
        if cfg.dataset.get("augmentation") is not None
        else None,
        # Checkpointing
        "checkpoint_interval_batches": cfg.dataset.get("checkpoint_interval_batches", 500),
        "max_consecutive_errors": cfg.dataset.get("max_consecutive_errors", 50),
        # WDS-specific
        "max_open_files": cfg.dataset.get("max_open_files", 64),
        # Multi-image
        "image_field_pattern": cfg.dataset.get("image_field_pattern"),
        "image_list_column": cfg.dataset.get("image_list_column"),
        # W&B
        "wandb": OmegaConf.to_container(cfg.get("wandb", {}), resolve=True),
    }

    # Conversation policy for SFT mode
    conv_policy = cfg.dataset.get("conversation_policy")
    if conv_policy is not None:
        from vision_tokenization.vokenizers.conversation_policy import ConversationPolicy

        pipeline_cfg["tokenizer_kwargs"] = {
            "conversation_policy": ConversationPolicy(
                **OmegaConf.to_container(conv_policy, resolve=True)
            ),
        }

    result = run_distributed_pipeline(pipeline_cfg)

    logger.info("Pipeline completed!")
    logger.info(f"Total processed: {result.get('samples_processed', 0)}")
    logger.info(f"Total tokens: {result.get('tokens_generated', 0)}")
    logger.info(f"Output directory: {result.get('output_dir', cfg.output_dir)}")

    return result


if __name__ == "__main__":
    main()
