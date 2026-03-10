"""Distributed vision tokenization pipeline (torch.distributed, no Ray).

Entry point: ``run_distributed_pipeline(cfg)``
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import torch

from .core import tokenize_loop
from .dry_run import dry_run_batch_plan

logger = logging.getLogger(__name__)

__all__ = ["run_distributed_pipeline"]


def _build_output_subdir(cfg: Dict[str, Any]) -> str:
    """Build a dataset-specific subdirectory path.

    Layout::

        image_only:     image_only/{output_name}
        sft:            sft/{output_name}
        image2text:     image2text/{output_name}
        text2image:     text2image/{output_name}
    """
    output_name = cfg.get("output_name")
    if not output_name:
        raise ValueError("'output_name' is required in the dataset config.")
    mode = cfg.get("mode", "image_only")
    return str(Path(mode) / output_name)


def run_distributed_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point for the distributed vision tokenization pipeline.

    Expects a pre-built manifest (via indexing).  Loads or computes a
    BatchPlan, tokenizes on GPU, and writes micro-shards with checkpointing.
    """
    # torchrun sets RANK/WORLD_SIZE/LOCAL_RANK.
    # srun (without torchrun) sets SLURM_PROCID/SLURM_NTASKS/SLURM_LOCALID.
    num_gpus = cfg.get("num_gpus")
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))

    # Handle dry-run mode early (no GPU, no world-size check needed)
    if cfg.get("dry_run", False):
        cfg["rank"] = 0
        cfg["world_size"] = 1
        cfg["local_rank"] = 0
        cfg["output_dir"] = str(Path(cfg["output_dir"]) / _build_output_subdir(cfg))

        from .core import _load_or_compute_batch_plan
        batch_plan = _load_or_compute_batch_plan(cfg)
        result = dry_run_batch_plan(
            batch_plan,
            spatial_factor=cfg.get("spatial_factor", 16),
        )
        result["output_dir"] = cfg["output_dir"]
        return result

    # Cross-check num_gpus against env-derived world_size
    if num_gpus is not None:
        num_gpus = int(num_gpus)
        if world_size == 1 and num_gpus > 1:
            raise RuntimeError(
                f"num_gpus={num_gpus} but only 1 process detected. "
                f"Launch with srun --ntasks={num_gpus} or "
                f"torchrun --nproc_per_node={num_gpus}."
            )
        elif world_size != num_gpus:
            raise RuntimeError(
                f"num_gpus={num_gpus} from config does not match "
                f"world_size={world_size} from environment. "
                f"Check SLURM --ntasks-per-node * --nodes matches num_gpus."
            )

    # Infer LOCAL_RANK if missing
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" not in os.environ:
        gpus_per_node = torch.cuda.device_count()
        if gpus_per_node > 0:
            local_rank = rank % gpus_per_node
            logger.warning(
                f"[rank {rank}] LOCAL_RANK not set, inferred {local_rank} "
                f"from rank % {gpus_per_node} GPUs"
            )

    # Only rank 0 logs at INFO
    if rank != 0:
        logging.getLogger("vision_tokenization").setLevel(logging.WARNING)

    cfg["rank"] = rank
    cfg["world_size"] = world_size
    cfg["local_rank"] = local_rank

    # Namespace output to avoid checkpoint collisions
    cfg["output_dir"] = str(Path(cfg["output_dir"]) / _build_output_subdir(cfg))

    torch.cuda.set_device(local_rank)

    logger.info(
        f"[rank {rank}/{world_size}] starting (local_rank={local_rank}, "
        f"no NCCL — each rank is independent)"
    )

    # Build tokenizer-agnostic handler
    from .handler import TokenizationHandler
    from .writer import MicroShardWriter

    mode = cfg.get("mode", "image_only")
    writer = MicroShardWriter()
    needs_text = mode in ("sft", "image2text", "text2image")
    handler = TokenizationHandler(writer, needs_text)

    return tokenize_loop(rank, world_size, cfg, handler)
