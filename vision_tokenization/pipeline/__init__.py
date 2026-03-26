"""Distributed vision tokenization pipeline (torch.distributed, no Ray).

Entry point: ``run_distributed_pipeline(cfg)``
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import torch

from .executor import run_executor
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
        interleave:     interleave/{output_name}
    """
    output_name = cfg.get("output_name")
    if not output_name:
        raise ValueError("'output_name' is required in the dataset config.")
    mode = cfg["mode"]
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

        from .dry_run import export_dry_run

        from .executor import _load_or_build_plan
        plan = _load_or_build_plan(cfg)
        result = {
            "total_documents": plan.total_documents,
            "total_components": plan.total_components,
            "total_image_components": plan.total_image_components,
            "total_text_components": plan.total_text_components,
            "total_batches": plan.total_batches,
            "total_image_tokens": sum(
                b.batch_token_count for b in plan.execution.image_batches
            ),
        }
        result["output_dir"] = cfg["output_dir"]
        export_dry_run(result, cfg["output_dir"])
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

    result = run_executor(rank, world_size, cfg)

    # Auto-rebuild for spill backend (multi-image / interleave)
    multi_image = bool(cfg.get("multi_image", False))
    mode = cfg["mode"]
    if (multi_image or mode == "interleave") and rank == 0:
        _maybe_rebuild(cfg, result)

    return result


def _maybe_rebuild(cfg: Dict[str, Any], tokenize_result: Dict[str, Any]) -> None:
    """Run offline rebuild after spill-based tokenization completes."""
    from .rebuild import rebuild_from_plan
    from .assembly import StructureTokenIds

    output_dir = cfg["output_dir"]
    plan_path = cfg.get("plan_path")
    if not plan_path or not Path(plan_path).exists():
        logger.warning("Skipping rebuild: no plan_path configured")
        return

    plan = torch.load(plan_path, map_location="cpu", weights_only=False)

    # Build StructureTokenIds from the tokenizer that was already loaded
    from vision_tokenization.discrete.emu import create_tokenizer
    tokenizer = create_tokenizer(
        mode=cfg["mode"],
        text_tokenizer_path=cfg["tokenizer_path"],
        device="cpu",
        min_pixels=cfg["tokenizer_min_pixels"],
        max_pixels=cfg["tokenizer_max_pixels"],
    )

    token_ids = StructureTokenIds(
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        img_start_id=tokenizer.img_start_id,
        img_end_id=tokenizer.img_end_id,
        img_token_start_id=tokenizer.img_token_start_id,
        eol_id=tokenizer.eol_id,
        eof_id=tokenizer.eof_id,
        vision_token_offset=tokenizer.vision_token_offset,
        image_token_id=getattr(tokenizer, "image_token_id", -1),
        dim_tokens_fn=getattr(tokenizer, "dim_tokens_fn", None),
    )

    logger.info(f"[rank 0] Starting rebuild from spill in {output_dir}")
    rebuild_from_plan(
        plan=plan,
        spill_dir=output_dir,
        token_ids=token_ids,
        vocab_size=200000,
        max_sequence_tokens=cfg.get("max_sequence_tokens"),
        seqlen_threshold=cfg.get("seqlen_threshold"),
        output_name="rebuilt",
    )
    logger.info(f"[rank 0] Rebuild complete")
