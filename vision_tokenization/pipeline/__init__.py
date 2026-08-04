"""Distributed vision tokenization pipeline (torch.distributed, no Ray).

Entry point: ``run_distributed_pipeline(cfg)``
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

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
        alignment:      alignment/{output_name}
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

    if cfg["mode"] == "alignment":
        # One single-image document per unique media — grouping knobs don't apply.
        if cfg.get("multi_image", False):
            raise ValueError("multi_image is meaningless for alignment")

    # Handle dry-run mode early (no GPU, no world-size check needed)
    if cfg.get("dry_run", False):
        cfg["rank"] = 0
        cfg["world_size"] = 1
        cfg["local_rank"] = 0
        cfg["output_dir"] = str(Path(cfg["output_dir"]) / _build_output_subdir(cfg))
        public_dir = Path(cfg["output_dir"])

        if cfg["mode"] == "alignment":
            # The plan is built from scan.parquet, which only the scan stage
            # produces — so the dry run runs the real scan stage (ingest IS
            # the scan; CPU-only), then reports plan-derived token counts. The
            # persisted scan.parquet is byte-identical to the GPU job's
            # (deterministic ingest), making this a true pre-flight.
            from .runtime.alignment import (
                run_scan_stage,
                _stage_scan_into_work,
                _unstage_work,
            )

            _stage_scan_into_work(cfg)
            run_scan_stage(cfg)

        from .runtime.dry_run import export_dry_run

        from .runtime.executor import _load_or_build_plan
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
        if plan.off_canonical_stats:
            result["off_canonical"] = plan.off_canonical_stats
        result["output_dir"] = str(public_dir)
        export_dry_run(result, str(public_dir))
        if cfg["mode"] == "alignment":
            _unstage_work(public_dir, cfg)
        return result

    import torch

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

    # Alignment ENCODE phase (multi-rank): read the pre-built scan and spill this
    # rank's disjoint media slice via SpillBackend. The scan runs inline
    # beforehand; publish_alignment_store assembles the store + manifest after.
    if cfg["mode"] == "alignment":
        from .runtime.alignment import run_alignment

        return run_alignment(cfg)

    from .runtime.executor import run_executor

    return run_executor(rank, world_size, cfg)
