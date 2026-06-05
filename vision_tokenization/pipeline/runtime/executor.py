"""Per-rank tokenization executor.

Main runtime loop driven by ``TokenizationPlan``.

Flow:
    1. Load/build TokenizationPlan (from file or manifest).
    2. Split image_batches across ranks.
    3. Iterate batches: GPU encode images → spill keyed component payloads.
    4. CPU-tokenize text components → spill keyed component payloads.
    5. Checkpoint by batch_index (deterministic).
    6. After all ranks finish: offline rebuild joins spill to plan.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ...indexing.planning.tokenization_plan import (
    TokenizationPlan,
    build_tokenization_plan,
)
from .checkpoint import WorkerStats, load_checkpoint, save_checkpoint
from .data import create_loader
from .prefetch import BatchPrefetcher, PrefetchResult
from .wandb_logger import SimpleWandbLogger

logger = logging.getLogger(__name__)


def _load_or_build_plan(cfg: Dict[str, Any]) -> TokenizationPlan:
    """Load a pre-computed TokenizationPlan from file, or build one."""
    plan_path = cfg.get("plan_path") or cfg.get("batch_plan")
    if plan_path and Path(plan_path).exists():
        logger.info(f"Loading plan from {plan_path}")
        plan = torch.load(plan_path, map_location="cpu", weights_only=False)
        if isinstance(plan, TokenizationPlan):
            return plan
        raise TypeError(f"Expected TokenizationPlan, got {type(plan)}")

    # Build from manifest
    plan = build_tokenization_plan(
        manifest_path=cfg["manifest_path"],
        mode=cfg["mode"],
        text_column=cfg.get("text_column"),
        parser=cfg.get("parser"),
        min_pixels=cfg.get("filter_min_pixels"),
        max_pixels=cfg.get("filter_max_pixels"),
        max_images_per_doc=cfg.get("max_images_per_doc"),
        batch_size=cfg.get("batch_size", 128),
        max_batch_tokens=cfg.get("max_batch_tokens", 32768),
        spatial_factor=cfg.get("spatial_factor", 16),
        resize_min_pixels=cfg.get("tokenizer_min_pixels", 16384),
        resize_max_pixels=cfg.get("tokenizer_max_pixels", 1960000),
        window_size=cfg.get("window_size", 2000),
    )

    if plan_path:
        Path(plan_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(plan, plan_path)
        logger.info(f"Saved plan to {plan_path}")

    return plan


def _get_rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return -1.0


def _is_cuda_oom(err: BaseException) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or (
        isinstance(err, RuntimeError) and "out of memory" in str(err).lower()
    )


def _build_run_name(cfg: Dict[str, Any], mode: str, world_size: int) -> str:
    output_name = cfg.get("output_name", "unknown")
    mbt = cfg.get("max_batch_tokens", "")
    bs = cfg.get("batch_size", "")
    return f"{output_name}_{mode}_g{world_size}_mbt{mbt}_bs{bs}"


def load_wandb_resume_state(resume: bool, ckpt: Optional[dict]) -> Optional[dict]:
    if resume and ckpt is not None and "wandb" in ckpt.get("extra", {}):
        return ckpt["extra"]["wandb"]
    return None


def _build_group_slices(doc_ids: np.ndarray) -> Optional[np.ndarray]:
    """Return contiguous group_slices for document-grouped rows."""
    if len(doc_ids) == 0:
        return None
    changes = np.where(np.diff(doc_ids) != 0)[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(doc_ids)]))
    slices = np.stack((starts, ends), axis=1).astype(np.int64)
    return slices if len(slices) > 0 else None


def _filter_prefetched_batch(
    images: List[Any],
    texts: Optional[List[Any]],
    component_indices: np.ndarray,
    group_slices: Optional[np.ndarray],
    stats: WorkerStats,
) -> tuple[List[Any], Optional[List[Any]], np.ndarray, Optional[np.ndarray]]:
    """Filter out invalid images, preserving grouped document structure."""
    if group_slices is not None:
        valid_images: List[Any] = []
        valid_texts: List[Any] = []
        valid_comp_indices: List[int] = []
        valid_slices: List[tuple[int, int]] = []

        for g_idx, (start, end) in enumerate(group_slices):
            start, end = int(start), int(end)
            group_images = images[start:end]
            group_text = texts[g_idx] if texts is not None else None

            if (texts is not None and group_text is None) or any(img is None for img in group_images):
                stats.samples_skipped += 1
                continue

            new_start = len(valid_images)
            valid_images.extend(group_images)
            valid_comp_indices.extend(int(ci) for ci in component_indices[start:end])
            valid_slices.append((new_start, len(valid_images)))
            if texts is not None:
                valid_texts.append(group_text)

        valid_slice_arr = np.array(valid_slices, dtype=np.int64) if valid_slices else None
        return (
            valid_images,
            valid_texts if texts is not None else None,
            np.asarray(valid_comp_indices, dtype=np.int64),
            valid_slice_arr,
        )

    if texts is not None:
        valid_images = []
        valid_texts = []
        valid_comp_indices = []
        for i, img in enumerate(images):
            txt = texts[i] if i < len(texts) else None
            if img is not None and txt is not None:
                valid_images.append(img)
                valid_texts.append(txt)
                valid_comp_indices.append(int(component_indices[i]))
            else:
                stats.samples_skipped += 1
        return (
            valid_images,
            valid_texts,
            np.asarray(valid_comp_indices, dtype=np.int64),
            None,
        )

    valid_positions = [i for i, img in enumerate(images) if img is not None]
    stats.samples_skipped += len(images) - len(valid_positions)
    return (
        [images[i] for i in valid_positions],
        None,
        np.asarray([int(component_indices[i]) for i in valid_positions], dtype=np.int64),
        None,
    )


# ---------------------------------------------------------------------------
# Executor loop
# ---------------------------------------------------------------------------


def run_executor(
    rank: int,
    world_size: int,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Main per-rank tokenization loop driven by TokenizationPlan.

    GPU-encodes image components, CPU-tokenizes text components, and
    spills all component payloads keyed by (document_id, component_index).
    """
    output_dir = cfg["output_dir"]
    mode = cfg["mode"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load / build TokenizationPlan and split for this rank
    # ------------------------------------------------------------------
    # One plan for all modes. Document-boundary-aware windowing ensures
    # no document is split across ranks after contiguous splitting.
    plan = _load_or_build_plan(cfg)
    worker_splits = plan.split_image_batches_for_workers(world_size)
    my_batches = worker_splits[rank] if rank < len(worker_splits) else []

    logger.info(
        f"[rank {rank}/{world_size}] Assigned {len(my_batches)} image batches"
    )

    if not my_batches:
        logger.warning(f"[rank {rank}] No batches assigned — exiting early")
        return {"rank": rank, "samples_processed": 0, "tokens_generated": 0}

    # ------------------------------------------------------------------
    # 3. Resume from checkpoint
    # ------------------------------------------------------------------
    resume = cfg.get("resume", False)
    start_batch_index = 0
    cumulative_stats = WorkerStats()
    ckpt = None

    if resume:
        ckpt = load_checkpoint(output_dir, rank)
        if ckpt is not None:
            ckpt_ws = ckpt.get("world_size")
            if ckpt_ws is not None and ckpt_ws != world_size:
                logger.warning(
                    f"[rank {rank}] Checkpoint world_size ({ckpt_ws}) != current ({world_size}). Ignoring."
                )
                ckpt = None
        if ckpt is not None:
            start_batch_index = ckpt["batch_index"] + 1
            prev = ckpt.get("stats", {})
            cumulative_stats.load_from_dict(prev)
            logger.info(
                f"[rank {rank}] Resumed from batch {start_batch_index}, "
                f"components={cumulative_stats.samples_processed}"
            )

    # ------------------------------------------------------------------
    # 4. Create tokenizer on GPU
    # ------------------------------------------------------------------
    from vision_tokenization.discrete.emu import create_tokenizer

    device = f"cuda:{cfg.get('local_rank', 0)}"
    tokenizer = create_tokenizer(
        mode=mode,
        text_tokenizer_path=cfg["tokenizer_path"],
        device=device,
        min_pixels=cfg["tokenizer_min_pixels"],
        max_pixels=cfg["tokenizer_max_pixels"],
        max_encode_pixels=cfg.get("max_encode_pixels"),
        **(cfg.get("tokenizer_kwargs", {})),
    )

    # ------------------------------------------------------------------
    # 5. Setup output backend, data loader, prefetcher, W&B
    # ------------------------------------------------------------------
    multi_image = bool(cfg.get("multi_image", False))
    use_spill = multi_image or mode == "interleave"

    if use_spill:
        from ..output.backend import SpillBackend
        backend = SpillBackend()
        backend.open(output_dir, rank, resume_state=ckpt)
    else:
        from ..output.backend import DirectBackend
        backend = DirectBackend(mode=mode, seqlen_threshold=cfg.get("seqlen_threshold"))
        backend.open(output_dir, rank, resume_state=ckpt, tokenizer=tokenizer)

    data_loader = create_loader(cfg)

    # W&B logger (rank 0 only)
    wandb_logger = None
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False) and rank == 0:
        wandb_resume_state = load_wandb_resume_state(resume, ckpt)
        wandb_logger = SimpleWandbLogger(
            project=wandb_cfg.get("project", "vision-tokenization"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("name") or _build_run_name(cfg, mode, world_size),
            tags=wandb_cfg.get("tags", []),
            config={
                "rank": rank, "world_size": world_size, "mode": mode,
                **{k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))},
            },
            log_interval_seconds=wandb_cfg.get("log_interval_seconds", 10.0),
            run_id=wandb_resume_state["run_id"] if wandb_resume_state else None,
            start_step=wandb_resume_state["step"] if wandb_resume_state else 0,
        )

    # ------------------------------------------------------------------
    # 6. Build a BatchAssignment-like adapter for the prefetcher
    # ------------------------------------------------------------------
    # The prefetcher expects objects with .sample_indices and .resize_height/width.
    # We adapt ImageBatch to work with the existing prefetcher by mapping
    # component_indices → manifest_rows.
    from dataclasses import dataclass, field as dc_field

    @dataclass
    class _PrefetchBatch:
        """Adapter: maps ImageBatch component_indices to manifest rows for loading."""
        sample_indices: np.ndarray   # manifest rows (for data loader)
        resize_height: int
        resize_width: int
        group_slices: Optional[np.ndarray] = None
        # Original component indices (for spill keying)
        _component_indices: np.ndarray = dc_field(default_factory=lambda: np.array([]))
        batch_token_count: int = 0

    prefetch_batches = []
    for ib in my_batches:
        comp_indices = np.asarray(ib.component_indices, dtype=np.int64)
        manifest_rows = plan.components.source_ref[comp_indices]
        doc_ids = plan.components.document_id[comp_indices]
        comp_order = plan.components.component_index[comp_indices]

        # Keep each document contiguous within a batch so loaders can fetch one
        # text payload / structured document per group.
        order = np.lexsort((manifest_rows, comp_order, doc_ids))
        comp_indices = comp_indices[order]
        manifest_rows = manifest_rows[order]
        doc_ids = doc_ids[order]

        group_slices = None
        if cfg.get("multi_image", False) or mode in ("sft", "interleave"):
            group_slices = _build_group_slices(doc_ids)

        prefetch_batches.append(_PrefetchBatch(
            sample_indices=manifest_rows,
            resize_height=ib.resize_height,
            resize_width=ib.resize_width,
            group_slices=group_slices,
            _component_indices=comp_indices,
            batch_token_count=ib.batch_token_count,
        ))

    # ------------------------------------------------------------------
    # 7. Main loop
    # ------------------------------------------------------------------
    checkpoint_interval = cfg.get("checkpoint_interval_batches", 2500)
    stats = cumulative_stats
    batch_count = 0
    last_batch_index = start_batch_index - 1
    consecutive_errors = 0
    max_consecutive_errors = cfg.get("max_consecutive_errors", 50)
    _loop_error = None

    prefetch_cfg = cfg.get("prefetch", {})
    prefetcher = BatchPrefetcher(
        data_loader,
        queue_size=prefetch_cfg.get("queue_size", 32),
        num_workers=prefetch_cfg.get("num_workers", 8),
    )

    logger.info(
        f"[rank {rank}] Starting unified tokenization loop "
        f"(start_batch={start_batch_index}, "
        f"checkpoint_interval={checkpoint_interval}, "
        f"prefetch_workers={prefetch_cfg.get('num_workers', 8)})"
    )

    try:
        batch_iter = prefetcher.iter_batches(prefetch_batches, start=start_batch_index)
        if rank == 0:
            from tqdm import tqdm
            batch_iter = tqdm(
                batch_iter, total=len(prefetch_batches) - start_batch_index,
                desc="rank 0", unit="batch", dynamic_ncols=True,
            )

        for result in batch_iter:
            last_batch_index = result.batch_index
            pb = prefetch_batches[result.batch_index]
            log_now = wandb_logger.should_log_now() if wandb_logger is not None else False

            # Memory monitoring
            if log_now or (batch_count % 50 == 0 and wandb_logger is None):
                rss_gb = _get_rss_gb()
                cuda_alloc = torch.cuda.memory_allocated() / 1024**3
                cuda_reserved = torch.cuda.memory_reserved() / 1024**3
                if batch_count % 50 == 0 and wandb_logger is None:
                    logger.warning(
                        f"[rank {rank}] batch={result.batch_index} "
                        f"RSS={rss_gb:.2f} GB CUDA_alloc={cuda_alloc:.2f} GB "
                        f"CUDA_rsv={cuda_reserved:.2f} GB "
                        f"resize=({pb.resize_height}x{pb.resize_width}) "
                        f"n_images={len(pb.sample_indices)}"
                    )

            # Prefetch error
            if result.error is not None:
                stats.errors += 1
                consecutive_errors += 1
                logger.warning(
                    f"[rank {rank}] Prefetch error on batch {result.batch_index} "
                    f"({consecutive_errors}/{max_consecutive_errors}): {result.error}"
                )
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"[rank {rank}] {max_consecutive_errors} consecutive errors"
                    ) from result.error
                continue

            try:
                resize_size = (pb.resize_height, pb.resize_width)
                valid_images, valid_texts, valid_comp_indices, valid_group_slices = (
                    _filter_prefetched_batch(
                        result.images,
                        result.texts,
                        pb._component_indices,
                        pb.group_slices,
                        stats,
                    )
                )

                if not valid_images:
                    consecutive_errors = 0
                    batch_count += 1
                    continue

                if use_spill:
                    # Spill path: GPU encode images, then spill components
                    # Spill mode tokenizes images eagerly in the executor, so
                    # tokenize wall time is exactly this encode section.
                    t0 = time.perf_counter() if log_now else None
                    token_sequences = tokenizer.tokenize_images(
                        valid_images, resize_size,
                    )
                    if log_now:
                        tokenize_wall_ms = (time.perf_counter() - t0) * 1000
                        gpu_ms = tokenize_wall_ms
                    else:
                        tokenize_wall_ms = 0.0
                        gpu_ms = 0.0

                    write_timing = backend.write_batch(
                        image_tokens=token_sequences,
                        texts=valid_texts,
                        component_indices=valid_comp_indices,
                        group_slices=valid_group_slices,
                        resize_height=pb.resize_height,
                        resize_width=pb.resize_width,
                        plan=plan,
                        tokenizer=tokenizer,
                        stats=stats,
                    )
                else:
                    # Direct path: handler tokenizes + writes in one step
                    device = f"cuda:{cfg.get('local_rank', 0)}"
                    write_timing = backend.write_batch(
                        images=valid_images,
                        resize_size=resize_size,
                        texts=valid_texts,
                        group_slices=valid_group_slices,
                        tokenizer=tokenizer,
                        stats=stats,
                        device=device,
                        timing_enabled=log_now,
                    )
                    gpu_ms = write_timing.get("tokenize_gpu_ms", 0)
                    tokenize_wall_ms = write_timing.get("tokenize_wall_ms", 0)

                write_ms = write_timing.get("write_ms", 0)
                consecutive_errors = 0

            except Exception as batch_err:
                stats.errors += 1
                consecutive_errors += 1
                if _is_cuda_oom(batch_err):
                    torch.cuda.empty_cache()
                    stats.cuda_oom_errors += 1
                    logger.warning(f"[rank {rank}] CUDA OOM on batch {result.batch_index}")
                else:
                    logger.warning(f"[rank {rank}] Batch error: {batch_err}")
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"[rank {rank}] {max_consecutive_errors} consecutive errors"
                    ) from batch_err
                continue

            batch_count += 1

            # W&B logging
            if wandb_logger is not None and log_now:
                wandb_logger.log(
                    samples=stats.samples_processed,
                    tokens=stats.tokens_generated,
                    image_tokens=stats.image_tokens,
                    text_tokens=stats.text_tokens,
                    errors=stats.errors,
                    skipped=stats.samples_skipped,
                    timing={
                        "load_ms": result.timing["load_ms"],
                        "tokenize_gpu_ms": gpu_ms,
                        "tokenize_wall_ms": tokenize_wall_ms,
                        "write_ms": write_ms,
                    },
                    metrics={
                        "batch/index": result.batch_index,
                        "batch/resize_height": pb.resize_height,
                        "batch/resize_width": pb.resize_width,
                        "batch/n_images": len(pb.sample_indices),
                        "memory/rss_gb": _get_rss_gb(),
                        "memory/cuda_alloc_gb": torch.cuda.memory_allocated() / 1024**3,
                        "memory/cuda_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                    },
                    elapsed_seconds=stats.current_elapsed_time(),
                )

            # Periodic checkpoint
            if batch_count % checkpoint_interval == 0:
                ckpt_meta = backend.checkpoint()
                save_checkpoint(
                    output_dir, rank,
                    batch_index=result.batch_index,
                    chunk_id=ckpt_meta.get("chunk_id", ckpt_meta.get("shard_id", 0)),
                    stats=stats.to_dict(),
                    world_size=world_size,
                    extra={"wandb": wandb_logger.state_dict()} if wandb_logger else None,
                )
                logger.info(
                    f"[rank {rank}] Checkpoint at batch {result.batch_index}, "
                    f"{stats.tokens_generated:,} tokens"
                )

    except Exception as e:
        logger.error(f"[rank {rank}] Fatal error: {e}", exc_info=True)
        stats.errors += 1
        _loop_error = e
    finally:
        prefetcher.shutdown()

    # ------------------------------------------------------------------
    # 8. Finalize
    # ------------------------------------------------------------------
    backend.finalize()

    # Per-rank rebuild: assemble documents from this rank's spill into
    # rank_XXXX_chunk_0000.bin/.idx so merge_shards works identically
    # for both spill and direct backend paths.
    if use_spill and _loop_error is None and cfg.get("rebuild", True):
        from ..output.rebuild import rebuild_rank
        from ...common.assembly import StructureTokenIds

        rebuild_token_ids = StructureTokenIds(
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            img_start_id=tokenizer.img_start_id,
            img_end_id=tokenizer.img_end_id,
            img_token_start_id=tokenizer.img_token_start_id,
            eol_id=tokenizer.eol_id,
            eof_id=tokenizer.eof_id,
            vision_token_offset=tokenizer.vision_token_offset,
            image_token_id=getattr(tokenizer, "image_token_id", -1),
            dim_tokens_fn=tokenizer._get_dim_tokens,
        )
        rebuild_stats = rebuild_rank(
            plan=plan,
            rank=rank,
            spill_dir=output_dir,
            token_ids=rebuild_token_ids,
            vocab_size=len(tokenizer.text_tokenizer),
            max_sequence_tokens=cfg.get("max_sequence_tokens"),
            seqlen_threshold=cfg.get("seqlen_threshold"),
        )
        stats.stage2_tokens = rebuild_stats.get("stage2_tokens", 0)
        stats.stage2_samples = rebuild_stats.get("stage2_sequences", 0)
        stats.lct_tokens = rebuild_stats.get("lct_tokens", 0)
        stats.lct_samples = rebuild_stats.get("lct_sequences", 0)

    save_checkpoint(
        output_dir, rank,
        batch_index=last_batch_index,
        chunk_id=0,
        stats=stats.to_dict(),
        world_size=world_size,
        extra={"wandb": wandb_logger.state_dict()} if wandb_logger else None,
    )

    result = stats.finalize()
    result["rank"] = rank
    result["output_dir"] = output_dir

    # Write per-rank stats for post-run aggregation
    from vision_tokenization.utils.json import json_dump
    stats_path = Path(output_dir) / f"rank_{rank:04d}_stats.json"
    json_dump(result, stats_path)

    if wandb_logger is not None:
        wandb_logger.finish()

    if _loop_error is not None:
        raise _loop_error

    logger.info(
        f"[rank {rank}] Unified loop complete: "
        f"{result['samples_processed']:,} images, "
        f"{result['tokens_generated']:,} tokens, "
        f"{result['image_tokens_per_second']:,.0f} img tok/s avg"
    )

    # Try to write aggregate stats summary (succeeds when all ranks are done)
    from ..output.stats_reducer import maybe_write_stats_summary
    summary = maybe_write_stats_summary(output_dir, expected_ranks=world_size)
    if summary is not None:
        logger.info(
            f"[rank {rank}] Stats summary: {summary['samples_processed']:,} samples, "
            f"{summary['tokens_generated']:,} tokens across {summary['num_ranks']} ranks"
        )

    return result
