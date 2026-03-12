"""Checkpointing, micro-shard I/O, stats tracking, and W&B logging.

Adapted from audio_tokenization/pipelines/lhotse/checkpoint.py.

- **Micro-shard chunking**: Each rank writes independent chunks named
  ``rank_XXXX_chunk_YYYY.{bin,idx}``.  Written to ``.tmp`` first,
  atomically renamed on finalize.
- **Batch-index checkpointing**: Deterministic BatchPlan iteration means
  checkpoint = (batch_index, chunk_id, stats).  No sampler state needed.
- **WorkerStats**: Inline dataclass tracking vision-specific metrics.
- **SimpleWandbLogger**: Rate-limited W&B logging (rank 0 only).
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from vision_tokenization.pipelines.indexed_dataset_megatron import (
    DType,
    IndexedDatasetBuilder,
)

logger = logging.getLogger(__name__)

__all__ = [
    "WorkerStats",
    "open_chunk_writer",
    "finalize_shard_writer",
    "save_checkpoint",
    "load_checkpoint",
    "SimpleWandbLogger",
    "is_cuda_oom",
]


# ---------------------------------------------------------------------------
# CUDA OOM detection
# ---------------------------------------------------------------------------


def is_cuda_oom(exc: BaseException) -> bool:
    """Return True if *exc* indicates a CUDA out-of-memory error."""
    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom_type is not None and isinstance(exc, cuda_oom_type):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "cuda out of memory" in msg or "out of memory" in msg
    return False


# ---------------------------------------------------------------------------
# Per-rank statistics
# ---------------------------------------------------------------------------


@dataclass
class WorkerStats:
    """Cumulative statistics tracked per rank."""

    samples_processed: int = 0
    tokens_generated: int = 0
    image_tokens: int = 0
    text_tokens: int = 0
    errors: int = 0
    samples_skipped: int = 0
    cuda_oom_errors: int = 0
    stage2_tokens: int = 0
    lct_tokens: int = 0
    start_time: float = field(default_factory=time.time)
    elapsed_time: float = 0.0
    throughput: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "samples_processed": self.samples_processed,
            "tokens_generated": self.tokens_generated,
            "image_tokens": self.image_tokens,
            "text_tokens": self.text_tokens,
            "errors": self.errors,
            "samples_skipped": self.samples_skipped,
            "cuda_oom_errors": self.cuda_oom_errors,
            "stage2_tokens": self.stage2_tokens,
            "lct_tokens": self.lct_tokens,
            "elapsed_time": self.elapsed_time,
            "throughput": self.throughput,
        }
        return d

    def finalize(self) -> Dict[str, Any]:
        """Compute elapsed time and throughput, return final stats dict."""
        self.elapsed_time = time.time() - self.start_time
        self.throughput = (
            self.tokens_generated / self.elapsed_time if self.elapsed_time > 0 else 0
        )
        return self.to_dict()


# ---------------------------------------------------------------------------
# Micro-shard chunk writer
# ---------------------------------------------------------------------------


def open_chunk_writer(
    output_dir: str,
    rank: int,
    chunk_id: int,
    vocab_size: int,
) -> Tuple[IndexedDatasetBuilder, str, str, str, str]:
    """Open an IndexedDatasetBuilder for a micro-shard chunk.

    Naming: ``rank_XXXX_chunk_YYYY.{bin,idx}``
    Writes to ``.tmp`` suffix; call ``finalize_shard_writer()`` to atomically
    rename to the final paths.

    Returns:
        (builder, tmp_bin_path, tmp_idx_path, final_bin_path, final_idx_path)
    """
    output_prefix = Path(output_dir) / f"rank_{rank:04d}_chunk_{chunk_id:04d}"
    bin_path = str(output_prefix) + ".bin"
    idx_path = str(output_prefix) + ".idx"
    tmp_bin_path = bin_path + ".tmp"
    tmp_idx_path = idx_path + ".tmp"
    dtype = DType.optimal_dtype(vocab_size)
    builder = IndexedDatasetBuilder(tmp_bin_path, dtype=dtype)
    return builder, tmp_bin_path, tmp_idx_path, bin_path, idx_path


def finalize_shard_writer(
    builder: IndexedDatasetBuilder,
    tmp_bin: str,
    tmp_idx: str,
    bin_path: str,
    idx_path: str,
) -> None:
    """Finalize index and atomically move temporary shard files in place.

    Calls ``fsync`` on both temp files before renaming to ensure data is
    durable on network filesystems (e.g. Lustre).
    """
    builder.finalize(tmp_idx)
    # fsync via O_WRONLY to flush write-back cache (O_RDONLY works on Linux
    # but is technically non-portable; O_WRONLY is POSIX-correct).
    for p in (tmp_bin, tmp_idx):
        fd = os.open(p, os.O_WRONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    os.replace(tmp_bin, bin_path)
    os.replace(tmp_idx, idx_path)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------


def _checkpoint_path(output_dir: str, rank: int) -> Path:
    return Path(output_dir) / f"rank_{rank:04d}_checkpoint.pt"


def save_checkpoint(
    output_dir: str,
    rank: int,
    batch_index: int,
    chunk_id: int,
    stats: Dict[str, Any],
    world_size: int = 1,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomically save checkpoint via ``.tmp`` + ``os.replace()``.

    *extra* is an optional dict merged into the payload (e.g.
    ``stage2_chunk_id`` / ``lct_chunk_id`` for split-mode writing).
    """
    ckpt_path = _checkpoint_path(output_dir, rank)
    tmp_path = str(ckpt_path) + ".tmp"
    payload = {
        "batch_index": batch_index,
        "chunk_id": chunk_id,
        "stats": stats,
        "world_size": world_size,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, tmp_path)
    os.replace(tmp_path, str(ckpt_path))
    logger.debug(f"[rank {rank}] Saved checkpoint batch_index={batch_index}, chunk_id={chunk_id}")


def load_checkpoint(output_dir: str, rank: int) -> Optional[Dict[str, Any]]:
    """Load checkpoint if it exists, else return None."""
    ckpt_path = _checkpoint_path(output_dir, rank)
    if not ckpt_path.exists():
        return None
    logger.info(f"[rank {rank}] Loading checkpoint from {ckpt_path}")
    return torch.load(str(ckpt_path), map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# W&B logger (rank 0 only)
# ---------------------------------------------------------------------------


class SimpleWandbLogger:
    """Lightweight W&B logger for rank 0.

    Logs running totals + throughput at a configurable interval.
    Calls are rate-limited: ``log()`` is a no-op unless ``log_interval_seconds``
    has elapsed since the last flush (or ``force=True``).
    """

    def __init__(
        self,
        project: str = "vision-tokenization",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[list] = None,
        config: Optional[dict] = None,
        log_interval_seconds: float = 10.0,
        stats_sampling_interval: float = 2.0,
    ):
        import wandb

        self._run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            tags=tags or [],
            config=config or {},
            resume="allow",
            settings=wandb.Settings(x_stats_sampling_interval=stats_sampling_interval),
        )
        self._interval = max(1.0, log_interval_seconds)
        self._last_flush = time.time()
        self._start_time = time.time()
        self._step = 0

    def log(
        self,
        samples: int,
        tokens: int,
        image_tokens: int = 0,
        text_tokens: int = 0,
        errors: int = 0,
        skipped: int = 0,
        timing: Optional[Dict[str, float]] = None,
        force: bool = False,
    ) -> None:
        """Log absolute totals if the flush interval has elapsed."""
        now = time.time()
        if not force and now - self._last_flush < self._interval:
            return
        import wandb

        elapsed = now - self._start_time
        payload = {
            "samples_processed": samples,
            "tokens_generated": tokens,
            "image_tokens": image_tokens,
            "text_tokens": text_tokens,
            "errors": errors,
            "samples_skipped": skipped,
            "samples_per_second": samples / elapsed if elapsed > 0 else 0,
            "tokens_per_second": tokens / elapsed if elapsed > 0 else 0,
            "elapsed_seconds": elapsed,
        }
        if timing:
            payload.update({f"timing/{k}": v for k, v in timing.items()})
        wandb.log(payload, step=self._step)
        self._step += 1
        self._last_flush = now

    def log_final(self, metrics: Dict[str, Any]) -> None:
        import wandb

        wandb.log({f"final/{k}": v for k, v in metrics.items()})

    def finish(self) -> None:
        import wandb

        wandb.finish()
