"""Checkpointing, micro-shard I/O, and per-rank stats tracking.

Adapted from audio_tokenization/pipelines/lhotse/checkpoint.py.

- **Micro-shard chunking**: Each rank writes independent chunks named
  ``rank_XXXX_chunk_YYYY.{bin,idx}``.  Written to ``.tmp`` first,
  atomically renamed on finalize.
- **Direct checkpointing**: Deterministic BatchPlan iteration means
  checkpoint = (batch_index, chunk_id, stats).  No sampler state needed.
- **WorkerStats**: Inline dataclass tracking vision-specific metrics.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from vision_tokenization.formats.megatron import (
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
    stage2_samples: int = 0
    lct_tokens: int = 0
    lct_samples: int = 0
    elapsed_offset: float = 0.0
    start_time: float = field(default_factory=time.time)
    elapsed_time: float = 0.0
    throughput: float = 0.0

    def current_elapsed_time(self) -> float:
        """Return total wall time across all resume segments."""
        return self.elapsed_offset + max(0.0, time.time() - self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        elapsed = self.current_elapsed_time()
        throughput = self.tokens_generated / elapsed if elapsed > 0 else 0
        image_tokens_per_second = self.image_tokens / elapsed if elapsed > 0 else 0
        d = {
            "samples_processed": self.samples_processed,
            "tokens_generated": self.tokens_generated,
            "image_tokens": self.image_tokens,
            "text_tokens": self.text_tokens,
            "errors": self.errors,
            "samples_skipped": self.samples_skipped,
            "cuda_oom_errors": self.cuda_oom_errors,
            "stage2_tokens": self.stage2_tokens,
            "stage2_samples": self.stage2_samples,
            "lct_tokens": self.lct_tokens,
            "lct_samples": self.lct_samples,
            "elapsed_time": elapsed,
            "throughput": throughput,
            "image_tokens_per_second": image_tokens_per_second,
        }
        return d

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Restore cumulative counters from a checkpoint stats payload."""
        self.samples_processed = data.get("samples_processed", 0)
        self.tokens_generated = data.get("tokens_generated", 0)
        self.image_tokens = data.get("image_tokens", 0)
        self.text_tokens = data.get("text_tokens", 0)
        self.errors = data.get("errors", 0)
        self.samples_skipped = data.get("samples_skipped", 0)
        self.cuda_oom_errors = data.get("cuda_oom_errors", 0)
        self.stage2_tokens = data.get("stage2_tokens", 0)
        self.stage2_samples = data.get("stage2_samples", 0)
        self.lct_tokens = data.get("lct_tokens", 0)
        self.lct_samples = data.get("lct_samples", 0)
        self.elapsed_time = float(data.get("elapsed_time", 0.0) or 0.0)
        self.elapsed_offset = self.elapsed_time
        self.throughput = float(data.get("throughput", 0.0) or 0.0)

    def finalize(self) -> Dict[str, Any]:
        """Compute elapsed time and throughput, return final stats dict."""
        final = self.to_dict()
        self.elapsed_time = final["elapsed_time"]
        self.throughput = final["throughput"]
        return final


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
    src_buf: Optional[List[int]] = None,
    src_path: Optional[str] = None,
) -> None:
    """Finalize index and atomically move temporary shard files in place.

    Calls ``fsync`` on both temp files before renaming to ensure data is
    durable on network filesystems (e.g. Lustre).

    When *src_buf* is provided (provenance enabled), the caller also supplies
    *src_path* (derived via ``provenance.sidecar_path`` so the suffix lives in
    one place).  A parallel ``.src.npy`` sidecar holding one int64
    source-manifest-row per written sequence shares the same tmp + fsync +
    atomic-rename transaction, and is renamed **last** so a crash never leaves a
    complete shard paired with a stale or half-written sidecar — merge treats a
    missing sidecar as "regenerate", never as silent misalignment.
    """
    builder.finalize(tmp_idx)

    tmp_paths = [tmp_bin, tmp_idx]
    tmp_src = None
    if src_buf is not None:
        if src_path is None:
            raise ValueError("src_path is required when src_buf is provided")
        # Lazy import: provenance imports nothing from the pipeline, so this is
        # cycle-free and keeps the .src.npy write in one place.
        from ..output.provenance import save_source_ids

        tmp_src = src_path + ".tmp"
        save_source_ids(tmp_src, src_buf)
        tmp_paths.append(tmp_src)

    # fsync via O_WRONLY to flush write-back cache (O_RDONLY works on Linux
    # but is technically non-portable; O_WRONLY is POSIX-correct).
    for p in tmp_paths:
        fd = os.open(p, os.O_WRONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    os.replace(tmp_bin, bin_path)
    os.replace(tmp_idx, idx_path)
    if tmp_src is not None:
        os.replace(tmp_src, src_path)


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
