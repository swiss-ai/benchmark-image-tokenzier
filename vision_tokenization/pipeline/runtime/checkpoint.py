"""Checkpointing, micro-shard I/O, and per-rank stats tracking.

Adapted from audio_tokenization/pipelines/lhotse/checkpoint.py.

- **Micro-shard chunking**: Each rank writes independent chunks named
  ``rank_XXXX_chunk_YYYY.{bin,idx}``.  Written to ``.tmp`` first,
  atomically renamed on finalize.
- **Direct checkpointing**: Deterministic BatchPlan iteration means
  checkpoint = (batch_index, chunk_id, stats).  No sampler state needed.
- **WorkerStats**: Inline dataclass tracking vision-specific metrics.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from vision_tokenization.formats.megatron import IndexedDatasetBuilder

logger = logging.getLogger(__name__)

__all__ = [
    "WorkerStats",
    "open_chunk_writer",
    "finalize_shard_writer",
    "save_checkpoint",
    "load_checkpoint",
    "verify_run_world_size",
    "write_rank_manifest",
    "load_rank_manifests",
]


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
    elapsed_offset: float = 0.0
    start_time: float = field(default_factory=time.time)
    setup_elapsed_time: float = 0.0
    loop_timing_started: bool = False
    tokenizer_load_time: float = 0.0
    model_load_time: float = 0.0
    text_tokenizer_load_time: float = 0.0
    elapsed_time: float = 0.0
    throughput: float = 0.0

    def record_tokenizer_load_time(
        self,
        *,
        tokenizer_load_time: float,
        model_load_time: float = 0.0,
        text_tokenizer_load_time: float = 0.0,
    ) -> None:
        """Record cumulative setup-only tokenizer/model load durations in seconds."""
        self.tokenizer_load_time += max(0.0, float(tokenizer_load_time or 0.0))
        self.model_load_time += max(0.0, float(model_load_time or 0.0))
        self.text_tokenizer_load_time += max(
            0.0, float(text_tokenizer_load_time or 0.0)
        )

    def start_loop_timer(self) -> None:
        """Exclude setup/model-load time from tokenization throughput."""
        if self.loop_timing_started:
            return
        now = time.time()
        self.setup_elapsed_time += max(0.0, now - self.start_time)
        self.start_time = now
        self.loop_timing_started = True

    def current_elapsed_time(self) -> float:
        """Return measured tokenization-loop time across resume segments."""
        return self.elapsed_offset + max(0.0, time.time() - self.start_time)

    def current_total_elapsed_time(self) -> float:
        """Return setup plus tokenization-loop wall time."""
        return self.setup_elapsed_time + self.current_elapsed_time()

    def to_dict(self) -> Dict[str, Any]:
        elapsed = self.current_elapsed_time()
        throughput = self.tokens_generated / elapsed if elapsed > 0 else 0
        image_tokens_per_second = self.image_tokens / elapsed if elapsed > 0 else 0
        total_elapsed = self.current_total_elapsed_time()
        d = {
            "samples_processed": self.samples_processed,
            "tokens_generated": self.tokens_generated,
            "image_tokens": self.image_tokens,
            "text_tokens": self.text_tokens,
            "errors": self.errors,
            "samples_skipped": self.samples_skipped,
            "cuda_oom_errors": self.cuda_oom_errors,
            "elapsed_time": elapsed,
            "setup_elapsed_time": self.setup_elapsed_time,
            "total_elapsed_time": total_elapsed,
            "tokenizer_load_time": self.tokenizer_load_time,
            "model_load_time": self.model_load_time,
            "text_tokenizer_load_time": self.text_tokenizer_load_time,
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
        self.elapsed_time = float(data.get("elapsed_time", 0.0) or 0.0)
        self.elapsed_offset = self.elapsed_time
        self.setup_elapsed_time = float(data.get("setup_elapsed_time", 0.0) or 0.0)
        self.tokenizer_load_time = float(data.get("tokenizer_load_time", 0.0) or 0.0)
        self.model_load_time = float(data.get("model_load_time", 0.0) or 0.0)
        self.text_tokenizer_load_time = float(
            data.get("text_tokenizer_load_time", 0.0) or 0.0
        )
        self.loop_timing_started = False
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
    from vision_tokenization.formats.megatron import DType, IndexedDatasetBuilder

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


CHECKPOINT_VERSION = 2


def save_checkpoint(
    output_dir: str,
    rank: int,
    batch_index: int,
    writer_state: Dict[str, Any],
    plan_fingerprint: Optional[Dict[str, Any]],
    stats: Dict[str, Any],
    world_size: int = 1,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomically save checkpoint via ``.tmp`` + ``os.replace()``.

    The checkpoint must capture every cursor needed to resume:
    *writer_state* is an opaque dict owned by the writer (round-tripped,
    never interpreted here), *plan_fingerprint* identifies the plan the
    batch_index was counted against (resume refuses on mismatch).
    """
    import torch

    ckpt_path = _checkpoint_path(output_dir, rank)
    tmp_path = str(ckpt_path) + ".tmp"
    payload = {
        "version": CHECKPOINT_VERSION,
        "batch_index": batch_index,
        "writer": dict(writer_state),
        "plan": plan_fingerprint,
        "stats": stats,
        "world_size": world_size,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, tmp_path)
    os.replace(tmp_path, str(ckpt_path))
    logger.debug(f"[rank {rank}] Saved checkpoint batch_index={batch_index}, writer={writer_state}")


def load_checkpoint(output_dir: str, rank: int) -> Optional[Dict[str, Any]]:
    """Load this rank's checkpoint if it exists. v2 only.

    Pre-v2 checkpoints (before writer-owned cursor state and plan
    fingerprints) are refused: re-tokenize the dataset with current code.
    """
    import torch

    ckpt_path = _checkpoint_path(output_dir, rank)
    if not ckpt_path.exists():
        return None
    logger.info(f"[rank {rank}] Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if ckpt.get("version", 1) < CHECKPOINT_VERSION:
        raise RuntimeError(
            f"[rank {rank}] Checkpoint at {ckpt_path} predates the v2 resume "
            f"protocol and cannot be resumed safely. Re-tokenize this dataset "
            f"from scratch (resume=false, fresh output_dir)."
        )
    return ckpt


MANIFEST_VERSION = 1


def _manifest_path(output_dir: str, rank: int) -> Path:
    return Path(output_dir) / f"rank_{rank:04d}_DONE.json"


def write_rank_manifest(
    output_dir: str,
    rank: int,
    world_size: int,
    plan_fingerprint: Optional[Dict[str, Any]],
    backend: str,
    files: list,
) -> None:
    """Publish this rank's completion claim — the LAST act of a successful run.

    *files* is the writer's own record of every final shard it shipped
    ({name, bytes, sequences, tokens}); the merge gate verifies the claim
    against disk instead of inferring completeness from markers and globs.
    """
    payload = {
        "version": MANIFEST_VERSION,
        "rank": rank,
        "world_size": world_size,
        "plan": plan_fingerprint,
        "backend": backend,
        "files": files,
        "sequences": sum(f["sequences"] for f in files),
        "tokens": sum(f["tokens"] for f in files),
    }
    path = _manifest_path(output_dir, rank)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)
    logger.info(f"[rank {rank}] Completion manifest: {len(files)} shards, "
                f"{payload['sequences']:,} seqs, {payload['tokens']:,} tokens")


def load_rank_manifests(output_dir) -> list:
    """All rank completion manifests in *output_dir*, sorted by rank."""
    out = []
    for p in sorted(Path(output_dir).glob("rank_*_DONE.json")):
        with open(p) as f:
            out.append(json.load(f))
    return sorted(out, key=lambda m: m["rank"])


def verify_run_world_size(output_dir: str, world_size: int, rank: int) -> None:
    """Fail fast when ``output_dir`` belongs to a run with a different world size.

    A world-size change re-splits the plan across ranks, so each rank's
    shards and checkpoints describe different batch slices — reprocessing
    into the same directory would merge two generations (duplicated or
    dropped documents). The directory records its own world size; refuse
    with the exact resubmit size.
    """
    import torch

    recorded = set()
    for cp in sorted(Path(output_dir).glob("rank_*_checkpoint.pt")):
        ws = torch.load(str(cp), map_location="cpu", weights_only=False).get("world_size")
        if ws is not None:
            recorded.add(int(ws))
    stale = sorted({
        int(m.group(1))
        for f in Path(output_dir).glob("rank_*")
        if (m := re.match(r"rank_(\d{4})(?:[_.]|$)", f.name)) and int(m.group(1)) >= world_size
    })
    foreign = sorted(recorded - {world_size})
    if not stale and not foreign:
        return
    if len(foreign) > 1:
        raise RuntimeError(
            f"[rank {rank}] {output_dir} mixes artifacts from runs with world sizes "
            f"{foreign} — clean the directory and re-tokenize."
        )
    original = foreign[0] if foreign else stale[-1] + 1
    raise RuntimeError(
        f"[rank {rank}] {output_dir} was written by a {original}-rank run, but this job "
        f"has world_size={world_size}"
        + (f" (found artifacts for ranks {stale})" if stale else "")
        + f". Merging the two generations would duplicate documents. "
        f"Resubmit with num_gpus={original}, or use a fresh output_dir."
    )


def verify_plan_fingerprint(ckpt: Dict[str, Any], current: Dict[str, Any], rank: int) -> None:
    """Refuse resume when the plan no longer matches the checkpoint."""
    if ckpt.get("plan") != current:
        raise RuntimeError(
            f"[rank {rank}] Plan no longer matches this checkpoint "
            f"(checkpoint {ckpt['plan']} vs current {current}). The planner, "
            f"config, or manifest changed mid-run — finish with the original "
            f"code/config or restart this dataset."
        )
