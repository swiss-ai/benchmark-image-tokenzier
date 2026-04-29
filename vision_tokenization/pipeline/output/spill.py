"""Keyed component spill writer for the unified tokenization pipeline.

Writes per-rank output as keyed component payloads:

    rank_XXXX/
        components.NNNNNN.parquet   — (document_id, component_index, kind,
                                       token_offset, token_length, token_hash)
        tokens.NNNNNN.bin           — concatenated raw token bytes
        progress.NNNNNN.json        — checkpoint sidecar
        worker_stats.json           — aggregate stats
        _SUCCESS                    — written after clean finalization

No documents.parquet — ``TokenizationPlan`` is the document truth.
The rebuild joins spilled tokens back to the plan by
``(document_id, component_index)``.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vision_tokenization.utils.json import json_dump, json_load

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parquet schema
# ---------------------------------------------------------------------------

COMPONENTS_SCHEMA = pa.schema([
    pa.field("document_id", pa.int64()),
    pa.field("component_index", pa.int32()),
    pa.field("kind", pa.int8()),
    pa.field("token_offset", pa.int64()),
    pa.field("token_length", pa.int64()),
    pa.field("token_hash", pa.string()),
    pa.field("resize_height", pa.int32()),
    pa.field("resize_width", pa.int32()),
])


def _token_hash(tokens: np.ndarray) -> str:
    """Fast 8-byte hash of a token array for dedup validation."""
    return hashlib.blake2b(tokens.tobytes(), digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# Recovery helpers
# ---------------------------------------------------------------------------

def _collect_shard_ids(worker_dir: Path, stem: str, suffix: str) -> set:
    ids = set()
    for path in worker_dir.glob(f"{stem}.*.{suffix}"):
        parts = path.name.split(".")
        if len(parts) == 3:
            try:
                ids.add(int(parts[1]))
            except ValueError:
                pass
    return ids


def recover_worker_shards(worker_dir: Path) -> int:
    """Return next shard id from contiguous complete shards, cleaning debris."""
    if not worker_dir.exists():
        return 0

    comp_ids = _collect_shard_ids(worker_dir, "components", "parquet")
    token_ids = _collect_shard_ids(worker_dir, "tokens", "bin")
    complete = comp_ids & token_ids

    next_id = 0
    while next_id in complete:
        next_id += 1

    # Clean dangling shards
    all_ids = comp_ids | token_ids | _collect_shard_ids(worker_dir, "progress", "json")
    for sid in sorted(all_ids - set(range(next_id))):
        for path in (
            worker_dir / f"components.{sid:06d}.parquet",
            worker_dir / f"tokens.{sid:06d}.bin",
            worker_dir / f"progress.{sid:06d}.json",
        ):
            if path.exists():
                path.unlink()
                logger.warning(f"Removed incomplete shard file: {path}")

    return next_id


def write_shard_progress(
    worker_dir: Path,
    shard_id: int,
    *,
    next_batch_index: int,
    stats: dict,
) -> None:
    """Atomically record progress for one flushed spill shard."""
    progress_path = worker_dir / f"progress.{shard_id:06d}.json"
    tmp_path = progress_path.with_suffix(".json.tmp")
    json_dump({
        "shard_id": int(shard_id),
        "next_batch_index": int(next_batch_index),
        "stats": dict(stats),
    }, tmp_path)
    os.replace(tmp_path, progress_path)


def recover_shard_progress(worker_dir: Path, num_shards: int) -> dict:
    """Recover latest contiguous progress state."""
    latest = None
    next_id = 0
    for sid in range(num_shards):
        path = worker_dir / f"progress.{sid:06d}.json"
        if not path.exists():
            break
        latest = json_load(path)
        next_id = sid + 1

    return {
        "next_shard_id": next_id,
        "next_batch_index": 0 if latest is None else int(latest["next_batch_index"]),
        "stats": {} if latest is None else dict(latest.get("stats", {})),
    }


# ---------------------------------------------------------------------------
# ComponentSpillWriter
# ---------------------------------------------------------------------------


class ComponentSpillWriter:
    """Write keyed component token payloads to disk.

    Usage::

        writer = ComponentSpillWriter(output_dir, rank)
        writer.open()

        writer.add_component(
            document_id=42, component_index=0, kind=IMAGE,
            tokens=vision_tokens, resize_height=512, resize_width=512,
        )
        writer.add_component(
            document_id=42, component_index=1, kind=TEXT,
            tokens=text_tokens,
        )

        writer.checkpoint()   # flush shard, start new one
        writer.finalize()     # flush + write stats + _SUCCESS
    """

    def __init__(
        self,
        output_dir: str,
        rank: int,
        token_dtype: np.dtype = np.int32,
    ):
        self._base_dir = Path(output_dir) / f"rank_{rank:04d}"
        self._rank = rank
        self._token_dtype = np.dtype(token_dtype)

        self._shard_id = 0
        self._shards_flushed = 0
        self._comp_rows: list[dict] = []
        self._token_file: Optional[object] = None
        self._token_offset: int = 0
        self._components_written: int = 0
        self._total_tokens: int = 0

    def open(self, start_shard_id: int = 0) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._shard_id = start_shard_id
        self._open_shard()

    @property
    def shard_id(self) -> int:
        return self._shard_id

    @property
    def has_pending(self) -> bool:
        return bool(self._comp_rows)

    def _open_shard(self) -> None:
        prefix = f"{self._shard_id:06d}"
        self._token_file = open(self._base_dir / f"tokens.{prefix}.bin", "wb")
        self._token_offset = 0
        self._comp_rows = []

    def add_component(
        self,
        document_id: int,
        component_index: int,
        kind: int,
        tokens: np.ndarray,
        resize_height: int = 0,
        resize_width: int = 0,
    ) -> None:
        """Write one component's tokens and record metadata."""
        tokens = np.asarray(tokens, dtype=self._token_dtype)
        token_bytes = tokens.tobytes()
        self._token_file.write(token_bytes)

        self._comp_rows.append({
            "document_id": int(document_id),
            "component_index": int(component_index),
            "kind": int(kind),
            "token_offset": self._token_offset,
            "token_length": len(tokens),
            "token_hash": _token_hash(tokens),
            "resize_height": int(resize_height),
            "resize_width": int(resize_width),
        })

        self._token_offset += len(token_bytes)
        self._components_written += 1
        self._total_tokens += len(tokens)

    def checkpoint(self) -> int:
        """Flush current shard and start a new one. Returns flushed shard_id."""
        self._flush_shard()
        done = self._shard_id
        self._shard_id += 1
        self._open_shard()
        return done

    def finalize(self) -> None:
        """Flush last shard, write stats, mark success."""
        if self._comp_rows:
            self._flush_shard()
        elif self._token_file is not None:
            self._token_file.close()
            # Remove empty token file
            empty_path = self._base_dir / f"tokens.{self._shard_id:06d}.bin"
            if empty_path.exists() and empty_path.stat().st_size == 0:
                empty_path.unlink()

        json_dump({
            "rank": self._rank,
            "shards_written": self._shards_flushed,
            "components_written": self._components_written,
            "total_tokens": self._total_tokens,
            "token_dtype": str(self._token_dtype),
        }, self._base_dir / "worker_stats.json")

        (self._base_dir / "_SUCCESS").touch()

        logger.info(
            f"[rank {self._rank}] Spill finalized: {self._components_written:,} components, "
            f"{self._total_tokens:,} tokens, {self._shards_flushed} shards -> {self._base_dir}"
        )

    def _flush_shard(self) -> None:
        prefix = f"{self._shard_id:06d}"

        if self._token_file is not None:
            self._token_file.flush()
            os.fsync(self._token_file.fileno())
            self._token_file.close()
            self._token_file = None

        if self._comp_rows:
            table = pa.table(
                {field.name: [r[field.name] for r in self._comp_rows]
                 for field in COMPONENTS_SCHEMA},
                schema=COMPONENTS_SCHEMA,
            )
            pq.write_table(
                table,
                self._base_dir / f"components.{prefix}.parquet",
                compression="zstd",
            )

        self._comp_rows = []
        self._shards_flushed += 1


# ---------------------------------------------------------------------------
# ComponentSpillReader
# ---------------------------------------------------------------------------


class ComponentSpillReader:
    """Read spilled component payloads from all ranks."""

    @staticmethod
    def read_rank(rank_dir: Path) -> pa.Table:
        """Read all component parquets from one rank directory."""
        files = sorted(rank_dir.glob("components.*.parquet"))
        if not files:
            return pa.table([], schema=COMPONENTS_SCHEMA)
        return pa.concat_tables([pq.read_table(f) for f in files])

    @staticmethod
    def read_all_ranks(output_dir: Path) -> pa.Table:
        """Read components from all rank directories."""
        output_dir = Path(output_dir)
        rank_dirs = sorted(p for p in output_dir.glob("rank_*") if p.is_dir())
        if not rank_dirs:
            raise FileNotFoundError(f"No rank directories found in {output_dir}")

        tables = []
        for rd in rank_dirs:
            if not (rd / "_SUCCESS").exists():
                logger.warning(f"Skipping rank {rd.name}: no _SUCCESS marker")
                continue
            tables.append(ComponentSpillReader.read_rank(rd))

        if not tables:
            raise FileNotFoundError(f"No complete rank directories in {output_dir}")
        return pa.concat_tables(tables)

    @staticmethod
    def load_tokens(
        rank_dir: Path,
        shard_id: int,
        token_offset: int,
        token_length: int,
        token_dtype: np.dtype = np.int32,
    ) -> np.ndarray:
        """Load one component's tokens from tokens.bin."""
        dtype = np.dtype(token_dtype)
        path = rank_dir / f"tokens.{shard_id:06d}.bin"
        with open(path, "rb") as f:
            f.seek(token_offset)
            data = f.read(token_length * dtype.itemsize)
        return np.frombuffer(data, dtype=dtype).copy()
