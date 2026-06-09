"""Keyed component spill writer for the unified tokenization pipeline.

Writes per-rank output as keyed component payloads:

    rank_XXXX/
        components.NNNNNN.parquet   — (document_id, component_index, kind,
                                       token_offset, token_length)
        tokens.NNNNNN.bin           — concatenated raw token bytes
        worker_stats.json           — aggregate stats
        _SUCCESS                    — written by the backend layer after
                                       clean finalization (see ``backend.py``
                                       ``_write_rank_success_marker``)

No documents.parquet — ``TokenizationPlan`` is the document truth.
The rebuild joins spilled tokens back to the plan by
``(document_id, component_index)``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vision_tokenization.utils.json import json_dump

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
    pa.field("resize_height", pa.int32()),
    pa.field("resize_width", pa.int32()),
])


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
    all_ids = comp_ids | token_ids
    for sid in sorted(all_ids - set(range(next_id))):
        for path in (
            worker_dir / f"components.{sid:06d}.parquet",
            worker_dir / f"tokens.{sid:06d}.bin",
        ):
            if path.exists():
                path.unlink()
                logger.warning(f"Removed incomplete shard file: {path}")

    return next_id


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
        writer.finalize()     # flush + write stats (backend writes _SUCCESS)
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
        """Flush last shard and write worker stats.

        Does NOT write the ``_SUCCESS`` marker — that is owned by the backend
        layer (see ``SpillBackend.finalize`` / ``DirectBackend.finalize``) so
        both backends produce the same terminal marker contract.
        """
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

    # Read only schema columns so shards written before a schema change
    # (e.g. ones still carrying the retired token_hash column) concat cleanly.
    _COLUMNS = [field.name for field in COMPONENTS_SCHEMA]

    @staticmethod
    def read_rank(rank_dir: Path) -> pa.Table:
        """Read all component parquets from one rank directory."""
        files = sorted(rank_dir.glob("components.*.parquet"))
        if not files:
            return pa.table([], schema=COMPONENTS_SCHEMA)
        return pa.concat_tables(
            [pq.read_table(f, columns=ComponentSpillReader._COLUMNS) for f in files]
        )

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
