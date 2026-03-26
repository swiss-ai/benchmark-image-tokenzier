"""SHAR-like spill writer for the pooled tokenization pipeline.

Writes per-rank output in a document-centric format:

    worker_XX/
        documents.NNNNNN.parquet   — one row per logical document
        components.NNNNNN.parquet  — one row per component
        tokens.NNNNNN.bin          — concatenated raw token bytes
        worker_stats.json          — aggregate stats
        _SUCCESS                   — written after clean finalization

Parquet for metadata only.  Raw uncompressed binary for token payload.
No giant token blobs as Parquet large_binary.
"""

from __future__ import annotations

from vision_tokenization.utils.json import json_dump, json_load
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .document import AtomicDocument, Component

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parquet schemas
# ---------------------------------------------------------------------------

DOCUMENTS_SCHEMA = pa.schema([
    pa.field("document_id", pa.int64()),
    pa.field("mode", pa.string()),
    pa.field("num_components", pa.int32()),
    pa.field("total_tokens", pa.int64()),
    pa.field("image_tokens", pa.int64()),
    pa.field("text_tokens", pa.int64()),
    pa.field("manifest_group_id", pa.int64()),
])

COMPONENTS_SCHEMA = pa.schema([
    pa.field("document_id", pa.int64()),
    pa.field("component_index", pa.int32()),
    pa.field("kind", pa.string()),
    pa.field("token_offset", pa.int64()),
    pa.field("token_length", pa.int64()),
    pa.field("resize_height", pa.int32()),
    pa.field("resize_width", pa.int32()),
    pa.field("manifest_row", pa.int64()),
])


def _collect_shard_ids(worker_dir: Path, stem: str, suffix: str) -> set[int]:
    shard_ids: set[int] = set()
    for path in worker_dir.glob(f"{stem}.*.{suffix}"):
        parts = path.name.split(".")
        if len(parts) != 3:
            continue
        try:
            shard_ids.add(int(parts[1]))
        except ValueError:
            continue
    return shard_ids


def _remove_shard_files(worker_dir: Path, shard_id: int) -> bool:
    removed = False
    for path in (
        worker_dir / f"documents.{shard_id:06d}.parquet",
        worker_dir / f"components.{shard_id:06d}.parquet",
        worker_dir / f"tokens.{shard_id:06d}.bin",
        worker_dir / f"progress.{shard_id:06d}.json",
    ):
        if path.exists():
            path.unlink()
            removed = True
    return removed


def recover_worker_shards(worker_dir: str | Path) -> int:
    """Return next shard id from contiguous flushed shards, cleaning tail debris.

    A shard counts as durable only when all three files exist:
    ``documents.NNNNNN.parquet``, ``components.NNNNNN.parquet``, and
    ``tokens.NNNNNN.bin``.
    Any higher-numbered partial or non-contiguous files are removed so resume
    can safely reopen the next shard in-place.
    """
    worker_dir = Path(worker_dir)
    if not worker_dir.exists():
        return 0

    doc_ids = _collect_shard_ids(worker_dir, "documents", "parquet")
    comp_ids = _collect_shard_ids(worker_dir, "components", "parquet")
    token_ids = _collect_shard_ids(worker_dir, "tokens", "bin")
    progress_ids = _collect_shard_ids(worker_dir, "progress", "json")

    complete = doc_ids & comp_ids & token_ids
    next_shard_id = 0
    while next_shard_id in complete:
        next_shard_id += 1

    valid_ids = set(range(next_shard_id))
    dangling_ids = (doc_ids | comp_ids | token_ids | progress_ids) - valid_ids
    for shard_id in sorted(dangling_ids):
        removed = _remove_shard_files(worker_dir, shard_id)
        if removed:
            logger.warning(
                "Removed incomplete pooled spill shard %06d under %s during resume recovery",
                shard_id,
                worker_dir,
            )

    return next_shard_id


def write_shard_progress(
    worker_dir: str | Path,
    shard_id: int,
    *,
    next_document_window_index: int,
    stats: dict,
) -> None:
    """Atomically record pooled progress for one flushed spill shard."""
    worker_dir = Path(worker_dir)
    progress_path = worker_dir / f"progress.{shard_id:06d}.json"
    tmp_path = progress_path.with_suffix(".json.tmp")
    payload = {
        "shard_id": int(shard_id),
        "next_document_window_index": int(next_document_window_index),
        "stats": dict(stats),
    }
    json_dump(payload, tmp_path)
    os.replace(tmp_path, progress_path)


def recover_shard_progress(worker_dir: str | Path, num_shards: int) -> dict:
    """Recover the latest contiguous pooled progress state.

    Shards without a matching progress sidecar are treated as uncommitted and
    removed so resume always restarts from a coherent window/shard boundary.
    """
    worker_dir = Path(worker_dir)
    next_shard_id = 0
    latest_progress = None

    for shard_id in range(num_shards):
        progress_path = worker_dir / f"progress.{shard_id:06d}.json"
        if not progress_path.exists():
            for dangling_id in range(shard_id, num_shards):
                removed = _remove_shard_files(worker_dir, dangling_id)
                if removed:
                    logger.warning(
                        "Removed pooled spill shard %06d under %s because its progress "
                        "sidecar was missing during resume recovery",
                        dangling_id,
                        worker_dir,
                    )
            break

        latest_progress = json_load(progress_path)
        next_shard_id = shard_id + 1

    return {
        "next_shard_id": next_shard_id,
        "next_document_window_index": (
            0 if latest_progress is None
            else int(latest_progress["next_document_window_index"])
        ),
        "stats": {} if latest_progress is None else dict(latest_progress.get("stats", {})),
    }


def summarize_worker_shards(worker_dir: str | Path, num_shards: int) -> dict[str, int]:
    """Summarize durable spill progress from flushed document parquet shards."""
    worker_dir = Path(worker_dir)
    totals = {
        "documents_written": 0,
        "total_tokens": 0,
        "image_tokens": 0,
        "text_tokens": 0,
    }

    for shard_id in range(num_shards):
        doc_path = worker_dir / f"documents.{shard_id:06d}.parquet"
        if not doc_path.exists():
            break
        table = pq.read_table(
            doc_path,
            columns=["total_tokens", "image_tokens", "text_tokens"],
        )
        totals["documents_written"] += table.num_rows
        totals["total_tokens"] += int(sum(table.column("total_tokens").to_pylist()))
        totals["image_tokens"] += int(sum(table.column("image_tokens").to_pylist()))
        totals["text_tokens"] += int(sum(table.column("text_tokens").to_pylist()))

    return totals


# ---------------------------------------------------------------------------
# SpillWriter
# ---------------------------------------------------------------------------

class SpillWriter:
    """Write atomic documents to SHAR-like shards on disk.

    Usage::

        writer = SpillWriter(output_dir, rank, token_dtype=np.uint16)
        writer.open()

        for doc in documents:
            writer.add_document(doc, component_tokens)

        # Periodic checkpoint (starts a new shard)
        writer.checkpoint()

        writer.finalize()
    """

    def __init__(
        self,
        output_dir: str,
        rank: int,
        token_dtype: np.dtype = np.int32,
        flush_every: int = 10_000,
    ):
        self._base_dir = Path(output_dir) / f"worker_{rank:02d}"
        self._rank = rank
        self._token_dtype = np.dtype(token_dtype)
        self._flush_every = flush_every

        self._shard_id = 0
        self._shards_flushed = 0
        self._doc_rows: list[dict] = []
        self._comp_rows: list[dict] = []
        self._token_file: Optional[object] = None
        self._token_offset: int = 0
        self._docs_written: int = 0
        self._total_tokens: int = 0
        self._is_open = False

    def open(self, start_shard_id: int = 0) -> None:
        """Initialize the writer and create the output directory."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._shard_id = start_shard_id
        self._open_shard()
        self._is_open = True

    @property
    def shard_id(self) -> int:
        return self._shard_id

    @property
    def has_pending_documents(self) -> bool:
        return bool(self._doc_rows)

    def _shard_prefix(self) -> str:
        return f"{self._shard_id:06d}"

    def _open_shard(self) -> None:
        prefix = self._shard_prefix()
        token_path = self._base_dir / f"tokens.{prefix}.bin"
        self._token_file = open(token_path, "wb")
        self._token_offset = 0
        self._doc_rows = []
        self._comp_rows = []

    def add_document(
        self,
        doc: AtomicDocument,
        component_tokens: List[np.ndarray],
    ) -> None:
        """Add one atomic document and its component token arrays.

        Args:
            doc: Document metadata.
            component_tokens: Ordered list of numpy arrays (one per component),
                each containing token IDs in the writer's dtype.  Must match
                ``doc.components`` in length and order.
        """
        if len(component_tokens) != len(doc.components):
            raise ValueError(
                f"Document {doc.document_id}: expected {len(doc.components)} "
                f"component token arrays, got {len(component_tokens)}"
            )

        for comp, tokens in zip(doc.components, component_tokens):
            token_bytes = np.asarray(tokens, dtype=self._token_dtype).tobytes()
            self._token_file.write(token_bytes)

            comp.token_offset = self._token_offset
            comp.token_length = len(tokens)
            self._token_offset += len(token_bytes)

            self._comp_rows.append({
                "document_id": doc.document_id,
                "component_index": comp.component_index,
                "kind": comp.kind,
                "token_offset": comp.token_offset,
                "token_length": comp.token_length,
                "resize_height": comp.resize_height,
                "resize_width": comp.resize_width,
                "manifest_row": comp.manifest_row,
            })

        self._doc_rows.append({
            "document_id": doc.document_id,
            "mode": doc.mode,
            "num_components": doc.num_components,
            "total_tokens": doc.total_tokens,
            "image_tokens": doc.image_tokens,
            "text_tokens": doc.text_tokens,
            "manifest_group_id": doc.manifest_group_id,
        })

        self._docs_written += 1
        self._total_tokens += doc.total_tokens

    def checkpoint(self) -> int:
        """Flush current shard and start a new one.

        Returns:
            The finalized shard_id.
        """
        self._flush_shard()
        done = self._shard_id
        self._shard_id += 1
        self._open_shard()
        return done

    def finalize(self) -> None:
        """Flush the last shard, write stats, and mark success."""
        if self._doc_rows:
            self._flush_shard()
        elif self._token_file is not None:
            self._token_file.close()
            # Remove empty token file
            prefix = self._shard_prefix()
            empty_path = self._base_dir / f"tokens.{prefix}.bin"
            if empty_path.exists() and empty_path.stat().st_size == 0:
                empty_path.unlink()

        # Write aggregate stats
        stats = {
            "rank": self._rank,
            "shards_written": self._shards_flushed,
            "documents_written": self._docs_written,
            "total_tokens": self._total_tokens,
            "token_dtype": str(self._token_dtype),
        }
        json_dump(stats, self._base_dir / "worker_stats.json")

        # Success marker
        (self._base_dir / "_SUCCESS").touch()
        self._is_open = False

        logger.info(
            f"[rank {self._rank}] Spill finalized: {self._docs_written:,} documents, "
            f"{self._total_tokens:,} tokens, {self._shard_id + 1} shards -> {self._base_dir}"
        )

    def _flush_shard(self) -> None:
        """Write parquet metadata and close token file for current shard."""
        prefix = self._shard_prefix()

        # Close token file
        if self._token_file is not None:
            self._token_file.flush()
            os.fsync(self._token_file.fileno())
            self._token_file.close()
            self._token_file = None

        # Write documents parquet
        if self._doc_rows:
            doc_table = pa.table(
                {field.name: [r[field.name] for r in self._doc_rows] for field in DOCUMENTS_SCHEMA},
                schema=DOCUMENTS_SCHEMA,
            )
            pq.write_table(
                doc_table,
                self._base_dir / f"documents.{prefix}.parquet",
                compression="zstd",
            )

        # Write components parquet
        if self._comp_rows:
            comp_table = pa.table(
                {field.name: [r[field.name] for r in self._comp_rows] for field in COMPONENTS_SCHEMA},
                schema=COMPONENTS_SCHEMA,
            )
            pq.write_table(
                comp_table,
                self._base_dir / f"components.{prefix}.parquet",
                compression="zstd",
            )

        self._doc_rows = []
        self._comp_rows = []
        self._shards_flushed += 1


# ---------------------------------------------------------------------------
# SpillReader — reads back spill shards for offline rebuild
# ---------------------------------------------------------------------------

class SpillReader:
    """Read spill shards produced by SpillWriter.

    Reads from a single worker directory or aggregates across multiple workers.
    """

    @staticmethod
    def read_worker(worker_dir: str | Path) -> tuple[pa.Table, pa.Table, Path]:
        """Read all shard parquets from one worker directory.

        Returns:
            (documents_table, components_table, worker_dir_path)
        """
        worker_dir = Path(worker_dir)
        doc_files = sorted(worker_dir.glob("documents.*.parquet"))
        comp_files = sorted(worker_dir.glob("components.*.parquet"))

        doc_tables = [pq.read_table(f) for f in doc_files]
        comp_tables = [pq.read_table(f) for f in comp_files]

        documents = pa.concat_tables(doc_tables) if doc_tables else pa.table([], schema=DOCUMENTS_SCHEMA)
        components = pa.concat_tables(comp_tables) if comp_tables else pa.table([], schema=COMPONENTS_SCHEMA)

        return documents, components, worker_dir

    @staticmethod
    def read_all_workers(output_dir: str | Path) -> tuple[pa.Table, pa.Table, list[Path]]:
        """Read spill from all worker directories under output_dir.

        Returns:
            (documents_table, components_table, list_of_worker_dirs)
        """
        output_dir = Path(output_dir)
        worker_dirs = sorted(output_dir.glob("worker_*"))
        if not worker_dirs:
            raise FileNotFoundError(f"No worker directories found in {output_dir}")

        all_docs = []
        all_comps = []
        all_dirs = []

        for wd in worker_dirs:
            if not (wd / "_SUCCESS").exists():
                logger.warning(f"Skipping incomplete worker: {wd}")
                continue
            docs, comps, path = SpillReader.read_worker(wd)
            all_docs.append(docs)
            all_comps.append(comps)
            all_dirs.append(path)

        documents = pa.concat_tables(all_docs) if all_docs else pa.table([], schema=DOCUMENTS_SCHEMA)
        components = pa.concat_tables(all_comps) if all_comps else pa.table([], schema=COMPONENTS_SCHEMA)

        return documents, components, all_dirs

    @staticmethod
    def load_component_tokens(
        worker_dir: str | Path,
        shard_id: int,
        token_offset: int,
        token_length: int,
        token_dtype: np.dtype = np.int32,
    ) -> np.ndarray:
        """Load one component's tokens from a tokens.bin file.

        Args:
            worker_dir: Path to the worker directory.
            shard_id: Shard index (determines which tokens.NNNNNN.bin to read).
            token_offset: Byte offset into the token file.
            token_length: Number of tokens to read.
            token_dtype: Dtype of the stored tokens.

        Returns:
            1-D numpy array of token IDs.
        """
        worker_dir = Path(worker_dir)
        token_path = worker_dir / f"tokens.{shard_id:06d}.bin"
        dtype = np.dtype(token_dtype)
        with open(token_path, "rb") as f:
            f.seek(token_offset)
            data = f.read(token_length * dtype.itemsize)
        return np.frombuffer(data, dtype=dtype).copy()
