"""Content-addressed media store: sealed (tokens.bin, raw.bin, media.parquet) triples.

Units contract (spec): offset_elems/length_elems are TOKEN ELEMENTS; raw_* are
bytes. Writer is append-only within one shard triple; ``seal()`` makes it
immutable and returns ``{relpath: byte_size}`` for the dataset manifest (the
commit record, written last via ``atomic_write_json``).

Note: stdlib ``json`` on purpose — this module is the on-disk format contract
and is copied verbatim into the NeMo-RL consumer, which must not inherit this
repo's optional dependencies.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TOKEN_DTYPE = np.dtype("<i4")

MEDIA_SCHEMA = pa.schema([
    pa.field("media_id", pa.string()),
    pa.field("shard", pa.int32()),
    pa.field("offset_elems", pa.int64()),
    pa.field("length_elems", pa.int64()),
    pa.field("raw_offset_bytes", pa.int64()),
    pa.field("raw_length_bytes", pa.int64()),
    pa.field("raw_ext", pa.string()),
    pa.field("resize_h", pa.int32()),
    pa.field("resize_w", pa.int32()),
    pa.field("kind", pa.string()),
    pa.field("source", pa.string()),
])


def atomic_write_json(path: Path, obj: dict) -> None:
    """Write JSON via tmp + fsync + ``os.replace`` (manifest commit record)."""
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class MediaStoreWriter:
    def __init__(self, root: Path, shard_id: int = 0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_id = shard_id
        self._tok_path = self.root / f"tokens.{shard_id:06d}.bin"
        self._raw_path = self.root / f"raw.{shard_id:06d}.bin"
        self._pq_path = self.root / f"media.{shard_id:06d}.parquet"
        self._tok_f = open(self._tok_path.with_suffix(".bin.tmp"), "wb")
        self._raw_f = open(self._raw_path.with_suffix(".bin.tmp"), "wb")
        self._rows: list[dict] = []
        self._seen: set[str] = set()
        self._tok_off = 0   # elements
        self._raw_off = 0   # bytes

    def add(self, media_id: str, *, tokens: np.ndarray, raw: bytes,
            resize_h: int, resize_w: int, kind: str, source: str,
            raw_ext: str) -> None:
        if media_id in self._seen:
            raise ValueError(f"duplicate media_id: {media_id}")
        self._seen.add(media_id)
        tokens = np.ascontiguousarray(tokens, dtype=TOKEN_DTYPE)
        self._tok_f.write(tokens.tobytes())
        self._raw_f.write(raw)
        self._rows.append({
            "media_id": media_id, "shard": self.shard_id,
            "offset_elems": self._tok_off, "length_elems": len(tokens),
            "raw_offset_bytes": self._raw_off, "raw_length_bytes": len(raw),
            "raw_ext": raw_ext, "resize_h": resize_h, "resize_w": resize_w,
            "kind": kind, "source": source,
        })
        self._tok_off += len(tokens)
        self._raw_off += len(raw)

    def seal(self) -> dict[str, int]:
        for f, final in ((self._tok_f, self._tok_path), (self._raw_f, self._raw_path)):
            f.flush()
            os.fsync(f.fileno())
            f.close()
            os.replace(str(final) + ".tmp", final)
        table = pa.Table.from_pylist(self._rows, schema=MEDIA_SCHEMA)
        tmp = self._pq_path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        os.replace(tmp, self._pq_path)
        return {p.name: p.stat().st_size
                for p in (self._pq_path, self._tok_path, self._raw_path)}


class MediaStoreReader:
    """Union index over one or more media roots. Init-only IO (spec contract)."""

    def __init__(self, roots: list[Path], token_dtype: str = "<i4",
                 load_to_ram_threshold_bytes: int = 16 << 30):
        if np.dtype(token_dtype) != TOKEN_DTYPE:
            raise ValueError(f"unsupported token_dtype {token_dtype!r}; expected <i4")
        self.index: dict[str, tuple[int, int, int]] = {}   # id -> (arena_idx, off, len)
        self._arenas: list[np.ndarray] = []
        self._raw_index: dict[str, tuple[Path, int, int]] = {}
        for root in (Path(r) for r in roots):
            for pq_file in sorted(root.glob("media.*.parquet")):
                shard = int(pq_file.name.split(".")[1])
                tok_path = root / f"tokens.{shard:06d}.bin"
                raw_path = root / f"raw.{shard:06d}.bin"
                size = tok_path.stat().st_size
                if size <= load_to_ram_threshold_bytes:
                    arena = np.fromfile(tok_path, dtype=TOKEN_DTYPE)
                else:
                    arena = np.memmap(tok_path, dtype=TOKEN_DTYPE, mode="r")
                a_idx = len(self._arenas)
                self._arenas.append(arena)
                for row in pq.read_table(pq_file).to_pylist():
                    mid = row["media_id"]
                    if mid in self.index:
                        raise ValueError(f"duplicate media_id across roots: {mid}")
                    self.index[mid] = (a_idx, row["offset_elems"], row["length_elems"])
                    self._raw_index[mid] = (raw_path, row["raw_offset_bytes"],
                                            row["raw_length_bytes"])

    def tokens(self, media_id: str) -> np.ndarray:
        a, off, ln = self.index[media_id]
        return np.asarray(self._arenas[a][off:off + ln])

    def raw(self, media_id: str) -> bytes:
        path, off, ln = self._raw_index[media_id]
        with open(path, "rb") as f:
            f.seek(off)
            return f.read(ln)
