"""Shard-local alignment payload schema and split helpers.

The public alignment artifact is row-local metadata plus flat mmap-able token
sidecars:

  views/*.parquet  -> text + per-row image token offsets
  tokens/*.i32     -> flat little-endian int32 image tokens
  raw/*.parquet    -> cold raw bytes for judges/audit
"""

from __future__ import annotations

import hashlib
import numpy as np
import pyarrow as pa

TOKEN_DTYPE = np.dtype("<i4")

MESSAGE_TYPE = pa.list_(
    pa.struct([
        pa.field("role", pa.string()),
        pa.field("content", pa.string()),
    ])
)

IMAGE_REF_TYPE = pa.list_(
    pa.struct([
        pa.field("media_id", pa.string()),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
        pa.field("resize_height", pa.int32()),
        pa.field("resize_width", pa.int32()),
        pa.field("token_offset", pa.int64()),
        pa.field("token_length", pa.int32()),
        pa.field("raw_row", pa.int32()),
    ])
)

VIEW_SCHEMA = pa.schema([
    pa.field("prompt", MESSAGE_TYPE),
    pa.field("chosen", pa.string()),
    pa.field("rejected", pa.string()),
    pa.field("prompt_id", pa.string()),
    pa.field("text_chars", pa.int64()),
    pa.field("media_tokens_total", pa.int64()),
    pa.field("images", IMAGE_REF_TYPE),
])

RAW_SCHEMA = pa.schema([
    pa.field("sample_id", pa.string()),
    pa.field("image_index", pa.int16()),
    pa.field("media_id", pa.string()),
    pa.field("width", pa.int32()),
    pa.field("height", pa.int32()),
    pa.field("resize_height", pa.int32()),
    pa.field("resize_width", pa.int32()),
    pa.field("raw_ext", pa.string()),
    pa.field("raw_bytes", pa.large_binary()),
])


def _validation_target_count(n_rows: int, requested_validation_rows: int) -> int:
    if requested_validation_rows <= 0 or n_rows < 2:
        return 0
    return min(requested_validation_rows, max(1, n_rows // 50), n_rows - 1)


def _split_group_key(row: dict, row_idx: int, split_key: str) -> str:
    key = row.get(split_key) or row.get("prompt_id")
    return str(key) if key else f"row:{row_idx}"


def _split_score(seed: int, group_key: str) -> bytes:
    return hashlib.sha256(f"{seed}:{group_key}".encode("utf-8")).digest()


def split_payload_rows(
    rows: list[dict],
    *,
    requested_validation_rows: int,
    split_key: str = "prompt_id",
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Deterministic group-preserving train/validation split."""
    target = _validation_target_count(len(rows), int(requested_validation_rows))
    if target == 0:
        return list(rows), []

    groups: dict[str, int] = {}
    row_keys = []
    for idx, row in enumerate(rows):
        key = _split_group_key(row, idx, split_key)
        row_keys.append(key)
        groups[key] = groups.get(key, 0) + 1

    validation_keys: set[str] = set()
    selected = 0
    for key, count in sorted(
        groups.items(),
        key=lambda item: (_split_score(seed, item[0]), item[0]),
    ):
        validation_keys.add(key)
        selected += count
        if selected >= target:
            break

    train = [row for row, key in zip(rows, row_keys) if key not in validation_keys]
    validation = [row for row, key in zip(rows, row_keys) if key in validation_keys]
    return train, validation


def _alignment_media_refs(row: dict) -> list[str]:
    return [
        *list(row.get("prompt_media_refs") or []),
        *list(row.get("chosen_media_refs") or []),
        *list(row.get("rejected_media_refs") or []),
    ]

