"""Alignment-mode parquet ingest: hash+dedup images, scan geometry, draft views.

The input marker is the dataset-level ``<image>``; the canonical on-disk marker
is the tokenizer special ``<|image|>`` (id 131079) so the consumer can count it
in token space. Per-field marker counts must equal per-field ref counts (spec
contract).

Ingest IS the scan stage (pipeline contract: every mode persists geometry
before planning): the one pass that reads each image's bytes for sha256 also
decodes width/height. Corrupt/undecodable and sub-``SPATIAL_FACTOR`` images are
skipped here — with every view row referencing them — and the surviving
geometry is persisted as ``scan.parquet`` (``write_scan_parquet``) before any
GPU work. The planner consumes dims from the scan; the runner never decodes.

``ROW_ADAPTERS`` keys the per-task row parser + view schema (``task:`` in the
dataset yaml): ``preference`` today, ``rl_prompt`` reserved for P4. The media
store, planner, runner, and Gate 2 never branch on task.
"""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

MARKER = "<|image|>"
INPUT_MARKER = "<image>"
SPATIAL_FACTOR = 16

SCAN_SCHEMA = pa.schema([
    pa.field("media_id", pa.string()),
    pa.field("width", pa.int32()),
    pa.field("height", pa.int32()),
    pa.field("raw_length_bytes", pa.int64()),
    pa.field("source", pa.string()),
])


class MarkerMismatch(ValueError):
    pass


@dataclass
class UniqueMedia:
    media_id: str
    raw: bytes
    source: str
    raw_ext: str
    width: int       # source pixels; (0, 0) marks undecodable bytes
    height: int


@dataclass
class IngestResult:
    unique_media: list = field(default_factory=list)
    view_rows: list = field(default_factory=list)
    n_skipped_media: int = 0


def _normalize(messages):
    out = []
    for m in messages:
        if m["role"] == "system":
            # 80/20 convention retired (spec): views carry no system messages;
            # the runtime template supplies the system prompt uniformly.
            raise MarkerMismatch(
                "system-role message in source row; views must not carry system prompts"
            )
        c = m["content"]
        if MARKER in c.replace(INPUT_MARKER, ""):
            raise MarkerMismatch(f"accidental {MARKER} in source text")
        out.append({"role": m["role"], "content": c.replace(INPUT_MARKER, MARKER)})
    return out


def _images_of(row) -> list[dict]:
    img = row["image"]
    return img if isinstance(img, list) else [img]


def _decode_dims(raw: bytes) -> tuple[int, int]:
    """(width, height) of the encoded image, or (0, 0) for undecodable bytes.

    Never raises: corrupt bytes must fail at scan as a skip, not a crash; the
    (0, 0) sentinel is caught by the sub-``SPATIAL_FACTOR`` gate in
    ``ingest_parquet``.
    """
    try:
        with Image.open(io.BytesIO(raw)) as im:
            return im.size
    except Exception:
        return (0, 0)


def _register_media(row: dict, seen: dict) -> list[str]:
    """Register each of the row's images into *seen* (``media_id ->
    UniqueMedia``, the cross-row dedup index) keyed by ``sha256(raw bytes)``
    full hex, decoding width/height once per unique media (the scan pass).
    Returns the row's refs in order. Shared across ROW_ADAPTERS.
    """
    refs = []
    for img in _images_of(row):
        mid = hashlib.sha256(img["bytes"]).hexdigest()
        if mid not in seen:
            ext = (img.get("path") or "bin").rsplit(".", 1)[-1]
            w, h = _decode_dims(img["bytes"])
            seen[mid] = UniqueMedia(mid, img["bytes"],
                                    source=str(row.get("source-id", "")),
                                    raw_ext=ext, width=w, height=h)
        refs.append(mid)
    return refs


def _refs_of(row: dict) -> list[str]:
    """All media refs of one view row (any ``*_media_refs`` field, by schema)."""
    return [m for k, v in row.items() if k.endswith("_media_refs") for m in v]


def _parse_preference_row(row: dict, seen: dict) -> dict:
    """Parse one mllm-dpo-shaped row into a preference view row."""
    prompt = _normalize(row["prompt"])
    accepted = _normalize(row["accepted"])
    rejected = _normalize(row["rejected"])
    refs = _register_media(row, seen)
    n_markers = sum(m["content"].count(MARKER) for m in prompt)
    if n_markers != len(refs):
        raise MarkerMismatch(
            f"{row.get('source-id')}: {n_markers} markers vs {len(refs)} images")
    for fieldname, msgs in (("chosen", accepted), ("rejected", rejected)):
        if any(MARKER in m["content"] for m in msgs):
            raise MarkerMismatch(f"accidental marker in {fieldname}")
    return {
        "prompt": prompt,
        "chosen": accepted[-1]["content"],
        "rejected": rejected[-1]["content"],
        "prompt_media_refs": refs,
        "chosen_media_refs": [], "rejected_media_refs": [],
        "prompt_id": str(row.get("source-id", "")),
    }


ROW_ADAPTERS = {"preference": _parse_preference_row}

# The output namespace comes from the TASK, not the mode (user directive):
# preference data lands under alignment/, RL data under rl/. Consumed by
# run_distributed_pipeline when namespacing the dataset root.
TASK_OUTPUT_DIRS = {"preference": "alignment", "rl_prompt": "rl"}


def ingest_parquet(path: Path, task: str) -> IngestResult:
    """Ingest one source parquet into unique media + drafted view rows.

    The scan gate runs here: media whose decoded ``min(width, height)`` falls
    below ``SPATIAL_FACTOR`` (corrupt bytes decode to (0, 0)) are skipped and
    counted, and every view row referencing them is dropped with its pair.
    """
    try:
        parse_row = ROW_ADAPTERS[task]
    except KeyError:
        raise ValueError(
            f"unknown task {task!r}; registered tasks: {sorted(ROW_ADAPTERS)}"
        ) from None
    table = pq.read_table(path)
    res = IngestResult()
    seen: dict[str, UniqueMedia] = {}
    rows = [parse_row(row, seen) for row in table.to_pylist()]
    skipped = {m.media_id for m in seen.values()
               if m.width < SPATIAL_FACTOR or m.height < SPATIAL_FACTOR}
    res.unique_media = [m for m in seen.values() if m.media_id not in skipped]
    res.view_rows = [r for r in rows if not skipped.intersection(_refs_of(r))]
    res.n_skipped_media = len(skipped)
    return res


def write_scan_parquet(path: Path, unique_media: list) -> int:
    """Persist the scan artifact: one geometry row per kept unique media,
    written atomically (tmp + ``os.replace``) BEFORE any GPU work. Returns the
    byte size for the dataset manifest's ``files`` map.
    """
    table = pa.Table.from_pylist(
        [{"media_id": m.media_id, "width": m.width, "height": m.height,
          "raw_length_bytes": len(m.raw), "source": m.source}
         for m in unique_media],
        schema=SCAN_SCHEMA)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    return path.stat().st_size
