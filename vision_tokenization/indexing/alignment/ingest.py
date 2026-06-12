"""Alignment-mode parquet ingest: hash+dedup images, validate markers, draft views.

The input marker is the dataset-level ``<image>``; the canonical on-disk marker
is the tokenizer special ``<|image|>`` (id 131079) so the consumer can count it
in token space. Per-field marker counts must equal per-field ref counts (spec
contract).

``ROW_ADAPTERS`` keys the per-task row parser + view schema (``task:`` in the
dataset yaml): ``preference`` today, ``rl_prompt`` reserved for P4. The media
store, planner, runner, and Gate 2 never branch on task.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

MARKER = "<|image|>"
INPUT_MARKER = "<image>"


class MarkerMismatch(ValueError):
    pass


@dataclass
class UniqueMedia:
    media_id: str
    raw: bytes
    source: str
    raw_ext: str


@dataclass
class IngestResult:
    unique_media: list = field(default_factory=list)
    view_rows: list = field(default_factory=list)


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


def _parse_preference_row(row: dict, seen: dict) -> dict:
    """Parse one mllm-dpo-shaped row into a preference view row.

    Registers each image into *seen* (``media_id -> UniqueMedia``, the
    cross-row dedup index) keyed by ``sha256(raw bytes)`` full hex.
    """
    prompt = _normalize(row["prompt"])
    accepted = _normalize(row["accepted"])
    rejected = _normalize(row["rejected"])
    refs = []
    for img in _images_of(row):
        mid = hashlib.sha256(img["bytes"]).hexdigest()
        if mid not in seen:
            ext = (img.get("path") or "bin").rsplit(".", 1)[-1]
            seen[mid] = UniqueMedia(mid, img["bytes"],
                                    source=str(row.get("source-id", "")),
                                    raw_ext=ext)
        refs.append(mid)
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


def ingest_parquet(path: Path, task: str) -> IngestResult:
    """Ingest one source parquet into unique media + drafted view rows."""
    try:
        parse_row = ROW_ADAPTERS[task]
    except KeyError:
        raise ValueError(
            f"unknown task {task!r}; registered tasks: {sorted(ROW_ADAPTERS)}"
        ) from None
    table = pq.read_table(path)
    res = IngestResult()
    seen: dict[str, UniqueMedia] = {}
    for row in table.to_pylist():
        res.view_rows.append(parse_row(row, seen))
    res.unique_media = list(seen.values())
    return res
