"""Alignment-mode view building from dedup-filtered media row refs.

The input marker is the dataset-level ``<image>``; the canonical on-disk marker
is the tokenizer special ``<|image|>`` — the manifest's ``token_layout`` records
its id so the consumer can count it in token space. Per-field marker counts
must equal per-field ref counts (spec contract).

The scan/dedup path owns media facts and row-media references. This module owns
only the task-specific text conversion for final views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq

MARKER = "<|image|>"
INPUT_MARKER = "<image>"
SPATIAL_FACTOR = 16
DEFAULT_INGEST_BATCH_SIZE = 1024


class MarkerMismatch(ValueError):
    pass


@dataclass
class IngestResult:
    unique_media: list = field(default_factory=list)
    view_rows: list = field(default_factory=list)
    n_skipped_media: int = 0


def _task_payload_columns(task: str, schema: pa.Schema) -> list[str]:
    if task == "preference":
        required = ["prompt", "accepted", "rejected"]
        optional = ["source-id"]
    else:
        required = []
        optional = ["source-id"]

    names = set(schema.names)
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"missing required payload column(s) for {task!r}: {missing}")
    return [name for name in [*required, *optional] if name in names]


def build_alignment_views_from_row_refs(
    source_path: Path | Sequence[Path],
    row_refs_path: Path,
    output_path: Path,
    *,
    task: str,
    batch_size: int = DEFAULT_INGEST_BATCH_SIZE,
) -> int:
    """Build alignment view rows from dedup-filtered row/media references.

    This is intentionally after scan+dedup: the scanner owns media facts and
    row refs only, while this function owns task text conversion.
    """
    if task != "preference":
        raise ValueError("only preference view building is implemented")

    source_paths = (
        [Path(source_path)]
        if isinstance(source_path, (str, Path))
        else [Path(path) for path in source_path]
    )
    row_refs_path = Path(row_refs_path)
    output_path = Path(output_path)
    refs_by_source: dict[str, dict[int, dict[int, list[str]]]] = {}
    for ref_row in pq.read_table(row_refs_path).to_pylist():
        source_key = str(Path(ref_row["source_path"]))
        refs_by_group = refs_by_source.setdefault(source_key, {})
        group_refs = refs_by_group.setdefault(int(ref_row["row_group"]), {})
        group_refs[int(ref_row["row_index"])] = list(ref_row["media_refs"])

    rows: list[dict] = []
    seen_sources = set()
    for path in source_paths:
        source_key = str(path)
        seen_sources.add(source_key)
        refs_by_group = refs_by_source.get(source_key)
        if not refs_by_group:
            continue
        parquet_file = pq.ParquetFile(path)
        columns = _task_payload_columns(task, parquet_file.schema_arrow)
        for row_group in sorted(refs_by_group):
            row_index = 0
            for batch in parquet_file.iter_batches(
                row_groups=[row_group],
                columns=columns,
                batch_size=batch_size,
            ):
                col_by_name = {
                    name: batch.column(i)
                    for i, name in enumerate(batch.schema.names)
                }
                for local_idx in range(batch.num_rows):
                    refs = refs_by_group[row_group].get(row_index)
                    if refs is not None:
                        row = {
                            name: col[local_idx].as_py()
                            for name, col in col_by_name.items()
                        }
                        rows.append(_parse_preference_row_with_refs(row, refs))
                    row_index += 1

    unknown_sources = sorted(set(refs_by_source) - seen_sources)
    if unknown_sources:
        raise ValueError(
            f"row refs contain source path(s) not present in source_path: {unknown_sources[:5]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(rows), tmp)
    tmp.replace(output_path)
    return len(rows)


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


def _parse_preference_row_with_refs(row: dict, refs: list[str]) -> dict:
    """Parse one mllm-dpo-shaped row using already-computed media refs."""
    prompt = _normalize(row["prompt"])
    accepted = _normalize(row["accepted"])
    rejected = _normalize(row["rejected"])
    if not accepted or not rejected:
        raise MarkerMismatch(
            f"{row.get('source-id')}: empty accepted/rejected message list")
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
