"""Posttraining mode: scan + publish stages around the unified executor.

The mode's CLI name is ``posttraining``; internals keep the ``alignment``
naming (matching the capstor dataset layout). No orchestration loop lives
here — ``run_executor`` owns plan/prefetch/encode/checkpoint. This module
owns only the stages unique to the mode:

  scan (rank 0, CPU): ``ingest_parquet`` dedups/validates and persists
      ``scan.parquet`` — the geometry artifact, and this mode's manifest;
  executor: plan builder (``build_plan_posttraining``), parquet-bytes loader
      (``AlignmentMediaLoader``), and ``MediaStoreBackend`` plug in via cfg;
  publish (rank 0): completeness gate (every scanned media in the sealed
      store), ``views/{train,validation}.parquet``, then ``manifest.json``
      LAST (the commit record).

The mode is task-neutral; ``cfg["task"]`` selects the row adapter + view
schema inside ``ingest_parquet`` (ROW_ADAPTERS) and the output namespace
(TASK_OUTPUT_DIRS). Nothing below branches on task.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from vision_tokenization.indexing.alignment.ingest import (
    MARKER,
    SPATIAL_FACTOR,
    _refs_of,
    ingest_parquet,
    write_scan_parquet,
)
from vision_tokenization.pipeline.output.media_store import atomic_write_json
from vision_tokenization.utils.json import json_load

from .executor import run_executor

logger = logging.getLogger(__name__)


class EncodeIncompleteError(RuntimeError):
    """Publish-gate contract: every scanned media must be in the sealed store."""


def check_encode_complete(unique_media: list, length_of: dict) -> None:
    """Raise :class:`EncodeIncompleteError` unless every scanned unique media
    has a block in the sealed store index (decode-failed media show up here)."""
    missing = [m.media_id for m in unique_media if m.media_id not in length_of]
    if missing:
        raise EncodeIncompleteError(
            f"encode incomplete: {len(missing)} of {len(unique_media)} scanned "
            f"media missing from store (first ids: {missing[:5]})"
        )


def _token_layout(tokenizer_config: dict) -> dict:
    """Manifest ``token_layout``, derived from tokenizer_config.json alone.

    Every id comes from the snapshot's ``added_tokens_decoder`` /
    ``omnimodal_config`` — consumers read ids from the manifest, never from
    literals, and publish never loads the tokenizer.
    """
    from vision_tokenization.discrete.emu.image_only import (
        STRUCTURE_TOKENS,
        resolve_token_ids_from_config,
        vision_band,
    )

    ids = resolve_token_ids_from_config(
        tokenizer_config, {"image_marker": MARKER, **STRUCTURE_TOKENS})
    vision_lo, vision_hi = vision_band(tokenizer_config)
    return {"image_marker": MARKER, "image_marker_id": ids.pop("image_marker"),
            **ids, "vision_lo": vision_lo, "vision_hi": vision_hi}


def run_posttraining(cfg: dict) -> dict:
    if cfg["resume"]:
        raise ValueError(
            "posttraining is seal-at-end (media store + views + manifest); "
            "resume is unsupported — re-run from scratch"
        )
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Scan stage (CPU): this mode's manifest scan
    # ------------------------------------------------------------------
    out = Path(cfg["output_dir"])
    res = ingest_parquet(Path(cfg["input_parquet"]), task=cfg["task"])
    if res.n_skipped_media:
        logger.warning("scan skipped %d corrupt/sub-%dpx images (and their pairs)",
                       res.n_skipped_media, SPATIAL_FACTOR)
    out.mkdir(parents=True, exist_ok=True)
    scan_size = write_scan_parquet(out / "scan.parquet", res.unique_media)

    # scan.parquet IS this mode's manifest (plan fingerprint source); the
    # inventory is its in-memory companion — same row order, plus raw bytes.
    cfg["manifest_path"] = str(out / "scan.parquet")
    cfg["media_inventory"] = res.unique_media

    result = run_executor(cfg["rank"], cfg["world_size"], cfg)

    # ------------------------------------------------------------------
    # Publish stage: completeness gate, views with exact media token stats,
    # manifest LAST
    # ------------------------------------------------------------------
    length_of = {r["media_id"]: r["length_elems"]
                 for f in sorted((out / "media").glob("media.*.parquet"))
                 for r in pq.read_table(f).to_pylist()}
    check_encode_complete(res.unique_media, length_of)
    rows = res.view_rows
    for row in rows:
        row["media_tokens_total"] = sum(length_of[m] for m in _refs_of(row))
        row["text_chars"] = (sum(len(m["content"]) for m in row["prompt"])
                             + len(row["chosen"]) + len(row["rejected"]))

    rng = random.Random(42)
    rng.shuffle(rows)
    n_val = min(cfg["val_rows"], max(1, len(rows) // 50))
    (out / "views").mkdir(parents=True, exist_ok=True)
    view_files = {}
    for name, part in (("validation", rows[:n_val]), ("train", rows[n_val:])):
        p = out / "views" / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(part), p)
        view_files[f"views/{name}.parquet"] = p.stat().st_size

    # Operational droppings (checkpoint/stats/DONE) leave the published root;
    # the manifest's files map never includes them.
    ops = out / "_pipeline"
    ops.mkdir(exist_ok=True)
    for p in [*out.glob("rank_*"), out / "stats_summary.json"]:
        if p.exists():
            p.rename(ops / p.name)

    tokenizer_config = json_load(Path(cfg["tokenizer_path"]) / "tokenizer_config.json")
    token_layout = _token_layout(tokenizer_config)
    tok_sha = hashlib.sha256(
        (Path(cfg["tokenizer_path"]) / "tokenizer.json").read_bytes()).hexdigest()
    media_files = {f"media/{p.name}": p.stat().st_size
                   for p in sorted((out / "media").iterdir())
                   if not p.name.endswith(".tmp")}
    atomic_write_json(out / "manifest.json", {
        "schema_version": 1,
        "tokenizer": {"path": cfg["tokenizer_path"], "sha256": tok_sha},
        "vision_tokenizer": {"version": tokenizer_config["vision_tokenizer"]["type"],
                             "min_pixels": cfg["tokenizer_min_pixels"],
                             "max_pixels": cfg["tokenizer_max_pixels"]},
        "token_dtype": "<i4",
        "token_layout": token_layout,
        "expected_min_model_vocab": max(
            m["offset"] + m["vocab_size"]
            for m in tokenizer_config["omnimodal_config"]["modalities"]),
        "media_roots": ["media/"],
        "store_raw": True,
        "files": {"scan.parquet": scan_size, **media_files, **view_files},
        "source_input": str(cfg["input_parquet"]),
        "n_pairs": len(rows),
        "n_unique_media": len(res.unique_media),
        "n_skipped_media": res.n_skipped_media,
    })

    elapsed = time.perf_counter() - t_start
    logger.info(
        "posttraining mode done: %d pairs, %d unique media (%d skipped) -> %s "
        "[%.1f s end-to-end, %.1f img/s]",
        len(rows), len(res.unique_media), res.n_skipped_media, out,
        elapsed, len(res.unique_media) / elapsed)
    return {**result,
            "n_pairs": len(rows),
            "n_unique_media": len(res.unique_media),
            "n_skipped_media": res.n_skipped_media}
