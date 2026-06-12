"""Posttraining mode: scan + publish stages around the unified executor.

The mode's CLI name is ``posttraining``; internals keep the ``alignment``
naming (matching the capstor dataset layout). No orchestration loop lives
here — ``run_executor`` owns plan/prefetch/encode/checkpoint. This module
owns only the stages unique to the mode:

  scan (rank 0, CPU): ``ingest_parquet`` dedups/validates and persists
      ``scan.parquet`` — the geometry artifact, and this mode's manifest;
  executor: plan builder (``build_plan_posttraining``), parquet-bytes loader
      (``AlignmentMediaLoader``), and ``MediaStoreBackend`` plug in via cfg;
  publish (rank 0): ``views/{train,validation}.parquet``, then
      ``manifest.json`` LAST (the commit record).

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
    ingest_parquet,
    write_scan_parquet,
)
from vision_tokenization.pipeline.output.media_store import atomic_write_json
from vision_tokenization.utils.json import json_load

from .executor import run_executor

logger = logging.getLogger(__name__)


def _token_layout(tokenizer_path: str) -> tuple[dict, dict]:
    """Manifest ``token_layout`` (plus the loaded tokenizer_config).

    Every id is derived from the tokenizer snapshot — consumers read ids from
    the manifest, never from literals.
    """
    from transformers import AutoTokenizer

    from vision_tokenization.discrete.emu.image_only import (
        STRUCTURE_TOKENS,
        resolve_token_ids,
        vision_band,
    )

    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=True)
    ids = resolve_token_ids(
        text_tokenizer, {"image_marker": MARKER, **STRUCTURE_TOKENS})
    tokenizer_config = json_load(Path(tokenizer_path) / "tokenizer_config.json")
    vision_lo, vision_hi = vision_band(tokenizer_config)
    layout = {"image_marker": MARKER, "image_marker_id": ids.pop("image_marker"),
              **ids, "vision_lo": vision_lo, "vision_hi": vision_hi}
    return layout, tokenizer_config


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
    # Publish stage: views with exact media token stats, manifest LAST
    # ------------------------------------------------------------------
    length_of = {r["media_id"]: r["length_elems"]
                 for f in sorted((out / "media").glob("media.*.parquet"))
                 for r in pq.read_table(f).to_pylist()}
    rows = res.view_rows
    for row in rows:
        row["media_tokens_total"] = sum(length_of[m] for m in row["prompt_media_refs"])
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

    token_layout, tokenizer_config = _token_layout(cfg["tokenizer_path"])
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
