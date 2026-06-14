"""Alignment mode: scan + publish stages around the unified executor.

The mode's CLI name is ``alignment``; internals keep the ``alignment``
naming (matching the capstor dataset layout). No orchestration loop lives
here — ``run_executor`` owns plan/prefetch/encode/checkpoint. This module
owns only the stages unique to the mode:

  scan (rank 0, CPU): shared parquet media scan + exact dedup persists
      ``scan.parquet`` as an internal geometry artifact for planning;
  executor: plan builder (``build_plan_alignment``), parquet-bytes loader
      (``AlignmentMediaLoader``), and ``AlignmentPayloadBackend`` plug in via
      cfg and write the final shard-local ``views/`` + ``tokens/`` + ``raw/``
      payload artifacts directly;
  publish (rank 0): write ``manifest.json`` LAST (the commit record).

The mode is task-neutral; ``cfg["task"]`` selects the post-dedup view builder.
Nothing below branches on task otherwise.
"""

from __future__ import annotations

import hashlib
import glob
import json
import logging
import os
import shutil
import time
from pathlib import Path

import pyarrow.parquet as pq

from vision_tokenization.indexing.alignment.ingest import (
    MARKER,
    SPATIAL_FACTOR,
    IngestResult,
    build_alignment_views_from_row_refs,
)
from vision_tokenization.indexing.scanners.parquet_media_scan import (
    dedup_media_scan,
    load_media_inventory,
    scan_parquet_media_refs_many,
)
from vision_tokenization.utils.json import json_load

logger = logging.getLogger(__name__)


def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _alignment_input_paths(cfg: dict) -> list[Path]:
    if cfg.get("input_pattern"):
        paths = [Path(path) for path in sorted(glob.glob(str(cfg["input_pattern"])))]
        if not paths:
            raise FileNotFoundError(f"input_pattern matched no parquet files: {cfg['input_pattern']}")
        return paths
    return [Path(cfg["input_parquet"])]


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


def run_scan_stage(cfg: dict) -> IngestResult:
    """Scan stage (rank 0, CPU): ingest the input parquet, persist
    ``scan.parquet``, and inject ``manifest_path`` + ``media_inventory``
    into *cfg*. The single scan writer — the GPU run and the CPU dry run
    both go through here."""
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    build_dir = out / "_scan_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    input_paths = _alignment_input_paths(cfg)
    scan = scan_parquet_media_refs_many(
        input_paths,
        build_dir,
        workers=int(cfg.get("scan_workers") or min(64, os.cpu_count() or 1)),
        batch_size=int(cfg.get("scan_batch_size", 1024)),
    )
    dedup = dedup_media_scan(
        build_dir,
        out,
        min_side=int(cfg.get("spatial_factor", SPATIAL_FACTOR)),
    )
    if dedup.n_invalid_media:
        logger.warning("scan skipped %d corrupt/sub-%dpx images (and their pairs)",
                       dedup.n_invalid_media, SPATIAL_FACTOR)

    view_tmp = out / "views.raw.parquet"
    n_views = build_alignment_views_from_row_refs(
        input_paths,
        Path(dedup.row_refs_path),
        view_tmp,
        task=cfg["task"],
        batch_size=int(cfg.get("scan_batch_size", 1024)),
    )
    rows = pq.read_table(view_tmp).to_pylist()
    if n_views != dedup.n_valid_rows:
        raise RuntimeError(
            f"view build produced {n_views} rows from {dedup.n_valid_rows} valid row refs"
        )

    unique_media = load_media_inventory(Path(dedup.media_unique_path))
    # The plan indexes the inventory by position (source_ref = arange(N)), so
    # scan.parquet and the inventory must share row order. Assert it before the
    # plan is built.
    scan_media_ids = (
        pq.read_table(dedup.scan_path, columns=["media_id"])
        .column("media_id").to_pylist()
    )
    if scan_media_ids != [m.media_id for m in unique_media]:
        raise RuntimeError(
            "alignment order contract violated: scan.parquet row order does not "
            "match the media inventory order (media_unique.parquet)"
        )
    res = IngestResult(
        unique_media=unique_media,
        view_rows=rows,
        n_skipped_media=dedup.n_invalid_media,
    )
    # scan.parquet is this mode's internal geometry artifact (plan fingerprint
    # source); the inventory is its in-memory companion — same row order
    # (asserted above), plus raw bytes.
    cfg["manifest_path"] = dedup.scan_path
    cfg["media_inventory"] = res.unique_media
    logger.info(
        "scan/dedup complete: %d source rows, %d row refs, %d unique media, "
        "%d valid rows, %d filtered rows",
        scan.n_source_rows, scan.n_row_refs, len(unique_media),
        dedup.n_valid_rows, dedup.n_filtered_rows,
    )
    return res


def _stage_scan_into_work(cfg: dict) -> Path:
    """Redirect the scan stage's writes into ``<out>/_work`` so the public root
    stays clean — only the published payload + ``manifest.json`` (real run) or
    ``dry_run_stats.json`` (dry run) are left at the top level. Returns the
    public output dir. The single definition of the scan-staging discipline,
    shared by the real run and the dry run."""
    out = Path(cfg["output_dir"])
    work = out / "_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    cfg["alignment_public_dir"] = str(out)
    cfg["output_dir"] = str(work)
    return out


def _unstage_work(public_dir: Path, cfg: dict) -> None:
    """Remove the ``<out>/_work`` staging dir unless intermediates are kept.
    Teardown half of the staging discipline; pairs with ``_stage_scan_into_work``."""
    if not cfg.get("keep_alignment_intermediates", False):
        shutil.rmtree(Path(public_dir) / "_work", ignore_errors=True)


def run_alignment(cfg: dict) -> dict:
    if cfg["resume"]:
        raise ValueError(
            "alignment is seal-at-end (media store + views + manifest); "
            "resume is unsupported — re-run from scratch"
        )
    if int(cfg.get("max_consecutive_errors", 50)) > 1:
        raise ValueError(
            "alignment is seal-at-end with no resume; a skipped batch is "
            "unrecoverable, so max_consecutive_errors must be <= 1"
        )
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Scan stage (CPU): this mode's manifest scan, staged into <out>/_work
    # ------------------------------------------------------------------
    out = _stage_scan_into_work(cfg)
    res = run_scan_stage(cfg)
    cfg["alignment_view_rows"] = res.view_rows

    from .executor import run_executor
    result = run_executor(cfg["rank"], cfg["world_size"], cfg)
    payload = result.get("alignment_payload")
    if not payload:
        raise RuntimeError("alignment executor did not return payload metadata")
    payload_files = payload["files"]
    views = payload["views"]
    _unstage_work(out, cfg)

    tokenizer_config = json_load(Path(cfg["tokenizer_path"]) / "tokenizer_config.json")
    token_layout = _token_layout(tokenizer_config)
    tok_sha = hashlib.sha256(
        (Path(cfg["tokenizer_path"]) / "tokenizer.json").read_bytes()).hexdigest()
    atomic_write_json(out / "manifest.json", {
        "schema_version": 3,
        "payload_format": "alignment_shard_local_v1",
        "tokenizer": {"path": cfg["tokenizer_path"], "sha256": tok_sha},
        "vision_tokenizer": {"version": tokenizer_config["vision_tokenizer"]["type"],
                             "min_pixels": cfg["tokenizer_min_pixels"],
                             "max_pixels": cfg["tokenizer_max_pixels"]},
        "token_dtype": "<i4",
        "token_layout": token_layout,
        "expected_min_model_vocab": max(
            m["offset"] + m["vocab_size"]
            for m in tokenizer_config["omnimodal_config"]["modalities"]),
        "views": views,
        "default_train_view": "train" if "train" in views else None,
        "default_validation_view": "validation" if "validation" in views else None,
        "files": payload_files,
        "source_input": (
            str(cfg["input_pattern"])
            if cfg.get("input_pattern")
            else str(cfg["input_parquet"])
        ),
        "n_pairs": len(res.view_rows),
        "n_unique_media": len(res.unique_media),
        "n_skipped_media": res.n_skipped_media,
    })

    elapsed = time.perf_counter() - t_start
    logger.info(
        "alignment mode done: %d pairs, %d unique media (%d skipped) -> %s "
        "[%.1f s end-to-end, %.1f img/s]",
        len(res.view_rows), len(res.unique_media), res.n_skipped_media, out,
        elapsed, len(res.unique_media) / elapsed)
    return {**result,
            "output_dir": str(out),
            "n_pairs": len(res.view_rows),
            "n_unique_media": len(res.unique_media),
            "n_skipped_media": res.n_skipped_media}
