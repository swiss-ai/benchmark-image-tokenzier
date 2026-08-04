"""Alignment mode: scan + encode phases around the unified executor.

The mode's CLI name is ``alignment``; internals keep the ``alignment`` naming
(matching the capstor dataset layout). No orchestration loop lives here —
``run_executor`` owns plan/prefetch/encode/checkpoint. This module owns the two
phases unique to the mode; the inline merge that assembles the store lives in
``pipeline.output.alignment_merge`` (``publish_alignment_store``):

  scan (inline, CPU, torch-free): shared parquet media scan + exact dedup
      persists ``scan.parquet`` / ``media_unique.parquet`` / ``views.raw.parquet``
      + ``publish_meta.json`` for the encode and the merge to read;
  encode (multi-rank GPU): read the pre-built scan, spill each rank's disjoint
      media slice via ``SpillBackend`` (the merge reads the spills directly).

The mode is task-neutral; ``cfg["task"]`` selects the post-dedup view builder.
Nothing below branches on task otherwise.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

from vision_tokenization.indexing.alignment.ingest import (
    SPATIAL_FACTOR,
    IngestResult,
    build_alignment_views_from_row_refs,
)
from vision_tokenization.indexing.scanners.parquet_media_scan import (
    dedup_media_scan,
    load_media_inventory,
    scan_parquet_media_refs_many,
)
from vision_tokenization.utils.json import json_dump

logger = logging.getLogger(__name__)


def _alignment_input_paths(cfg: dict) -> list[Path]:
    if cfg.get("input_pattern"):
        paths = [Path(path) for path in sorted(glob.glob(str(cfg["input_pattern"])))]
        if not paths:
            raise FileNotFoundError(f"input_pattern matched no parquet files: {cfg['input_pattern']}")
        return paths
    return [Path(cfg["input_parquet"])]


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
        materialize_raw=bool(cfg.get("materialize_raw_store", False)),
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
    cfg["output_dir"] = str(work)
    return out


def _unstage_work(public_dir: Path, cfg: dict) -> None:
    """Remove the ``<out>/_work`` staging dir unless intermediates are kept.
    Teardown half of the staging discipline; pairs with ``_stage_scan_into_work``."""
    if not cfg.get("keep_alignment_intermediates", False):
        shutil.rmtree(Path(public_dir) / "_work", ignore_errors=True)


def run_alignment_scan(cfg: dict) -> dict:
    """Alignment SCAN phase (inline, CPU, torch-free): ingest the preference
    parquet(s), dedup media globally, build the raw views, and persist the
    artifacts the GPU encode + inline merge consume — ``scan.parquet``,
    ``media_unique.parquet``, ``views.raw.parquet`` — plus ``publish_meta.json``,
    the config-derived fields the merge stamps into ``manifest.json``. Mirrors
    the sft/interleave contract (a pre-built scan the GPU job reads); the scan is
    an explicit inline step here because global content-dedup must precede encode."""
    out = Path(cfg["output_dir"])
    res = run_scan_stage(cfg)
    json_dump({
        "task": cfg["task"],
        "tokenizer_path": cfg["tokenizer_path"],
        "tokenizer_min_pixels": cfg["tokenizer_min_pixels"],
        "tokenizer_max_pixels": cfg["tokenizer_max_pixels"],
        "val_rows": int(cfg.get("val_rows", 0)),
        "source_input": (str(cfg["input_pattern"]) if cfg.get("input_pattern")
                         else str(cfg["input_parquet"])),
        "n_skipped_media": res.n_skipped_media,
    }, out / "publish_meta.json")
    logger.info("alignment scan: %d pairs, %d unique media (%d skipped) -> %s",
                len(res.view_rows), len(res.unique_media), res.n_skipped_media, out)
    return {"output_dir": str(out), "n_pairs": len(res.view_rows),
            "n_unique_media": len(res.unique_media), "n_skipped_media": res.n_skipped_media}


def run_alignment(cfg: dict) -> dict:
    """Alignment ENCODE phase (multi-rank GPU): read the pre-built scan and spill
    this rank's disjoint slice of media-token blocks via the shared SpillBackend.
    The scan runs inline beforehand (``run_alignment_scan``);
    ``publish_alignment_store`` assembles the store afterwards. Seal-at-end, no
    resume — each unique media is encoded by exactly one rank, so a skipped batch
    is unrecoverable."""
    if cfg["resume"]:
        raise ValueError(
            "alignment is seal-at-end (spill + views + manifest); "
            "resume is unsupported — re-run from scratch"
        )
    if int(cfg.get("max_consecutive_errors", 50)) > 1:
        raise ValueError(
            "alignment is seal-at-end with no resume; a skipped batch is "
            "unrecoverable, so max_consecutive_errors must be <= 1"
        )
    out = Path(cfg["output_dir"])
    scan_path = out / "scan.parquet"
    inventory_path = out / "media_unique.parquet"
    if not (scan_path.exists() and inventory_path.exists()):
        raise FileNotFoundError(
            f"alignment encode: pre-built scan missing in {out} "
            f"(run the scan phase on the head node first)"
        )
    cfg["manifest_path"] = str(scan_path)
    cfg["media_inventory"] = load_media_inventory(inventory_path)
    spill_dir = out / "_spill"
    cfg["output_dir"] = str(spill_dir)

    # Tokenize this rank's pair text in a SEPARATE process, concurrent with the GPU
    # vision encode below. A distinct interpreter avoids a transformers lazy-import
    # race with the executor's tokenizer load; the merge joins the spilled pieces.
    text_proc = subprocess.Popen([
        sys.executable, "-m", "vision_tokenization.pipeline.runtime.alignment_text",
        str(out / "views.raw.parquet"), str(spill_dir), str(cfg["rank"]), str(cfg["world_size"]),
        "--tokenizer-path", str(cfg["tokenizer_path"]),
        "--system", str(cfg.get("binidx_system", "empty")),
        "--task", str(cfg["task"]),
    ])
    try:
        from .executor import run_executor
        result = run_executor(cfg["rank"], cfg["world_size"], cfg)
    except BaseException:
        text_proc.kill()
        text_proc.wait()
        raise
    if text_proc.wait() != 0:
        raise RuntimeError(f"alignment text pass failed (rank {cfg['rank']}, rc={text_proc.returncode})")
    return result


def run_dpo_binidx(cfg: dict) -> dict:
    """Alignment binidx phase (CPU): build the ``.bin/.idx`` + ``index`` from the published store
    (preference: ``[prompt|chosen|rejected]`` per pair; rl_prompt: ``[prompt]`` per prompt),
    register them in ``manifest.json`` (schema 4), and retire the now-redundant deduped ``tokens/``."""
    out = Path(cfg["output_dir"])
    if not (out / "manifest.json").exists():
        raise FileNotFoundError(
            f"dpo binidx: manifest missing in {out} (run scan -> encode -> merge first)"
        )
    from vision_tokenization.pipeline.output.alignment_merge import stamp_dpo_section
    from vision_tokenization.pipeline.output.dpo_binidx import build_dpo_binidx
    section = build_dpo_binidx(str(out), task=cfg["task"], system=cfg.get("binidx_system", "empty"))
    delete = cfg.get("binidx_delete_tokens", False)
    stamp_dpo_section(out, section, delete_tokens=delete)
    count_key, noun = ("n_pairs", "pairs") if cfg["task"] == "preference" else ("n_samples", "prompts")
    logger.info("alignment binidx: %d %s, schema 4 stamped, tokens/ %s -> %s",
                sum(s[count_key] for s in section["splits"].values()), noun,
                "retired" if delete else "kept", out)
    return section
