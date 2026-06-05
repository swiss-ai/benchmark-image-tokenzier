#!/usr/bin/env python3
"""Rebuild a spill-backend rank's bin/idx with contaminated docs excluded.

Reads the existing spill at ``--spill-dir/rank_NNNN/`` (preserved from the
original tokenize run), derives the set of contaminated doc_ids from
``(manifest, contamination_ids_file)``, and runs ``rebuild_rank`` writing the
new bin/idx under ``--output-dir`` (defaults to a sibling decontaminated dir).

Designed for slurm array parallelism: one task per rank.

Usage::

    python scripts/rebuild_decontaminated.py \\
        --manifest /capstor/scratch/.../manifest.parquet \\
        --contamination-ids /iopsstor/.../innovator_vl_contaminated_ids.txt \\
        --spill-dir /capstor/store/.../tokenized/sft/innovator_vl_46m \\
        --output-dir /capstor/store/.../tokenized/sft/innovator_vl_46m_decontaminated \\
        --tokenizer-path /capstor/store/.../tokenizer/apertus_emu3.5_wavtok_instruct \\
        --rank 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow.parquet as pq

from vision_tokenization.common.assembly import StructureTokenIds
from vision_tokenization.discrete.emu import create_tokenizer
from vision_tokenization.indexing.planning.tokenization_plan import build_tokenization_plan
from vision_tokenization.pipeline.output.rebuild import rebuild_rank
from vision_tokenization.utils.contamination import chunk_offsets, load_contamination_index

logger = logging.getLogger(__name__)


def compute_rejected_doc_ids(
    manifest_path: Path,
    contamination_ids_path: Path,
    contamination_format: str,
) -> set[int]:
    """Map contamination IDs -> set of plan doc_ids to skip during rebuild.

    Pipeline: contamination ID -> (shard_stem, source_row_in_shard) ->
    manifest row whose ``chunk_offsets[chunk_index] + row_in_chunk == source_row``
    -> manifest ``group_id`` -> plan doc_id (position in ``np.unique(group_ids)``).
    """
    index = load_contamination_index(contamination_ids_path, format=contamination_format)
    logger.info(
        "Loaded %d contamination IDs across %d source shards",
        index.total_ids,
        len(index.by_source),
    )

    tbl = pq.read_table(
        manifest_path,
        columns=["shard_path", "chunk_index", "row_in_chunk", "group_id"],
    )
    shard_paths = tbl.column("shard_path").to_pylist()
    chunk_indices = tbl.column("chunk_index").combine_chunks().to_numpy(zero_copy_only=False)
    rows_in_chunk = tbl.column("row_in_chunk").combine_chunks().to_numpy(zero_copy_only=False)
    group_ids = tbl.column("group_id").combine_chunks().to_numpy(zero_copy_only=False)

    path_cache: dict[str, tuple] = {}
    contaminated_gids: set[int] = set()
    for row_idx, shard_path in enumerate(shard_paths):
        cached = path_cache.get(shard_path)
        if cached is None:
            rows = index.rows_for_path(shard_path)
            offsets = chunk_offsets(shard_path) if rows else None
            cached = (rows, offsets)
            path_cache[shard_path] = cached
        rows, offsets = cached
        if not rows or offsets is None:
            continue
        chunk_idx = int(chunk_indices[row_idx])
        source_row = int(offsets[chunk_idx]) + int(rows_in_chunk[row_idx])
        if source_row in rows:
            contaminated_gids.add(int(group_ids[row_idx]))

    logger.info(
        "Found %d contaminated group_ids in manifest (%d manifest rows mention them)",
        len(contaminated_gids),
        sum(1 for g in group_ids if int(g) in contaminated_gids),
    )

    # Plan doc_id N corresponds to the Nth-smallest unique group_id
    # (planner uses ``np.unique`` over valid group_ids).
    unique_gids = np.unique(group_ids)
    if not contaminated_gids:
        return set()
    contam_arr = np.array(sorted(contaminated_gids), dtype=unique_gids.dtype)
    positions = np.searchsorted(unique_gids, contam_arr)
    # Sanity-check: every contaminated gid must resolve to a unique gid in the manifest
    valid = (positions < len(unique_gids)) & (unique_gids[np.minimum(positions, len(unique_gids) - 1)] == contam_arr)
    if not valid.all():
        missing = contam_arr[~valid]
        logger.warning(
            "%d contaminated group_ids did not match any unique gid in manifest "
            "(first few: %s)",
            (~valid).sum(),
            missing[:5].tolist(),
        )
    return set(int(p) for p in positions[valid])


def _build_plan(args: argparse.Namespace):
    return build_tokenization_plan(
        manifest_path=args.manifest,
        mode=args.mode,
        text_column=args.text_column,
        parser=None,
        min_pixels=None,
        max_pixels=None,
        max_images_per_doc=None,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        spatial_factor=args.spatial_factor,
        resize_min_pixels=args.tokenizer_min_pixels,
        resize_max_pixels=args.tokenizer_max_pixels,
        window_size=args.window_size,
    )


def _build_token_ids(args: argparse.Namespace) -> tuple[StructureTokenIds, int]:
    tokenizer = create_tokenizer(
        mode=args.mode,
        text_tokenizer_path=args.tokenizer_path,
        device=args.device,
        min_pixels=args.tokenizer_min_pixels,
        max_pixels=args.tokenizer_max_pixels,
    )
    token_ids = StructureTokenIds(
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        img_start_id=tokenizer.img_start_id,
        img_end_id=tokenizer.img_end_id,
        img_token_start_id=tokenizer.img_token_start_id,
        eol_id=tokenizer.eol_id,
        eof_id=tokenizer.eof_id,
        vision_token_offset=tokenizer.vision_token_offset,
        image_token_id=getattr(tokenizer, "image_token_id", -1),
        dim_tokens_fn=tokenizer._get_dim_tokens,
    )
    return token_ids, len(tokenizer.text_tokenizer)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--contamination-ids", required=True, type=Path)
    parser.add_argument("--contamination-format", default="innovator_vl")
    parser.add_argument("--spill-dir", type=Path, default=None, help="Required unless --prepare-only.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=None, help="Required unless --prepare-only.")
    parser.add_argument("--tokenizer-path", required=True, type=str)
    parser.add_argument("--mode", default="sft")
    parser.add_argument("--text-column", default="conversations")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--max-sequence-tokens", type=int, default=None)
    parser.add_argument("--seqlen-threshold", type=int, default=None)
    parser.add_argument("--spatial-factor", type=int, default=16)
    parser.add_argument("--tokenizer-min-pixels", type=int, default=16384)
    parser.add_argument("--tokenizer-max-pixels", type=int, default=1960000)
    parser.add_argument("--window-size", type=int, default=2000)
    parser.add_argument(
        "--plan-pt",
        type=Path,
        default=None,
        help="Path for cached TokenizationPlan. Defaults to <output-dir>/plan.pt — "
        "auto-loaded if present, auto-saved on first build.",
    )
    parser.add_argument(
        "--reject-doc-ids-json",
        type=Path,
        default=None,
        help="Path for cached reject_doc_ids list. Defaults to <output-dir>/reject_doc_ids.json — "
        "auto-loaded if present, auto-saved on first compute.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Compute plan and reject_doc_ids, save to --plan-pt / --reject-doc-ids-json, then exit. "
        "Use on a head node before launching the slurm array.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Cache paths default under output_dir. Prepare mode writes them; rebuild
    # mode requires both to exist (run --prepare-only first if missing).
    if args.plan_pt is None:
        args.plan_pt = args.output_dir / "plan.pt"
    if args.reject_doc_ids_json is None:
        args.reject_doc_ids_json = args.output_dir / "reject_doc_ids.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import json
    import torch

    if args.prepare_only:
        logger.info("Computing rejected doc_ids …")
        reject_doc_ids = compute_rejected_doc_ids(
            args.manifest,
            args.contamination_ids,
            args.contamination_format,
        )
        args.reject_doc_ids_json.parent.mkdir(parents=True, exist_ok=True)
        args.reject_doc_ids_json.write_text(json.dumps(sorted(reject_doc_ids)))
        logger.info("Saved reject_doc_ids to %s (%d entries)", args.reject_doc_ids_json, len(reject_doc_ids))

        logger.info("Building TokenizationPlan from %s …", args.manifest)
        plan = _build_plan(args)
        args.plan_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(plan, args.plan_pt)
        logger.info("Saved plan to %s", args.plan_pt)
        logger.info(
            "Plan: %d documents, %d components, %d batches",
            plan.total_documents,
            len(plan.components.document_id),
            len(plan.execution.image_batches),
        )
        return 0

    if not args.plan_pt.exists():
        parser.error(
            f"plan.pt not found at {args.plan_pt}. "
            "Run with --prepare-only first to produce it."
        )
    if not args.reject_doc_ids_json.exists():
        parser.error(
            f"reject_doc_ids.json not found at {args.reject_doc_ids_json}. "
            "Run with --prepare-only first to produce it."
        )

    logger.info("Loading reject_doc_ids from %s", args.reject_doc_ids_json)
    reject_doc_ids = set(json.loads(args.reject_doc_ids_json.read_text()))
    logger.info("Rejecting %d doc_ids during rebuild", len(reject_doc_ids))

    logger.info("Loading TokenizationPlan from %s", args.plan_pt)
    plan = torch.load(args.plan_pt, map_location="cpu", weights_only=False)
    logger.info(
        "Plan: %d documents, %d components, %d batches",
        plan.total_documents,
        len(plan.components.document_id),
        len(plan.execution.image_batches),
    )

    if args.rank is None or args.spill_dir is None:
        parser.error("--rank and --spill-dir are required unless --prepare-only is set.")

    logger.info("Loading tokenizer to derive StructureTokenIds …")
    token_ids, vocab_size = _build_token_ids(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Rebuilding rank %04d  spill=%s  output=%s",
        args.rank,
        args.spill_dir,
        args.output_dir,
    )
    stats = rebuild_rank(
        plan=plan,
        rank=args.rank,
        spill_dir=args.spill_dir,
        token_ids=token_ids,
        vocab_size=vocab_size,
        output_dir=args.output_dir,
        max_sequence_tokens=args.max_sequence_tokens,
        seqlen_threshold=args.seqlen_threshold,
        reject_doc_ids=reject_doc_ids,
    )
    logger.info("Rebuild stats for rank %04d: %s", args.rank, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
