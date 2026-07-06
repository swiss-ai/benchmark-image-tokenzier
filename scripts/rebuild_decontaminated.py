#!/usr/bin/env python3
"""Rebuild a spill-backend dataset's bin/idx with contaminated docs excluded.

Two-step flow (no GPU, no re-encode — reuses the preserved per-rank spill):

1. PREPARE (head node, ``--prepare-only``): rebuilds the TokenizationPlan from
   the manifest, derives the contaminated doc_ids from
   ``(manifest, contamination_ids_file)`` — aborting via the document-count
   guardrail if the plan args do not reproduce the original run's document set —
   and caches ``plan.pt`` + ``reject_doc_ids.json`` under ``--output-dir``.
2. REBUILD (one slurm array task per rank, no ``--prepare-only``): loads the
   cached plan and reject list, reads the spill at ``--spill-dir/rank_NNNN/``,
   and writes clean bin/idx under ``--output-dir`` via ``rebuild_rank``.

Pass the SAME plan-shaping args as the original tokenize run in both steps —
see the per-arg ``--help`` text and docs/dataset-exclusion-filtering.md §6.

Usage::

    # 1) head node
    python scripts/rebuild_decontaminated.py \\
        --manifest /capstor/scratch/.../manifest.parquet \\
        --contamination-ids /iopsstor/.../innovator_vl_contaminated_ids.txt \\
        --output-dir /capstor/store/.../tokenized/sft/innovator_vl_46m_decontaminated \\
        --tokenizer-path /capstor/store/.../tokenizer/apertus_emu3.5_wavtok_instruct \\
        --mode sft --min-pixels "2048*2048" --max-pixels "2048*2048" \\
        --prepare-only

    # 2) per rank (slurm array over 0..EXPECTED_RANKS-1); same args plus:
    python scripts/rebuild_decontaminated.py \\
        ... \\
        --spill-dir /capstor/store/.../tokenized/sft/innovator_vl_46m \\
        --rank $SLURM_ARRAY_TASK_ID
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402  (repo is not pip-installable)

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from vision_tokenization.common.assembly import StructureTokenIds  # noqa: E402
from vision_tokenization.discrete.emu import create_tokenizer  # noqa: E402
from vision_tokenization.indexing.planning.tokenization_plan import build_tokenization_plan  # noqa: E402
from vision_tokenization.pipeline.output.rebuild import rebuild_rank  # noqa: E402
from vision_tokenization.utils.contamination import (  # noqa: E402
    HfSourceRowResolver,
    assert_unique_resolution,
    load_contamination_index,
)

logger = logging.getLogger(__name__)


def _pixel_count(value: str) -> int:
    """Parse a pixel budget given as an int or a product like ``2048*2048``.

    Matches how dataset configs express ``min_pixels``/``max_pixels`` so operators
    can copy the original tokenize run's value verbatim.
    """
    total = 1
    for part in value.split("*"):
        total *= int(part)
    return total


def compute_rejected_doc_ids(
    manifest_path: Path,
    contamination_ids_path: Path,
    contamination_format: str,
    plan,
) -> set[int]:
    """Map contamination IDs -> set of plan doc_ids to skip during rebuild.

    Pipeline: contamination ID -> (shard_stem, source_row_in_shard) ->
    manifest row whose ``chunk_offsets[chunk_index] + row_in_chunk == source_row``
    -> manifest ``group_id`` -> plan doc_id (position in ``np.unique(group_ids)``).

    This positional mapping is only valid when ``np.unique(group_ids)`` over the
    whole manifest matches the planner's documents. The planner derives documents
    from ``np.unique(valid_gids)`` over *pixel/min-dimension-filtered* rows and
    drops a whole group if any of its images fails, so a single dropped group
    shifts every later doc_id. We therefore abort if the unique-group count does
    not equal ``plan.total_documents`` rather than silently dropping the wrong
    documents (see CLAUDE.md "Known limitations").
    """
    index = load_contamination_index(contamination_ids_path, format=contamination_format)
    logger.info(
        "Loaded %d contamination IDs across %d source shards",
        index.total_ids,
        len(index.by_source),
    )

    # Single-image manifests (image2text/image_only/text2image) have no group_id
    # column; each row is its own document keyed by sample_index, matching the
    # planner's synthesized group ids.
    parquet_file = pq.ParquetFile(manifest_path)
    doc_key_col = "group_id" if "group_id" in parquet_file.schema_arrow.names else "sample_index"
    tbl = parquet_file.read(columns=["shard_path", "chunk_index", "row_in_chunk", doc_key_col])
    shard_paths = tbl.column("shard_path").to_pylist()
    chunk_indices = tbl.column("chunk_index").combine_chunks().to_numpy(zero_copy_only=False)
    rows_in_chunk = tbl.column("row_in_chunk").combine_chunks().to_numpy(zero_copy_only=False)
    group_ids = tbl.column(doc_key_col).combine_chunks().to_numpy(zero_copy_only=False)

    # Abort on ambiguous ids (one id matching multiple shards) before rejecting
    # anything — consistent with the scan-time and decontaminate_manifest paths.
    unmatched = assert_unique_resolution(index, list(dict.fromkeys(shard_paths)), source="manifest")
    if unmatched:
        logger.warning(
            "%d contamination id key(s) matched no shard in the manifest (first few: %s)",
            len(unmatched),
            unmatched[:5],
        )

    resolver = HfSourceRowResolver(index)
    contaminated_gids: set[int] = set()
    for row_idx, shard_path in enumerate(shard_paths):
        if resolver.contaminated_source_row(shard_path, chunk_indices[row_idx], rows_in_chunk[row_idx]) is not None:
            contaminated_gids.add(int(group_ids[row_idx]))

    contam_arr_for_count = np.fromiter(contaminated_gids, dtype=group_ids.dtype, count=len(contaminated_gids))
    logger.info(
        "Found %d contaminated group_ids in manifest (%d manifest rows mention them)",
        len(contaminated_gids),
        int(np.isin(group_ids, contam_arr_for_count).sum()),
    )

    # Plan doc_id N corresponds to the Nth-smallest unique group_id
    # (planner uses ``np.unique`` over valid group_ids).
    unique_gids = np.unique(group_ids)
    if not contaminated_gids:
        return set()
    # Guardrail: the position-based mapping below is only correct when no group was
    # dropped by the planner's pixel/min-dimension filter. If counts differ, the
    # doc_id positions are shifted and we would drop the wrong documents.
    if len(unique_gids) != plan.total_documents:
        raise SystemExit(
            f"Cannot map contaminated group_ids to plan doc_ids: manifest has "
            f"{len(unique_gids):,} unique group_ids but the plan has "
            f"{plan.total_documents:,} documents. Pixel/min-dimension filtering dropped "
            f"{len(unique_gids) - plan.total_documents:,} group(s), so np.unique(group_ids) "
            f"positions no longer match the planner's documents. Use scan-time filtering or "
            f"scripts/decontaminate_manifest.py instead, or extend compute_rejected_doc_ids "
            f"to derive doc_ids from the loaded plan (see CLAUDE.md 'Known limitations')."
        )
    contam_arr = np.array(sorted(contaminated_gids), dtype=unique_gids.dtype)
    positions = np.searchsorted(unique_gids, contam_arr)
    # Sanity-check: every contaminated gid must resolve to a unique gid in the manifest
    valid = (positions < len(unique_gids)) & (unique_gids[np.minimum(positions, len(unique_gids) - 1)] == contam_arr)
    if not valid.all():
        missing = contam_arr[~valid]
        logger.warning(
            "%d contaminated group_ids did not match any unique gid in manifest " "(first few: %s)",
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
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_images_per_doc=args.max_images_per_doc,
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


#: Plan args that change the DOCUMENT SET — a mismatch vs the original tokenize run
#: shifts doc_id positions and is caught by the document-count guardrail (abort).
_DOC_FILTER_HELP = (
    "Document filter — MUST match the original tokenize run; a mismatch is caught by the doc_id guardrail (abort)."
)
#: Plan args that shape batches/ordering but not the document set — copy them
#: verbatim from the original run; a mismatch here is NOT auto-detected.
_PLAN_ARG_HELP = "Plan arg — copy verbatim from the original tokenize run (a mismatch is not auto-detected)."


def main(argv: Optional[list[str]] = None) -> int:
    # RawDescriptionHelpFormatter keeps the two-step Usage:: block readable in --help.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--contamination-ids", required=True, type=Path)
    parser.add_argument("--contamination-format", default="innovator_vl")
    parser.add_argument("--spill-dir", type=Path, default=None, help="Required unless --prepare-only.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=None, help="Required unless --prepare-only.")
    parser.add_argument("--tokenizer-path", required=True, type=str)
    parser.add_argument("--mode", default="sft", help=_DOC_FILTER_HELP)
    parser.add_argument("--text-column", default="conversations", help=_PLAN_ARG_HELP)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128, help=_PLAN_ARG_HELP)
    parser.add_argument("--max-batch-tokens", type=int, default=32768, help=_PLAN_ARG_HELP)
    parser.add_argument("--max-sequence-tokens", type=int, default=None)
    parser.add_argument("--seqlen-threshold", type=int, default=None)
    parser.add_argument(
        "--min-pixels",
        type=_pixel_count,
        default=None,
        help=_DOC_FILTER_HELP + ' Accepts the config product form, e.g. "2048*2048".',
    )
    parser.add_argument(
        "--max-pixels",
        type=_pixel_count,
        default=None,
        help=_DOC_FILTER_HELP + ' Accepts the config product form, e.g. "2048*2048".',
    )
    parser.add_argument("--max-images-per-doc", type=int, default=None, help=_DOC_FILTER_HELP)
    parser.add_argument("--spatial-factor", type=int, default=16, help=_DOC_FILTER_HELP)
    parser.add_argument("--tokenizer-min-pixels", type=int, default=16384, help=_PLAN_ARG_HELP)
    parser.add_argument("--tokenizer-max-pixels", type=int, default=1960000, help=_PLAN_ARG_HELP)
    parser.add_argument("--window-size", type=int, default=2000, help=_PLAN_ARG_HELP)
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

    # This script reconstructs HF (shard_path/chunk_index/row_in_chunk) doc_ids; it
    # cannot map WebDataset sample_key ids. Reject wds_key loudly rather than
    # silently rejecting nothing (str keys never match the int source_row lookup).
    if args.contamination_format not in ("innovator_vl", "stem_row"):
        parser.error(
            f"rebuild_decontaminated.py supports HF contamination formats only "
            f"(innovator_vl, stem_row), got {args.contamination_format!r}. For WebDataset, "
            f"exclude at scan time or with scripts/decontaminate_manifest.py."
        )

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

    import torch

    if args.prepare_only:
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

        # Build reject_doc_ids against the plan so the unique-group guardrail can
        # abort before the slurm array if positions would not align.
        logger.info("Computing rejected doc_ids …")
        reject_doc_ids = compute_rejected_doc_ids(
            args.manifest,
            args.contamination_ids,
            args.contamination_format,
            plan,
        )
        args.reject_doc_ids_json.parent.mkdir(parents=True, exist_ok=True)
        args.reject_doc_ids_json.write_text(json.dumps(sorted(reject_doc_ids)))
        logger.info("Saved reject_doc_ids to %s (%d entries)", args.reject_doc_ids_json, len(reject_doc_ids))
        return 0

    if not args.plan_pt.exists():
        parser.error(f"plan.pt not found at {args.plan_pt}. " "Run with --prepare-only first to produce it.")
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
