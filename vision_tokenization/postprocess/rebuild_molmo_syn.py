#!/usr/bin/env python3
"""Rebuild Molmo-SynMultiImageQA spill into two training formats.

Format 1 — image2text: [BOS] image_i code_i [EOS]
    One sequence per image-code pair. Routes by seqlen_threshold.

Format 2 — multi-image: [BOS] image_0 image_1 ... image_N overall_description [EOS]
    One sequence per document. Skips documents with empty overall_description.
    Routes by seqlen_threshold.

Usage::

    python -m vision_tokenization.preprocess.rebuild_molmo_syn \
        --spill-dir /path/to/spill \
        --output-dir /path/to/output \
        --seqlen-threshold 8192 \
        --num-workers 16
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm

import numpy as np

logger = logging.getLogger(__name__)

IMAGE_KIND = 0
TEXT_KIND = 1


def _load_rank_components(rank_dir: Path):
    """Load all components + token loader for one rank."""
    from vision_tokenization.pipeline.output.spill import ComponentSpillReader
    from vision_tokenization.pipeline.output.rebuild import (
        _extract_sorted_spill,
        _build_provenance_single_rank,
        _make_token_loader_single_rank,
        _PROVENANCE_KEY_SHIFT,
    )

    spill_table = ComponentSpillReader.read_rank(rank_dir)
    if len(spill_table) == 0:
        return None

    spill = _extract_sorted_spill(spill_table)
    spill_key = (
        spill["doc_ids"].astype(np.int64) * (1 << _PROVENANCE_KEY_SHIFT)
        + spill["comp_idx"].astype(np.int64)
    )
    prov_shard_id, prov_offset, prov_length, token_mmaps = _build_provenance_single_rank(
        rank_dir, spill_key, spill["n"],
    )
    load_tokens = _make_token_loader_single_rank(
        prov_shard_id, prov_offset, prov_length, token_mmaps, rank_dir,
    )
    return spill, prov_shard_id, load_tokens


def _write_seq(seq, threshold, builders, stats_key):
    """Route a sequence to stage2 or lct based on threshold."""
    seq_len = len(seq)
    if threshold is not None and seq_len > threshold:
        builders[f"{stats_key}_lct"].add_item(seq)
        builders[f"{stats_key}_lct"].end_document()
        return f"{stats_key}_lct", seq_len
    builders[f"{stats_key}_stage2"].add_item(seq)
    builders[f"{stats_key}_stage2"].end_document()
    return f"{stats_key}_stage2", seq_len


def _partition_text_components(
    texts,
    sorted_img_indices,
):
    """Split Molmo text components into per-image code blocks and desc.

    Molmo spill comes from an interleave parser that emits:
    ``[image, code?, image, code?, ..., overall_description?]``.
    Empty code strings are omitted entirely, so text component indices are not
    dense. We therefore recover code/description by looking at the text
    components that fall between consecutive image components.

    The only ambiguous case is a single trailing text after the last image:
    it could be the last code block or the overall description. To avoid
    fabricating incorrect outputs, that case is treated as "no desc" and no
    last image/code pair.
    """
    if not texts or not sorted_img_indices:
        return {}, None

    code_by_image: dict[int, np.ndarray] = {}
    text_items = sorted(texts.items())

    for idx, img_ci in enumerate(sorted_img_indices[:-1]):
        next_img_ci = sorted_img_indices[idx + 1]
        between = [(ci, toks) for ci, toks in text_items if img_ci < ci < next_img_ci]
        if between:
            code_by_image[img_ci] = between[0][1]

    last_img_ci = sorted_img_indices[-1]
    trailing = [(ci, toks) for ci, toks in text_items if ci > last_img_ci]
    if len(trailing) >= 2:
        code_by_image[last_img_ci] = trailing[0][1]
        return code_by_image, trailing[-1][1]
    return code_by_image, None


def _extract_overall_description(texts, sorted_img_indices):
    """Return the dedicated trailing overall_description component, if present."""
    _code_by_image, desc = _partition_text_components(texts, sorted_img_indices)
    return desc


def rebuild_rank(
    rank_dir: Path,
    output_dir: Path,
    rank: int,
    bos_id: int,
    eos_id: int,
    vocab_size: int,
    seqlen_threshold: int = 8192,
) -> Dict:
    """Rebuild one rank into image2text + multi-image formats with stage2/lct split."""
    from vision_tokenization.formats.megatron import DType, IndexedDatasetBuilder

    result = _load_rank_components(rank_dir)
    if result is None:
        return {"rank": rank}

    spill, prov_shard_id, load_tokens = result
    unique_docs = spill["unique_docs"]
    doc_starts = spill["doc_starts"]
    doc_ends = spill["doc_ends"]
    kinds = spill["kinds"]
    comp_idx = spill["comp_idx"]

    megatron_dtype = DType.optimal_dtype(vocab_size)
    shard_name = f"rank_{rank:04d}_chunk_0000"

    # Create builders: 4 outputs (image2text × {stage2,lct} + multiimage × {stage2,lct})
    builder_specs = {
        "i2t_stage2": output_dir / "image2text" / "stage2",
        "i2t_lct": output_dir / "image2text" / "lct",
        "mi_stage2": output_dir / "multiimage" / "stage2",
        "mi_lct": output_dir / "multiimage" / "lct",
    }
    builders = {}
    for key, d in builder_specs.items():
        d.mkdir(parents=True, exist_ok=True)
        builders[key] = IndexedDatasetBuilder(str(d / shard_name) + ".bin", dtype=megatron_dtype)

    bos = np.array([bos_id], dtype=np.int32)
    eos = np.array([eos_id], dtype=np.int32)

    counters: Dict[str, int] = {}

    for di in range(len(unique_docs)):
        cs = int(doc_starts[di])
        ce = int(doc_ends[di])

        images = {}
        texts = {}
        for ri in range(cs, ce):
            if prov_shard_id[ri] < 0:
                continue
            ci = int(comp_idx[ri])
            tokens = load_tokens(ri)
            if kinds[ri] == IMAGE_KIND:
                images[ci] = tokens
            else:
                texts[ci] = tokens

        if not images:
            continue

        sorted_img_indices = sorted(images.keys())

        code_by_image, desc = _partition_text_components(texts, sorted_img_indices)

        # --- image2text: image_i + code_i pairs ---
        for img_ci in sorted_img_indices:
            code = code_by_image.get(img_ci)
            if code is None or len(code) == 0:
                continue
            seq = np.concatenate([bos, images[img_ci], code, eos]).astype(megatron_dtype)
            bucket, seq_len = _write_seq(seq, seqlen_threshold, builders, "i2t")
            counters[f"{bucket}_seqs"] = counters.get(f"{bucket}_seqs", 0) + 1
            counters[f"{bucket}_tokens"] = counters.get(f"{bucket}_tokens", 0) + seq_len

        # --- multi-image: all images + overall_description ---
        if desc is None or len(desc) == 0:
            counters["mi_skipped"] = counters.get("mi_skipped", 0) + 1
            continue

        parts = [bos]
        for img_ci in sorted_img_indices:
            parts.append(images[img_ci])
        parts.append(desc)
        parts.append(eos)
        seq = np.concatenate(parts).astype(megatron_dtype)
        bucket, seq_len = _write_seq(seq, seqlen_threshold, builders, "mi")
        counters[f"{bucket}_seqs"] = counters.get(f"{bucket}_seqs", 0) + 1
        counters[f"{bucket}_tokens"] = counters.get(f"{bucket}_tokens", 0) + seq_len

    from vision_tokenization.pipeline.output.rebuild import finalize_builders
    prefixes = {key: d / shard_name for key, d in builder_specs.items()}
    finalize_builders(builders, prefixes)

    counters["rank"] = rank
    return counters


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--spill-dir",
        default="/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/interleave/molmo_syn_multiimage",
    )
    parser.add_argument(
        "--output-dir",
        default="/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/molmo_syn_multiimage",
    )
    parser.add_argument("--seqlen-threshold", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--tokenizer-path",
        default="/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    spill_dir = Path(args.spill_dir)
    output_dir = Path(args.output_dir)

    rank_dirs = sorted(
        p for p in spill_dir.glob("rank_*")
        if p.is_dir() and (p / "_SUCCESS").exists()
    )
    logger.info(f"Found {len(rank_dirs)} rank dirs, threshold={args.seqlen_threshold}")

    totals: Dict[str, int] = {}

    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {
            pool.submit(
                rebuild_rank, rd, output_dir, i,
                tok.bos_token_id, tok.eos_token_id, len(tok),
                args.seqlen_threshold,
            ): i
            for i, rd in enumerate(rank_dirs)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Ranks", unit="rank"):
            stats = future.result()
            for k, v in stats.items():
                if k != "rank":
                    totals[k] = totals.get(k, 0) + v

    logger.info("=== Results ===")
    for k, v in sorted(totals.items()):
        logger.info(f"  {k}: {v:,}")

    import json
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
