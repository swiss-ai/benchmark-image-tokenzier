#!/usr/bin/env python3
"""Rebuild multilingual recap spill into per-language image-text pairs.

Each document has: [image, text_lang0, text_lang1, ...]
Produces: [BOS] image text_lang_i [EOS] — one sequence per language.

Usage::

    # MIT-10M recap (14 languages)
    python -m vision_tokenization.preprocess.rebuild_multilingual \
        --spill-dir /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/interleave/mit_10m_recap \
        --output-dir /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/mit_10m_recap_perlang \
        --num-workers 16

    # Art Museums recap (20+ languages)
    python -m vision_tokenization.preprocess.rebuild_multilingual \
        --spill-dir /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/interleave/art_museums_recap \
        --output-dir /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/art_museums_recap_perlang \
        --num-workers 16
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

IMAGE_KIND = 0
TEXT_KIND = 1


def _load_rank_components(rank_dir: Path):
    """Load spill components + token loader for one rank."""
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


def _write_seq(seq, threshold, builders):
    """Route a sequence to stage2 or lct."""
    seq_len = len(seq)
    bucket = "lct" if threshold is not None and seq_len > threshold else "stage2"
    builders[bucket].add_item(seq)
    builders[bucket].end_document()
    return bucket, seq_len


def rebuild_rank(
    rank_dir: Path,
    output_dir: Path,
    rank: int,
    bos_id: int,
    eos_id: int,
    vocab_size: int,
    seqlen_threshold: int = 8192,
) -> Dict:
    """Rebuild one rank: each image × each language → one sequence."""
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

    builders = {}
    prefixes = {}
    for bucket in ("stage2", "lct"):
        d = output_dir / bucket
        d.mkdir(parents=True, exist_ok=True)
        prefixes[bucket] = d / shard_name
        builders[bucket] = IndexedDatasetBuilder(
            str(prefixes[bucket]) + ".bin", dtype=megatron_dtype,
        )

    bos = np.array([bos_id], dtype=np.int32)
    eos = np.array([eos_id], dtype=np.int32)

    counters: Dict[str, int] = {}

    for di in range(len(unique_docs)):
        cs = int(doc_starts[di])
        ce = int(doc_ends[di])

        image_tokens = None
        text_components = []

        for ri in range(cs, ce):
            if prov_shard_id[ri] < 0:
                continue
            tokens = load_tokens(ri)
            if kinds[ri] == IMAGE_KIND:
                image_tokens = tokens
            else:
                text_components.append((int(comp_idx[ri]), tokens))

        if image_tokens is None or not text_components:
            continue

        text_components.sort(key=lambda x: x[0])

        for _, text_tokens in text_components:
            seq = np.concatenate([bos, image_tokens, text_tokens, eos]).astype(
                megatron_dtype,
            )
            bucket, seq_len = _write_seq(seq, seqlen_threshold, builders)
            counters[f"{bucket}_seqs"] = counters.get(f"{bucket}_seqs", 0) + 1
            counters[f"{bucket}_tokens"] = counters.get(f"{bucket}_tokens", 0) + seq_len

    from vision_tokenization.pipeline.output.rebuild import finalize_builders
    finalize_builders(builders, prefixes)

    counters["rank"] = rank
    return counters


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--spill-dir", required=True)
    parser.add_argument("--output-dir", required=True)
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
