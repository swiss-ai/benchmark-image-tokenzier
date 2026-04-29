#!/usr/bin/env python3
"""Rebuild Argimi-Finance spill into interleaved sequences with page-atomic splitting.

Each document has interleaved [text?, image0, text..., image1, text..., ...].
Pre-merges each image + following text into an atomic "page". Greedy-packs
consecutive pages into sequences up to max_sequence_tokens with BOS/EOS.

Usage::

    python -m vision_tokenization.preprocess.rebuild_argimi_finance \
        --spill-dir /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/interleave/argimi_finance \
        --output-dir /capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/argimi_finance_interleave \
        --max-sequence-tokens 32768 \
        --seqlen-threshold 8192 \
        --num-workers 16
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

IMAGE_KIND = 0
TEXT_KIND = 1


def _load_rank_components(rank_dir: Path):
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


def _merge_into_pages(components: List[tuple]) -> List[tuple]:
    """Merge components into atomic pages: (tokens, token_count) per page.

    Each page = image + all following text until the next image.
    Text before first image prepends to first page.
    """
    components.sort(key=lambda x: x[0])

    pages: List[List[np.ndarray]] = []
    pending: List[np.ndarray] = []

    for _, kind, tokens in components:
        if kind == IMAGE_KIND:
            if pages and pending:
                pages[-1].extend(pending)
                pending = []
            pages.append(pending + [tokens])
            pending = []
        else:
            pending.append(tokens)

    if pending and pages:
        pages[-1].extend(pending)

    # Return (concatenated_tokens, length) for fast packing
    result = []
    for parts in pages:
        cat = np.concatenate(parts) if len(parts) > 1 else parts[0]
        result.append((cat, len(cat)))
    return result


def rebuild_rank(
    rank_dir: Path,
    output_dir: Path,
    rank: int,
    bos_id: int,
    eos_id: int,
    vocab_size: int,
    max_sequence_tokens: int = 32768,
    seqlen_threshold: int = 8192,
) -> Dict:
    from vision_tokenization.formats.megatron import DType, IndexedDatasetBuilder
    from vision_tokenization.pipeline.output.rebuild import finalize_builders

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

    bos_arr = np.array([bos_id], dtype=np.int32)
    eos_arr = np.array([eos_id], dtype=np.int32)

    counters: Dict[str, int] = {}

    def _emit(parts: List[np.ndarray], length: int) -> None:
        parts.append(eos_arr)
        seq = np.concatenate(parts).astype(megatron_dtype)
        bucket = "lct" if length + 1 > seqlen_threshold else "stage2"
        builders[bucket].add_item(seq)
        builders[bucket].end_document()
        counters[f"{bucket}_seqs"] = counters.get(f"{bucket}_seqs", 0) + 1
        counters[f"{bucket}_tokens"] = counters.get(f"{bucket}_tokens", 0) + length + 1

    for di in range(len(unique_docs)):
        cs = int(doc_starts[di])
        ce = int(doc_ends[di])

        components = []
        for ri in range(cs, ce):
            if prov_shard_id[ri] < 0:
                continue
            components.append((int(comp_idx[ri]), int(kinds[ri]), load_tokens(ri)))

        if not components:
            continue

        pages = _merge_into_pages(components)
        if not pages:
            continue

        # Greedy pack: BOS + pages... + EOS
        cur_parts = [bos_arr]
        cur_len = 1  # BOS

        for page_tokens, page_len in pages:
            # Single page + BOS + EOS > max → emit alone
            if page_len + 2 > max_sequence_tokens:
                if cur_len > 1:
                    _emit(cur_parts, cur_len)
                _emit([bos_arr, page_tokens], 1 + page_len)
                cur_parts = [bos_arr]
                cur_len = 1
                continue

            # Adding this page would exceed max → flush first
            if cur_len + page_len + 1 > max_sequence_tokens:  # +1 for EOS
                _emit(cur_parts, cur_len)
                cur_parts = [bos_arr]
                cur_len = 1

            cur_parts.append(page_tokens)
            cur_len += page_len

        if cur_len > 1:
            _emit(cur_parts, cur_len)

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
        default="/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/interleave/argimi_finance",
    )
    parser.add_argument(
        "--output-dir",
        default="/capstor/store/cscs/swissai/infra01/vision-datasets/tokenized/argimi_finance_interleave",
    )
    parser.add_argument("--max-sequence-tokens", type=int, default=32768)
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
    logger.info(
        f"Found {len(rank_dirs)} rank dirs, "
        f"max_seq={args.max_sequence_tokens}, threshold={args.seqlen_threshold}"
    )

    totals: Dict[str, int] = {}

    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {
            pool.submit(
                rebuild_rank, rd, output_dir, i,
                tok.bos_token_id, tok.eos_token_id, len(tok),
                args.max_sequence_tokens, args.seqlen_threshold,
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
