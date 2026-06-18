"""Engine-side per-pair text tokenization for alignment, run CPU-parallel to the GPU
vision encode. Each rank tokenizes a disjoint slice of preference pairs (text only,
vision-free) and spills them to ``views_tokenized/rank_NNNN.parquet``; the merge joins
these with the encoded vision block lengths to build the ``.bin`` and the DPO slice
lengths. Image slots are positional here — no vision tokens are read.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from vision_tokenization.discrete.conversation import ConversationPolicy
from vision_tokenization.discrete.dpo_pairs import tokenize_pair_text, tokenized_to_row
from vision_tokenization.discrete.sft_segments import build_image_marker_candidates
from vision_tokenization.indexing.alignment.payload import (
    TOKENIZED_SCHEMA,
    _alignment_media_refs,
    tokenized_views_dir,
)
from vision_tokenization.pipeline.output.alignment_merge import _fsync_file


def _write_parquet_atomic(table: pa.Table, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    _fsync_file(tmp)
    os.replace(tmp, path)


def tokenize_pair_views(views_path, spill_dir, rank: int, world_size: int,
                        *, tokenizer_path, system: str = "empty") -> int:
    """Tokenize this rank's disjoint slice of preference pairs from ``views_path`` and
    spill to ``<spill_dir>/views_tokenized/rank_NNNN.parquet``. CPU-only and vision-free;
    safe to run concurrently with the rank's GPU vision encode. Returns the row count."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True, use_fast=True)
    policy = ConversationPolicy(add_system_message=(system == "empty"))
    markers = build_image_marker_candidates(text_tokenizer=tok, conversation_policy=policy)

    table = pq.read_table(views_path)
    mine = table.take(list(range(rank, table.num_rows, world_size)))
    rows = []
    for row in mine.to_pylist():
        tp = tokenize_pair_text(row, tok, policy, markers)
        n_refs = len(_alignment_media_refs(row))
        if len(tp.image_insert_positions) != n_refs:
            raise ValueError(
                f"{row['prompt_id']}: {len(tp.image_insert_positions)} image slots but {n_refs} media refs")
        rows.append(tokenized_to_row(tp, row["prompt_id"]))

    out_dir = tokenized_views_dir(spill_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet_atomic(pa.Table.from_pylist(rows, schema=TOKENIZED_SCHEMA), out_dir / f"rank_{rank:04d}.parquet")
    return len(rows)


def main(argv=None) -> int:
    """Run one rank's text pass as a standalone process, concurrent with the GPU encode."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("views_path")
    p.add_argument("spill_dir")
    p.add_argument("rank", type=int)
    p.add_argument("world_size", type=int)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--system", default="empty")
    a = p.parse_args(argv)
    n = tokenize_pair_views(a.views_path, a.spill_dir, a.rank, a.world_size,
                            tokenizer_path=a.tokenizer_path, system=a.system)
    print(f"alignment text pass: rank {a.rank} tokenized {n} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
