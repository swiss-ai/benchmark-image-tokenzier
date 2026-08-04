"""Bucket a tokenized dataset by sequence length.

Writes only sequences whose length falls within ``[--min-token, --max-token]``
to a new dataset. Optionally strips the default Apertus 1.5 system prompt from
edge sequences (just over ``--max-token``) so they shrink enough to fit.

Built on ``rewrite_dataset`` (Megatron-native ``IndexedDataset`` read +
``IndexedDatasetBuilder`` write) — no hand-rolled .bin/.idx byte surgery.

Examples
--------
Build a 16k bucket from an lct dataset, shrinking edge cases to fit::

    python -m vision_tokenization.pipeline.output.bucket_by_length \
        --input  /.../Apertus1p5_sft_lct_tokenized/google_maptrace \
        --output /.../Apertus1p5_sft_16k_tokenized/google_maptrace \
        --min-token 0 --max-token 16384 \
        --strip-default-sysprompt

Dry run (report counts, write nothing)::

    python -m vision_tokenization.pipeline.output.bucket_by_length \
        --input  /.../google_maptrace --output /tmp/x \
        --max-token 16384 --strip-default-sysprompt --dry-run
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Known default system blocks (tokenizer apertus_emu3.5_wavtok_instruct_thinking_token_fixed).
# Each block is [<|system_start|>(61), ...content..., <|system_end|>(62)], and in a sequence
# is preceded by <s>(BOS). Stripping keeps <s> + both markers and removes only the content,
# turning the block into the empty form [..., 61, 62, ...].
#
#   Block 0 — Apertus 1.5 Omni default (43 content tok): "You are Apertus 1.5 Omni, a
#             multimodal assistant developed by the Swiss AI Initiative...". Used across the
#             vision/omni SFT (innovator, nemotron, sensenova, ...).
#   Block 1 — audio Chinese default (13 content tok): "你是个有用的音频理解助手。" ("You are a
#             helpful audio understanding assistant."). Used by teleantifraud_matching_sft.
#
# To recognize a new default, append its exact [61, ...content..., 62] array here — the
# dropout + edge-fit logic below is block-agnostic (it strips whichever block matched).
DEFAULT_SYS_BLOCKS = [
    np.array(
        [61, 4568, 1584, 80417, 1374, 1032, 1049, 1046, 1053, 27829, 2729, 1044,
         1261, 59450, 50786, 27089, 7291, 1536, 1278, 35922, 26554, 54106, 1046,
         69540, 1562, 80417, 1374, 1032, 1049, 5059, 8971, 16425, 15981, 1044,
         1636, 5048, 8061, 1321, 16023, 1321, 9148, 1294, 3403, 1046, 62],
        dtype=np.int64,
    ),
    np.array(
        [61, 7543, 2499, 6973, 4673, 12600, 2713, 16607, 12684, 1145, 94167,
         43340, 13571, 1320, 62],
        dtype=np.int64,
    ),
]
# Widest default-content length (markers excluded) — sizes the edge-fit band.
SYS_CONTENT_LEN = max(len(b) for b in DEFAULT_SYS_BLOCKS) - 2  # 43
BOS_ID = 1


def _match_default_block(seq: np.ndarray):
    """Return the default block (np.ndarray) the seq carries, or None.

    A seq carries a default block iff it starts with <s> immediately followed by
    the exact block [<|system_start|>, ...content..., <|system_end|>].
    """
    if len(seq) < 1 or seq[0] != BOS_ID:
        return None
    for blk in DEFAULT_SYS_BLOCKS:
        L = len(blk)
        if len(seq) >= 1 + L and np.array_equal(np.asarray(seq[1:1 + L]), blk):
            return blk
    return None


def _strip_default_content(seq: np.ndarray, blk: np.ndarray) -> np.ndarray:
    """Remove a matched block's content tokens, keeping <s> + both markers + rest.

    [<s>, <|system_start|>, ...content..., <|system_end|>, <dev>, ...]
      ->  [<s>, <|system_start|>, <|system_end|>, <dev>, ...]
    """
    return np.concatenate([seq[:2], seq[len(blk):]])


def make_length_transform(
    min_token: int,
    max_token: int,
    strip_default_sysprompt: bool,
    sysprompt_dropout_rate: float = 0.0,
    seed: int = 42,
):
    """Build a per-sequence transform for ``rewrite_dataset``.

    Order of operations per sequence:

    1. **System-prompt dropout** (if ``sysprompt_dropout_rate > 0``): for a seq
       carrying the exact default block, with probability ``dropout_rate`` drop
       the 43 content tokens (markers kept → empty-block form). This is the
       train/inference robustness augmentation; it never drops a sequence, only
       shortens it.
    2. **Length bucketing**: keep seqs with ``min_token <= len <= max_token``.
    3. **Edge fit** (if ``strip_default_sysprompt``): a seq still in
       ``(max_token, max_token + 43]`` with the default block gets force-stripped
       so it fits — this catches edge cases that dropout left untouched, so the
       bucket boundary is always respected.

    Sequences outside the bucket (and unshrinkable) are dropped (return ``None``).
    Dropout uses a seeded RNG; since ``rewrite_dataset`` iterates in order, the
    decision sequence is deterministic for a given seed.
    """
    rng = np.random.default_rng(seed) if sysprompt_dropout_rate > 0 else None

    def transform(seq: np.ndarray) -> Optional[np.ndarray]:
        # 1. system-prompt dropout
        if rng is not None:
            blk = _match_default_block(seq)
            if blk is not None and rng.random() < sysprompt_dropout_rate:
                seq = _strip_default_content(seq, blk)

        # 2. length bucketing
        length = len(seq)
        if min_token <= length <= max_token:
            return seq

        # 3. edge fit (force-strip the 20% that dropout left, if they're edge cases)
        if strip_default_sysprompt and max_token < length <= max_token + SYS_CONTENT_LEN:
            blk = _match_default_block(seq)
            if blk is not None:
                stripped = _strip_default_content(seq, blk)
                if min_token <= len(stripped) <= max_token:
                    return stripped
        return None

    return transform


def _dry_run_report(input_prefix: str, min_token: int, max_token: int,
                    strip_default_sysprompt: bool) -> None:
    """Report length-based counts from the .idx without writing anything."""
    import sys
    sys.path.insert(0, "/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM")
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    ds = IndexedDataset(input_prefix)
    lengths = np.asarray(ds.index.sequence_lengths)
    n = len(lengths)

    under_min = int((lengths < min_token).sum())
    fit = int(((lengths >= min_token) & (lengths <= max_token)).sum())
    edge = int(((lengths > max_token) & (lengths <= max_token + SYS_CONTENT_LEN)).sum())
    over = int((lengths > max_token + SYS_CONTENT_LEN).sum())

    kept = fit + (edge if strip_default_sysprompt else 0)
    dropped = n - kept

    print(f"[dry-run] {input_prefix}")
    print(f"  total sequences:                 {n:,}")
    print(f"  under min ({min_token}):              {under_min:,}")
    print(f"  fit [{min_token},{max_token}] as-is:        {fit:,}")
    print(f"  edge ({max_token},{max_token + SYS_CONTENT_LEN}] (strippable*): {edge:,}")
    print(f"  over {max_token + SYS_CONTENT_LEN}:                    {over:,}")
    print(f"  --> would keep:                  {kept:,}  (strip={'on' if strip_default_sysprompt else 'off'})")
    print(f"  --> would drop:                  {dropped:,}")
    print("  * edge count is length-based; actual strip also requires the exact")
    print("    default system block (content-gated at write time).")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input dataset prefix (no .bin/.idx)")
    parser.add_argument("--output", required=True, help="Output dataset prefix (no .bin/.idx)")
    parser.add_argument("--min-token", type=int, default=0, help="Minimum sequence length (inclusive)")
    parser.add_argument("--max-token", type=int, required=True, help="Maximum sequence length (inclusive)")
    parser.add_argument("--strip-default-sysprompt", action="store_true",
                        help="Strip the 43-token default system-prompt content from edge "
                             "sequences in (max-token, max-token+43] so they fit")
    parser.add_argument("--sysprompt-dropout-rate", type=float, default=0.0,
                        help="Probability of dropping the 43 default system-prompt content "
                             "tokens from each sequence carrying the default block (0.0=off). "
                             "Markers are kept (empty-block form). E.g. 0.8 keeps the prompt "
                             "in ~20%% of sequences.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for dropout")
    parser.add_argument("--dry-run", action="store_true", help="Report counts; write nothing")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.dry_run:
        _dry_run_report(args.input, args.min_token, args.max_token,
                        args.strip_default_sysprompt)
        return 0

    from vision_tokenization.pipeline.output.merge import rewrite_dataset

    transform = make_length_transform(
        args.min_token, args.max_token, args.strip_default_sysprompt,
        sysprompt_dropout_rate=args.sysprompt_dropout_rate, seed=args.seed,
    )
    stats = rewrite_dataset(args.input, args.output, transform)

    print(f"bucket_by_length: {args.input} -> {args.output}")
    print(f"  range: [{args.min_token}, {args.max_token}]  edge-strip={'on' if args.strip_default_sysprompt else 'off'}"
          f"  dropout={args.sysprompt_dropout_rate} (seed={args.seed})")
    print(f"  input:   {stats.input_count:,}")
    print(f"  written: {stats.written_count:,}")
    print(f"  dropped: {stats.skipped_count:,}")
    print(f"  output_tokens: {stats.output_tokens:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
