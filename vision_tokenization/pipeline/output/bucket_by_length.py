"""Bucket a tokenized dataset by sequence length.

Writes only sequences whose length falls within ``[--min-token, --max-token]``
to a new dataset.

Built on ``rewrite_dataset`` (Megatron-native ``IndexedDataset`` read +
``IndexedDatasetBuilder`` write) — no hand-rolled .bin/.idx byte surgery.

Examples
--------
Build a 16k bucket from an lct dataset::

    python -m vision_tokenization.pipeline.output.bucket_by_length \
        --input  /.../Apertus1p5_sft_lct_tokenized/google_maptrace \
        --output /.../Apertus1p5_sft_16k_tokenized/google_maptrace \
        --min-token 0 --max-token 16384

Dry run (report counts, write nothing)::

    python -m vision_tokenization.pipeline.output.bucket_by_length \
        --input  /.../google_maptrace --output /tmp/x \
        --max-token 16384 --dry-run
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def make_length_transform(min_token: int, max_token: int):
    """Build a per-sequence transform for ``rewrite_dataset``.

    Keeps sequences with ``min_token <= len <= max_token``; everything else is
    dropped (return ``None``).
    """

    def transform(seq: np.ndarray) -> Optional[np.ndarray]:
        return seq if min_token <= len(seq) <= max_token else None

    return transform


def _dry_run_report(input_prefix: str, min_token: int, max_token: int) -> None:
    """Report length-based counts from the .idx without writing anything."""
    import sys
    sys.path.insert(0, "/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM")
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    ds = IndexedDataset(input_prefix)
    lengths = np.asarray(ds.index.sequence_lengths)
    n = len(lengths)

    under_min = int((lengths < min_token).sum())
    kept = int(((lengths >= min_token) & (lengths <= max_token)).sum())
    over = int((lengths > max_token).sum())

    print(f"[dry-run] {input_prefix}")
    print(f"  total sequences:                 {n:,}")
    print(f"  under min ({min_token}):              {under_min:,}")
    print(f"  over max ({max_token}):               {over:,}")
    print(f"  --> would keep:                  {kept:,}")
    print(f"  --> would drop:                  {n - kept:,}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input dataset prefix (no .bin/.idx)")
    parser.add_argument("--output", required=True, help="Output dataset prefix (no .bin/.idx)")
    parser.add_argument("--min-token", type=int, default=0, help="Minimum sequence length (inclusive)")
    parser.add_argument("--max-token", type=int, required=True, help="Maximum sequence length (inclusive)")
    parser.add_argument("--dry-run", action="store_true", help="Report counts; write nothing")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.dry_run:
        _dry_run_report(args.input, args.min_token, args.max_token)
        return 0

    from vision_tokenization.pipeline.output.merge import rewrite_dataset

    transform = make_length_transform(args.min_token, args.max_token)
    stats = rewrite_dataset(args.input, args.output, transform)

    print(f"bucket_by_length: {args.input} -> {args.output}")
    print(f"  range: [{args.min_token}, {args.max_token}]")
    print(f"  input:   {stats.input_count:,}")
    print(f"  written: {stats.written_count:,}")
    print(f"  dropped: {stats.skipped_count:,}")
    print(f"  output_tokens: {stats.output_tokens:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
