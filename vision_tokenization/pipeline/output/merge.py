"""Merge per-rank tokenized shards into a single dataset.

Follows the same "last rank out" pattern as ``stats_reducer``: each rank
calls ``maybe_merge_shards`` after finishing tokenization. The call checks
whether all expected ranks have completed (checkpoint files exist). The
first rank to observe a complete set performs the merge; others return
immediately.

Can also be run standalone::

    python -m vision_tokenization.pipeline.output.merge \
        /path/to/output_dir --expected-ranks 80
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Megatron import helper
# ---------------------------------------------------------------------------

def _ensure_megatron_importable() -> None:
    """Make ``megatron.core`` importable if it isn't already.

    Tries the normal import path first.  Falls back to a sibling
    ``Megatron-LM`` checkout relative to this repo, then ``MEGATRON_PATH``.
    """
    try:
        import megatron.core  # noqa: F401
        return
    except ImportError:
        pass

    candidates = [
        str(Path(__file__).resolve().parents[4] / "Megatron-LM"),
        os.environ.get("MEGATRON_PATH", ""),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            sys.path.insert(0, candidate)
            try:
                import megatron.core  # noqa: F401
                return
            except ImportError:
                pass

    raise ImportError(
        "Cannot import megatron.core. Either install megatron-core, "
        "set MEGATRON_PATH, or place a Megatron-LM checkout next to this repo."
    )


# ---------------------------------------------------------------------------
# Token-level transforms
# ---------------------------------------------------------------------------

import numba


@numba.njit
def _strip_thinking_inner(tokens, think_id, end_think_id, out):
    """Numba-compiled one-pass state machine. Returns write count."""
    w = 0
    inside = False
    for i in range(len(tokens)):
        tok = tokens[i]
        if inside:
            if tok == end_think_id:
                inside = False
        elif tok == think_id:
            inside = True
        elif tok == end_think_id:
            pass  # orphan close — drop
        else:
            out[w] = tok
            w += 1
    return w


def strip_thinking_tokens(
    tokens: np.ndarray,
    think_id: int = 32,
    end_think_id: int = 33,
) -> Optional[np.ndarray]:
    """Remove ``<think>...</think>`` spans from a token sequence.

    One-pass two-state delimiter machine (numba-compiled):

    - **outside**: keep tokens; ``think_id`` enters *inside*, ``end_think_id``
      (orphan close) is dropped.
    - **inside**: drop everything including repeated ``think_id``;
      ``end_think_id`` exits back to *outside*.

    Returns ``None`` if the result is empty after stripping.
    """
    n = len(tokens)
    if n == 0:
        return None
    if not ((tokens == think_id) | (tokens == end_think_id)).any():
        return tokens  # fast path — no copy

    out = np.empty(n, dtype=tokens.dtype)
    w = _strip_thinking_inner(tokens, think_id, end_think_id, out)

    return out[:w] if w > 0 else None


# ---------------------------------------------------------------------------
# Dataset rewrite
# ---------------------------------------------------------------------------

@dataclass
class RewriteStats:
    """Statistics from a ``rewrite_dataset`` call."""

    input_count: int
    written_count: int
    skipped_count: int
    output_tokens: int


def _index_has_sequence_modes(path_prefix: str) -> bool:
    """Inspect the .idx file on disk to detect sequence_modes metadata.

    Parses only the fixed 34-byte header to extract sequence_count and
    document_count, then checks whether the file contains the extra
    ``sequence_modes`` array.  Does not require a live ``IndexedDataset``.

    MMIDIDX v1 layout::

        header (9B) + version (8B) + dtype_code (1B)       = 18 B
        + sequence_count (8B) + document_count (8B)         = 16 B
        + sequence_lengths (seq_count × 4B)
        + sequence_pointers (seq_count × 8B)
        + document_indices (doc_count × 8B)
        [+ sequence_modes (seq_count × 1B)]   ← only if multimodal
    """
    import struct

    idx_path = Path(f"{path_prefix}.idx")
    idx_size = idx_path.stat().st_size

    with open(idx_path, "rb") as f:
        f.seek(18)  # skip header (9) + version (8) + dtype_code (1)
        seq_count, doc_count = struct.unpack("<QQ", f.read(16))

    plain_size = (
        34              # fixed header
        + seq_count * 4  # sequence_lengths (int32)
        + seq_count * 8  # sequence_pointers (int64)
        + doc_count * 8  # document_indices (int64)
    )

    if idx_size == plain_size:
        return False
    if idx_size == plain_size + seq_count:
        return True
    raise ValueError(
        f"Unrecognized indexed dataset layout for {path_prefix}: "
        f"idx_size={idx_size}, expected {plain_size} or {plain_size + seq_count}"
    )


def resolve_thinking_token_ids(tokenizer_path: str) -> tuple[int, int]:
    """Resolve ``<think>``/``</think>`` token IDs from a tokenizer.json file.

    Uses the Rust-based ``tokenizers`` library for fast loading (no full
    HuggingFace AutoTokenizer initialization).

    Raises ``ValueError`` if either token is not in the vocabulary.
    """
    from tokenizers import Tokenizer

    json_path = Path(tokenizer_path)
    if json_path.is_dir():
        json_path = json_path / "tokenizer.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"tokenizer.json not found at {json_path}")

    tok = Tokenizer.from_file(str(json_path))

    think_id = tok.token_to_id("<think>")
    end_think_id = tok.token_to_id("</think>")

    if think_id is None:
        raise ValueError(f"<think> not found in tokenizer at {json_path}")
    if end_think_id is None:
        raise ValueError(f"</think> not found in tokenizer at {json_path}")

    return think_id, end_think_id


def rewrite_dataset(
    input_prefix: str,
    output_prefix: str,
    transform: Callable[[np.ndarray], Optional[np.ndarray]],
) -> RewriteStats:
    """Read a merged dataset, apply *transform* per-sequence, write a new one.

    The output preserves the input's sequence order, dtype, and sequence modes
    (if present).  Sequences for which *transform* returns ``None`` are dropped.

    Raises ``FileExistsError`` if the output .bin or .idx already exists.
    """
    _ensure_megatron_importable()
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    from vision_tokenization.formats.megatron import IndexedDatasetBuilder

    multimodal = _index_has_sequence_modes(input_prefix)
    dataset = IndexedDataset(input_prefix, multimodal=multimodal)

    bin_path = output_prefix + ".bin"
    idx_path = output_prefix + ".idx"
    if Path(bin_path).exists() or Path(idx_path).exists():
        raise FileExistsError(
            f"Output already exists: {output_prefix}.bin/.idx — "
            f"remove manually or use a different output name"
        )

    builder = IndexedDatasetBuilder(
        bin_path, dtype=dataset.index.dtype, multimodal=multimodal,
    )

    n = len(dataset)
    written = 0
    skipped = 0
    output_tokens = 0
    log_interval = max(1, n // 10)
    for i in range(n):
        item = dataset[i]
        if multimodal:
            seq, mode = item
        else:
            seq = item
            mode = 0

        seq = transform(seq)
        if seq is not None and len(seq) > 0:
            builder.add_item(seq, mode=mode)
            builder.end_document()
            written += 1
            output_tokens += len(seq)
        else:
            skipped += 1

        if (i + 1) % log_interval == 0:
            logger.info(
                "Rewrite progress: %d/%d (%.0f%%) — %d written, %d skipped",
                i + 1, n, (i + 1) / n * 100, written, skipped,
            )

    builder.finalize(idx_path)

    return RewriteStats(
        input_count=n,
        written_count=written,
        skipped_count=skipped,
        output_tokens=output_tokens,
    )


def _find_shard_pairs(directory: Path) -> list[tuple[str, str]]:
    """All rank chunk .bin/.idx pairs in *directory* (flat — one stream per rank)."""
    pairs = []
    for f in sorted(directory.glob("rank_*_chunk_*.bin")):
        idx = f.with_suffix(".idx")
        if idx.exists() and f.stat().st_size > 0:
            pairs.append((str(f), str(idx)))
    return pairs


def _all_ranks_done(output_dir: Path, expected_ranks: int) -> bool:
    """Verify every expected rank has reached its terminal _SUCCESS marker.

    ``rank_NNNN/_SUCCESS`` is written LAST by each rank, after all shards
    have been atomically renamed from ``.tmp`` and stats are flushed. It is
    the Spark/Hadoop convention for a clean-finalize signal.

    The previous implementation checked for ``rank_NNNN_checkpoint.pt``,
    which is a periodic resume marker written every 2500 batches — it
    exists long before a rank is done, so the check could pass on
    in-progress jobs and silently produce a truncated merge.
    """
    for rank in range(expected_ranks):
        success = output_dir / f"rank_{rank:04d}" / "_SUCCESS"
        if not success.exists():
            return False
    return True


DEFAULT_BANDS = [8192, 16384, 32768, 65536, 131072, 262144]


def _band_name(edges, i):
    return f"{edges[i]//1024}k" if i < len(edges) else f"gt{edges[-1]//1024}k"


def _read_idx_lengths(prefix: str):
    import struct
    with open(prefix + ".idx", "rb") as fh:
        fh.seek(18)
        n, _ = struct.unpack("<QQ", fh.read(16))
        return np.frombuffer(fh.read(n * 4), dtype=np.int32).astype(np.int64)


def band_table(pairs, edges):
    """Per-band (sequences, tokens) from .idx headers only — no .bin reads."""
    counts = np.zeros(len(edges) + 1, dtype=np.int64)
    tokens = np.zeros(len(edges) + 1, dtype=np.int64)
    for bin_path, _ in pairs:
        lens = _read_idx_lengths(bin_path[:-4])
        b = np.searchsorted(edges, lens, side="left")
        counts += np.bincount(b, minlength=len(edges) + 1)
        tokens += np.bincount(b, weights=lens, minlength=len(edges) + 1).astype(np.int64)
    return counts, tokens


def split_bands(merged_prefix: str, edges) -> dict:
    """Write per-band .idx views over the merged .bin (idx-only; bytes shared)."""
    import struct
    idx_path = merged_prefix + ".idx"
    with open(idx_path, "rb") as fh:
        header = fh.read(18)
        n, n_doc = struct.unpack("<QQ", fh.read(16))
        lens = np.frombuffer(fh.read(n * 4), dtype=np.int32)
        ptrs = np.frombuffer(fh.read(n * 8), dtype=np.int64)
    bands = np.searchsorted(edges, lens.astype(np.int64), side="left")
    out = {}
    for i in range(len(edges) + 1):
        sel = np.where(bands == i)[0]
        if len(sel) == 0:
            continue
        path = f"{merged_prefix}_{_band_name(edges, i)}.idx"
        with open(path, "wb") as fh:
            fh.write(header)
            fh.write(struct.pack("<QQ", len(sel), len(sel) + 1))
            fh.write(lens[sel].tobytes())
            fh.write(ptrs[sel].tobytes())
            fh.write(np.arange(len(sel) + 1, dtype=np.int64).tobytes())
        out[_band_name(edges, i)] = (len(sel), int(lens[sel].astype(np.int64).sum()), path)
    return out


def merge_shards(
    output_dir: Path,
    output_name: str = "merged",
    *,
    shuffle: bool = False,
    seed: int = 42,
    bands: Optional[List[int]] = None,
    dry_run: bool = False,
) -> Optional[Path]:
    """Concatenate all rank shard pairs; optionally split per-band idx views.

    ``dry_run`` prints the gating + band table from .idx headers and writes nothing.
    """
    _ensure_megatron_importable()
    import shutil
    shutil.COPY_BUFSIZE = 64 << 20  # 64 KiB default cripples multi-100GB merges
    from megatron.core.datasets.indexed_dataset import (
        IndexedDataset, IndexedDatasetBuilder, get_bin_path, get_idx_path,
    )

    output_dir = Path(output_dir)
    pairs = _find_shard_pairs(output_dir)
    if not pairs:
        logger.warning("No shard pairs found in %s", output_dir)
        return None

    edges = bands or []
    total_bytes = sum(Path(b).stat().st_size for b, _ in pairs)
    if dry_run or edges:
        counts, tokens = band_table(pairs, edges or DEFAULT_BANDS)
        ed = edges or DEFAULT_BANDS
        print(f"shards: {len(pairs)} pairs, {total_bytes/2**30:.1f} GB -> {output_dir / output_name}.bin")
        print(f"{'band':>8} {'seqs':>12} {'tokens':>18}")
        for i in range(len(ed) + 1):
            if counts[i]:
                print(f"{_band_name(ed, i):>8} {counts[i]:>12,} {tokens[i]:>18,}")
        print(f"{'TOTAL':>8} {counts.sum():>12,} {tokens.sum():>18,}")
    if dry_run:
        return None

    import random
    prefixes = [bp.rsplit(".bin", 1)[0] for bp, _ in pairs]
    if shuffle:
        random.seed(seed)
        random.shuffle(prefixes)

    output_prefix = str(output_dir / output_name)
    logger.info("Merging %d shard pairs into %s", len(prefixes), output_prefix)
    builder = None
    for prefix in prefixes:
        if builder is None:
            dataset = IndexedDataset(prefix)
            builder = IndexedDatasetBuilder(get_bin_path(output_prefix), dtype=dataset.index.dtype)
            del dataset
        builder.add_index(prefix)
    builder.finalize(get_idx_path(output_prefix))

    if edges:
        for name, (n, tok, path) in split_bands(output_prefix, edges).items():
            logger.info("band %s: %s seqs, %s tokens -> %s", name, f"{n:,}", f"{tok:,}", path)

    logger.info("Merge complete: %s (%d shards)", output_prefix, len(prefixes))
    return Path(output_prefix)


def maybe_merge_shards(
    output_dir: Path | str,
    *,
    expected_ranks: int,
    output_name: str = "merged",
    shuffle: bool = False,
    seed: int = 42,
) -> Optional[Path]:
    """Merge shards if all ranks are done. Returns None if not ready or merge disabled."""
    output_dir = Path(output_dir)

    # Check if already merged
    merged_bin = output_dir / f"{output_name}.bin"
    if merged_bin.exists():
        logger.debug("Merged file already exists: %s", merged_bin)
        return Path(output_dir / output_name)

    if not _all_ranks_done(output_dir, expected_ranks):
        return None

    try:
        return merge_shards(
            output_dir,
            output_name=output_name,
            shuffle=shuffle,
            seed=seed,
        )
    except Exception:
        logger.warning("Failed to merge shards in %s", output_dir, exc_info=True)
        return None


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Directory containing rank shard files")
    parser.add_argument(
        "--expected-ranks", type=int, default=None,
        help="Wait for this many rank checkpoints before merging",
    )
    parser.add_argument("--output-name", default="merged", help="Output prefix name")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bands", type=lambda v: [int(x) for x in v.split(",")],
                        default=None, help="Band edges, e.g. 8192,16384,32768")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print gating + per-band table; write nothing")
    parser.add_argument(
        "--strip-thinking", action="store_true",
        help="After merging, produce a second no-CoT variant with "
        "<think>...</think> spans removed",
    )
    parser.add_argument("--no-cot-output-name", default="merged_no_cot")
    parser.add_argument(
        "--think-id", type=int, default=32,
        help="<think> token ID (default: 32)",
    )
    parser.add_argument(
        "--end-think-id", type=int, default=33,
        help="</think> token ID (default: 33)",
    )
    parser.add_argument(
        "--resolve-thinking-ids", action="store_true",
        help="Resolve think/end-think IDs from --tokenizer-path instead of using defaults",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/"
        "apertus_emu3.5_wavtok_instruct",
        help="Tokenizer path for --resolve-thinking-ids",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Resolve thinking token IDs early so we fail before merge, not after
    think_id, end_think_id = args.think_id, args.end_think_id
    if args.strip_thinking and args.resolve_thinking_ids:
        think_id, end_think_id = resolve_thinking_token_ids(args.tokenizer_path)
        logger.info(
            "Resolved thinking IDs from %s: think=%d, end_think=%d",
            args.tokenizer_path, think_id, end_think_id,
        )

    output_dir = Path(args.output_dir)
    if args.expected_ranks is None:
        args.expected_ranks = len([p for p in output_dir.glob("rank_*") if p.is_dir()])
        print(f"gating on {args.expected_ranks} rank dirs found")
    if not _all_ranks_done(output_dir, args.expected_ranks):
        missing = [
            r for r in range(args.expected_ranks)
            if not (output_dir / f"rank_{r:04d}" / "_SUCCESS").exists()
        ]
        print(
            f"Not all {args.expected_ranks} ranks have finished yet: "
            f"missing _SUCCESS for ranks {missing}"
        )
        return 1

    result = merge_shards(
        output_dir,
        output_name=args.output_name,
        shuffle=args.shuffle,
        seed=args.seed,
        bands=args.bands,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return 0
    if result is None:
        print("No shards found to merge.")
        return 1
    print(f"Merged to {result}.bin / {result}.idx")

    if args.strip_thinking:
        from functools import partial

        transform = partial(
            strip_thinking_tokens,
            think_id=think_id,
            end_think_id=end_think_id,
        )
        no_cot_prefix = str(output_dir / args.no_cot_output_name)
        logger.info(
            "Rewriting %s → %s (stripping think_id=%d, end_think_id=%d)",
            result, no_cot_prefix, think_id, end_think_id,
        )
        stats = rewrite_dataset(str(result), no_cot_prefix, transform)
        logger.info(
            "Rewrite complete: %d input → %d written (%d tokens), %d skipped",
            stats.input_count, stats.written_count, stats.output_tokens,
            stats.skipped_count,
        )
        print(
            f"No-CoT variant: {no_cot_prefix}.bin / {no_cot_prefix}.idx "
            f"({stats.written_count} sequences, {stats.output_tokens:,} tokens, "
            f"{stats.skipped_count} skipped)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
