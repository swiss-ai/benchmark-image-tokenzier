"""Merge per-rank tokenized shards into a single dataset.

Run standalone after all ranks finish::

    python -m vision_tokenization.pipeline.output.merge /path/to/output_dir \
        [--bands 8192,16384,...] [--dry-run]

Gating verifies rank completion manifests (rank_NNNN_DONE.json): a dataset
merges iff every rank's claim verifies against disk. Pre-manifest run dirs
are not mergeable — re-tokenize with current code.
"""

from __future__ import annotations

import argparse
import logging
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

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
    """Numba-compiled one-pass state machine.

    Returns the write count and whether the sequence ended mid-span.
    """
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
    return w, inside


def strip_thinking_tokens(
    tokens: np.ndarray,
    think_id: int,
    end_think_id: int,
) -> Tuple[Optional[np.ndarray], bool]:
    """Remove reasoning spans from a token sequence.

    One-pass two-state delimiter machine (numba-compiled):

    - **outside**: keep tokens; ``think_id`` enters *inside*, ``end_think_id``
      (orphan close) is dropped.
    - **inside**: drop everything including repeated ``think_id``;
      ``end_think_id`` exits back to *outside*.

    Returns the stripped sequence (``None`` if empty afterwards),
    and whether the sequence ended still inside a span.
    An unclosed span drops everything after the opener,
    which the output cannot be distinguished from a correct strip.
    Hence the flag rather than silence.
    """
    n = len(tokens)
    if n == 0:
        return None, False
    if not ((tokens == think_id) | (tokens == end_think_id)).any():
        # fast path — no copy
        return tokens, False

    out = np.empty(n, dtype=tokens.dtype)
    w, unclosed = _strip_thinking_inner(tokens, think_id, end_think_id, out)

    return (out[:w] if w > 0 else None), unclosed


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


def resolve_reasoning_delimiters(tokenizer_dir: str) -> Tuple[int, int]:
    """Delimiter ids for *tokenizer_dir*, by encoding the strings a chat template writes.

    Encoding rather than looking the names up is what makes this correct across tokenizer
    revisions: apertus_emu3.5_wavtok_instruct_thinking_token_fixed normalizes ``<think>``
    into ``<|inner_prefix|>``, so a name lookup returns 69 while the id in its data is 32.
    """
    from tokenizers import Tokenizer

    path = Path(tokenizer_dir)
    if path.is_dir():
        path = path / "tokenizer.json"
    tok = Tokenizer.from_file(str(path))

    ids = []
    for text in ("<think>", "</think>"):
        encoded = tok.encode(text, add_special_tokens=False).ids
        if len(encoded) != 1:
            raise ValueError(
                f"{text!r} is not a single token in {path} (encodes to {encoded}). "
                f"Pass --think-id/--end-think-id explicitly."
            )
        ids.append(encoded[0])
    return ids[0], ids[1]


def strip_thinking_dataset(
    input_prefix: str,
    output_prefix: str,
    think_id: int,
    end_think_id: int,
) -> RewriteStats:
    """Write *input_prefix* to *output_prefix* with reasoning spans removed."""
    unclosed = 0

    def transform(seq):
        nonlocal unclosed
        stripped, ended_inside = strip_thinking_tokens(seq, think_id, end_think_id)
        unclosed += ended_inside
        return stripped

    stats = rewrite_dataset(input_prefix, output_prefix, transform)
    if unclosed:
        logger.warning(
            "%d of %d sequences ended inside an unclosed reasoning span — everything "
            "after the opener was dropped. Expected for truncated generations; if it is "
            "most of the dataset, think_id=%d/end_think_id=%d belong to another tokenizer.",
            unclosed, stats.input_count, think_id, end_think_id,
        )
    return stats


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


def _run_tokenizer_path(manifests: list) -> Optional[str]:
    """The tokenizer this run was written with, if every rank agrees on it."""
    paths = {m.get("tokenizer_path") for m in manifests}
    paths.discard(None)
    return paths.pop() if len(paths) == 1 else None


def verify_manifests(output_dir: Path, manifests: list) -> tuple:
    """Verify rank completion claims against disk. Returns (problems, totals).

    A dataset merges iff: manifests form ranks 0..N-1 with unanimous
    world_size and plan fingerprint, and the union of claimed shard files
    matches the rank shards on disk exactly — both directions, with sizes.
    """
    problems = []
    sizes = {m["world_size"] for m in manifests}
    if len(sizes) > 1:
        problems.append(f"manifests disagree on world_size {sorted(sizes)} — mixed runs")
        return problems, None
    world_size = sizes.pop()
    ranks = [m["rank"] for m in manifests]
    missing = sorted(set(range(world_size)) - set(ranks))
    if missing:
        problems.append(f"no completion manifest for ranks {missing} — run incomplete or crashed")
    extra = sorted(set(ranks) - set(range(world_size)))
    if extra:
        problems.append(f"manifests for ranks {extra} exceed world_size={world_size} — stale leftovers")
    plans = {json.dumps(m.get("plan"), sort_keys=True) for m in manifests}
    if len(plans) > 1:
        problems.append("manifests carry different plan fingerprints — mixed runs in one directory")

    claimed = {f["name"]: f["bytes"] for m in manifests for f in m["files"]}
    on_disk = {p.name: p.stat().st_size for p in output_dir.glob("rank_*_chunk_*.bin")}
    for name, nbytes in sorted(claimed.items()):
        if name not in on_disk:
            problems.append(f"claimed shard missing on disk: {name}")
        elif on_disk[name] != nbytes:
            problems.append(f"size mismatch for {name}: manifest {nbytes:,} B, disk {on_disk[name]:,} B")
        elif not (output_dir / name).with_suffix(".idx").exists():
            problems.append(f"claimed shard has no .idx: {name}")
    stale = sorted(set(on_disk) - set(claimed))
    if stale:
        problems.append(f"shards on disk not claimed by any manifest (stale leftovers?): {stale}")

    totals = {
        "ranks": world_size,
        "files": len(claimed),
        "sequences": sum(m["sequences"] for m in manifests),
        "tokens": sum(m["tokens"] for m in manifests),
    }
    return problems, totals


def _band_name(edges, i):
    return f"{edges[i]//1024}k" if i < len(edges) else f"gt{edges[-1]//1024}k"


def _validate_edges(edges) -> list:
    """np.searchsorted needs ascending edges; KiB-floored names must be unique
    or one band's view file silently overwrites another's."""
    edges = [int(e) for e in edges]
    if any(b <= a for a, b in zip(edges, edges[1:])) or edges[0] <= 0:
        raise ValueError(f"band edges must be positive and strictly ascending, got {edges}")
    names = [_band_name(edges, i) for i in range(len(edges))]
    if len(set(names)) != len(names):
        raise ValueError(f"band edges collide on names {names} — keep edges >= 1 KiB apart")
    return edges


def _band_of(lengths, edges):
    return np.searchsorted(edges, lengths.astype(np.int64), side="left")


def band_table(pairs, edges):
    """Per-band (sequences, tokens) from .idx headers only — no .bin reads."""
    from vision_tokenization.formats.megatron import read_idx

    edges = _validate_edges(edges)
    counts = np.zeros(len(edges) + 1, dtype=np.int64)
    tokens = np.zeros(len(edges) + 1, dtype=np.int64)
    for bin_path, _ in pairs:
        _, lens, _, _ = read_idx(bin_path[:-4])
        b = _band_of(lens, edges)
        counts += np.bincount(b, minlength=len(edges) + 1)
        tokens += np.bincount(b, weights=lens, minlength=len(edges) + 1).astype(np.int64)
    return counts, tokens


def split_bands(merged_prefix: str, edges) -> dict:
    """Write per-band .idx views over the merged .bin (idx-only; bytes shared)."""
    from vision_tokenization.formats.megatron import read_idx, write_idx_view

    edges = _validate_edges(edges)
    header, lens, ptrs, _ = read_idx(merged_prefix)
    bands = _band_of(lens, edges)
    out = {}
    for i in range(len(edges) + 1):
        sel = np.where(bands == i)[0]
        if len(sel) == 0:
            continue
        name = _band_name(edges, i)
        path = f"{merged_prefix}_{name}.idx"
        write_idx_view(path, header, lens[sel], ptrs[sel])
        # Megatron derives <prefix>.bin from the idx prefix: alias the shared
        # bin per view (hardlink: same inode, zero bytes).
        alias = Path(f"{merged_prefix}_{name}.bin")
        alias.unlink(missing_ok=True)
        os.link(merged_prefix + ".bin", alias)
        out[name] = (len(sel), int(lens[sel].astype(np.int64).sum()), path)
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
    output_prefix = str(output_dir / output_name)
    pairs = _find_shard_pairs(output_dir)
    if not pairs:
        logger.warning("No shard pairs found in %s", output_dir)
        return None

    if dry_run:
        edges = bands or DEFAULT_BANDS
        counts, tokens = band_table(pairs, edges)
        total_bytes = sum(Path(b).stat().st_size for b, _ in pairs)
        print(f"shards: {len(pairs)} pairs, {total_bytes/2**30:.1f} GB -> {output_prefix}.bin")
        print(f"{'band':>8} {'seqs':>12} {'tokens':>18}")
        for i in range(len(edges) + 1):
            if counts[i]:
                print(f"{_band_name(edges, i):>8} {counts[i]:>12,} {tokens[i]:>18,}")
        print(f"{'TOTAL':>8} {counts.sum():>12,} {tokens.sum():>18,}")
        return None

    if Path(output_prefix + ".bin").exists():
        logger.info("Merged file already exists: %s.bin — skipping to band views", output_prefix)
    else:
        import random
        prefixes = [bp.rsplit(".bin", 1)[0] for bp, _ in pairs]
        if shuffle:
            random.seed(seed)
            random.shuffle(prefixes)

        logger.info("Merging %d shard pairs into %s", len(prefixes), output_prefix)
        builder = None
        for prefix in prefixes:
            if builder is None:
                dataset = IndexedDataset(prefix)
                builder = IndexedDatasetBuilder(get_bin_path(output_prefix), dtype=dataset.index.dtype)
                del dataset
            builder.add_index(prefix)
        builder.finalize(get_idx_path(output_prefix))

    if bands:
        for name, (n, tok, path) in split_bands(output_prefix, bands).items():
            logger.info("band %s: %s seqs, %s tokens -> %s", name, f"{n:,}", f"{tok:,}", path)

    logger.info("Merge complete: %s (%d shards)", output_prefix, len(pairs))
    return Path(output_prefix)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Directory containing rank shard files")
    parser.add_argument("--output-name", default="merged", help="Output prefix name")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bands", type=lambda v: _validate_edges(v.split(",")),
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
        "--think-id", type=int, default=None,
        help="Override the reasoning-span opening token ID; by default it is "
             "resolved from the tokenizer this run recorded",
    )
    parser.add_argument(
        "--end-think-id", type=int, default=None,
        help="Override the reasoning-span closing token ID",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(args.output_dir)
    from ..runtime.checkpoint import load_rank_manifests
    manifests = load_rank_manifests(output_dir)
    if manifests:
        problems, totals = verify_manifests(output_dir, manifests)
        if problems:
            print("REFUSING to merge — completion manifests do not verify:")
            for prob in problems:
                print(f"  - {prob}")
            return 1
        print(
            f"manifest gate: {totals['ranks']} ranks verified — {totals['files']} shards, "
            f"{totals['sequences']:,} sequences, {totals['tokens']:,} tokens"
        )
    else:
        print(
            "No completion manifests found — this directory predates the manifest "
            "protocol (or the run never finished). Re-tokenize with current code."
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
        think_id, end_think_id = args.think_id, args.end_think_id
        if think_id is None or end_think_id is None:
            tokenizer_path = _run_tokenizer_path(manifests)
            if tokenizer_path is None:
                raise SystemExit(
                    "--strip-thinking needs the delimiter ids. This run's manifests do "
                    "not record a tokenizer_path (they predate it), so pass --think-id "
                    "and --end-think-id, resolved from the tokenizer that produced it."
                )
            think_id, end_think_id = resolve_reasoning_delimiters(tokenizer_path)
            logger.info("Resolved delimiters from %s: think=%d end_think=%d",
                        tokenizer_path, think_id, end_think_id)

        no_cot_prefix = str(output_dir / args.no_cot_output_name)
        logger.info(
            "Rewriting %s → %s (stripping think_id=%d, end_think_id=%d)",
            result, no_cot_prefix, think_id, end_think_id,
        )
        stats = strip_thinking_dataset(str(result), no_cot_prefix,
                                       think_id, end_think_id)
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
