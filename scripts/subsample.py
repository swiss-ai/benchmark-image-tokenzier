"""Subsample each dataset in --src to ~RATIO of its TOKENS into --dest, writing
ONLY the kept subset (no discard half). Use this instead of separate.py when
carving a small fraction from a large pool — separate.py writes BOTH ratio-splits
to temp (the discarded majority dominates runtime for long-context bands).

Per dataset: shuffle documents (per-dataset seed from master rng), accumulate doc
tokens until >= ratio*dataset_tokens, write those docs via IndexedDatasetBuilder
.add_document (same primitive separate.py uses — plain, no multimodal modes, matching
the other LC bands made by separate.py). Output: dest/<dataset>.{bin,idx}.

Usage:
  python scripts/subsample.py --src <pool> --dest <out> --ratio 0.04305 --seed 42 --workers 9
"""
import argparse, glob, os, sys
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, "/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM")
from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder


def _sample(t):
    prefix_in, prefix_out, ratio, seed = t
    ds = IndexedDataset(prefix_in)
    doc_idx = ds.document_indices
    seqlens = np.asarray(ds.sequence_lengths)
    dtype = ds.index.dtype
    n_docs = len(doc_idx) - 1
    total = int(seqlens.sum())
    doc_tokens = np.add.reduceat(seqlens, doc_idx[:-1].astype(int))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_docs)
    cum = np.cumsum(doc_tokens[perm].astype(np.int64))
    k = min(int(np.searchsorted(cum, ratio * total)) + 1, n_docs)
    keep = np.sort(perm[:k])
    os.makedirs(os.path.dirname(prefix_out), exist_ok=True)
    b = IndexedDatasetBuilder(prefix_out + ".bin", dtype=dtype)
    for d in keep:
        s, e = int(doc_idx[d]), int(doc_idx[d + 1])
        seqs = ds[s:e]
        b.add_document(np.concatenate(seqs), [len(x) for x in seqs])
    b.finalize(prefix_out + ".idx")
    del ds
    return os.path.basename(prefix_out), int(k), int(doc_tokens[keep].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.dest, exist_ok=True)
    master = np.random.default_rng(a.seed)
    tasks = []
    for idx in sorted(glob.glob(a.src + "/*.idx")):
        ds = os.path.basename(idx)[:-4]
        if os.path.exists(f"{a.dest}/{ds}.idx"):
            print(f"  SKIP {ds} (exists)", flush=True); continue
        tasks.append((idx[:-4], f"{a.dest}/{ds}", a.ratio, int(master.integers(0, 2**63))))
    print(f"sampling ratio={a.ratio} from {len(tasks)} datasets -> {a.dest}", flush=True)
    tot = 0
    with Pool(min(a.workers, max(1, len(tasks)))) as p:
        for name, k, t in p.imap_unordered(_sample, tasks):
            tot += t
            print(f"  {name}: {k:,} docs, {t:,} tok", flush=True)
    print(f"TOTAL kept: {tot:,} tok ({tot/1e9:.2f}B)", flush=True)


if __name__ == "__main__":
    main()
