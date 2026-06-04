# Provenance: mapping tokenized output back to source rows

Tokenization writes Megatron `.bin/.idx` micro-shards whose sequence order is **not**
the source order — the planner sorts samples by image resolution for GPU-batch
efficiency, and data-dependent skips (null text, undecodable images, invalid SFT
conversations) drop samples that the plan cannot predict. The **provenance** feature
records, per output sequence, the **source manifest row** it came from, so you can map
any output position back to the original dataset row — or join two tokenizations of the
same images on their shared source.

It is **opt-in** and **off by default**: with `emit_provenance: false` the output is
byte-identical to a run without the feature and no extra files are written.

## Primary use case

Tokenize the same image dataset twice with two different text columns (vanilla
`image2text` or `sft`), then group the two outputs by source sample:

```
run A (text column 1) ─┐
                       ├─►  group_map.parquet  (source_id, pos_in_A, pos_in_B)
run B (text column 2) ─┘
```

Because the plan is image-driven, both runs are maximally comparable; the join is robust
even when the two columns trigger different skips.

## 1. Enable during tokenization

Set the Hydra flag (config: `vision_tokenization/configs/config.yaml`, key
`emit_provenance`):

```bash
python -m vision_tokenization.tokenize \
    mode=image2text dataset=<your_dataset> \
    num_gpus=$NUM_GPUS emit_provenance=true
```

Each micro-shard gains a sibling sidecar `rank_XXXX_chunk_YYYY.src.npy` (int64, one
source manifest row per output sequence). It is written inside the shard's atomic
`tmp → fsync → rename` transaction, so a crash never leaves a shard paired with a stale
sidecar.

> **Enable from the start of a run.** Provenance must be on for every chunk. Resuming a
> run that was started with the flag off leaves earlier chunks without sidecars, and
> merge will refuse to emit a (misaligned) `merged.src.npy`.

Works for both backends: the direct path (`image_only`, `image2text`, `text2image`) and
the spill + offline-rebuild path (`sft`, `interleave`, multi-image). For SFT the source
id is **per conversation/document** (its first manifest row).

## 2. Merge → sidecar + typed Parquet

The standalone merge concatenates the per-shard sidecars (in the same order as the
`.bin/.idx` merge, shuffle-aware) into `merged.src.npy`, and — when given the manifest —
resolves it to a typed `merged.provenance.parquet`:

```bash
python -m vision_tokenization.pipeline.output.merge <output_dir> \
    --manifest /path/to/manifest.parquet
```

- `merged.src.npy` is emitted automatically when every shard has a sidecar (use
  `--emit-provenance` to **require** them and fail loudly if any are missing).
- `--manifest` must be the **same manifest used to build the plan**.
- `--strip-thinking` additionally produces `merged_no_cot.src.npy`, dropped in lockstep
  with the stripped sequences, so it aligns with both `merged` and `merged_no_cot`.

You can also run resolution standalone on an already-merged dataset:

```bash
python -m vision_tokenization.pipeline.output.provenance resolve \
    <output_dir>/merged --manifest /path/to/manifest.parquet
```

## 3. Group two runs

```bash
python -m vision_tokenization.pipeline.output.provenance group \
    <runA_dir>/merged.provenance.parquet \
    <runB_dir>/merged.provenance.parquet \
    --out group_map.parquet
```

`group_map.parquet` has one row per source sample present in **both** runs.

## Output files

| File | Schema | Meaning |
|---|---|---|
| `rank_XXXX_chunk_YYYY.src.npy` | int64 array | per-shard: `[i]` = source manifest row of output seq `i` |
| `merged.src.npy` | int64 array | merged: `[i]` = source manifest row of merged output seq `i` |
| `merged.provenance.parquet` | `output_index:int64, manifest_row:int64, source_id:str\|int` | typed, portable; `source_id` is `sample_key` (WebDataset) or `sample_index` (HuggingFace) |
| `merged_no_cot.src.npy` | int64 array | provenance for the `--strip-thinking` variant |
| `group_map.parquet` | `source_id:str\|int, pos_in_A:int64, pos_in_B:int64` | output positions of each shared source sample in two runs |

## Fast lookups in code

The merged sidecar stores the **contiguous manifest row** (an int), which is identical
across runs of the same dataset. That makes lookups an O(1) array inverse — no dict
needed, and far leaner than a dict at hundreds of millions of rows:

```python
import numpy as np
from vision_tokenization.pipeline.output.provenance import load_source_ids

srcA = load_source_ids("runA/merged.src.npy")    # output_pos -> manifest_row
srcB = load_source_ids("runB/merged.src.npy")

N = int(max(srcA.max(), srcB.max())) + 1
posA = np.full(N, -1, np.int64); posA[srcA] = np.arange(len(srcA))   # row -> output_pos, O(1)
posB = np.full(N, -1, np.int64); posB[srcB] = np.arange(len(srcB))

out_for_row_X = posA[X]                           # -1 if row X was skipped in run A
shared = np.nonzero((posA >= 0) & (posB >= 0))[0] # manifest rows present in both runs
# group_map equivalent: zip(shared, posA[shared], posB[shared])
```

To recover the original dataset identity from a manifest row, index the manifest's
`sample_key`/`sample_index` column (this is exactly what `resolve` does):

```python
from vision_tokenization.indexing.manifest import load_manifest
keys = load_manifest("manifest.parquet", columns=["sample_index"]).column("sample_index").to_numpy()
original_ids = keys[src]
```

## Notes and limitations

- **Manifest row vs. original id.** Sidecars store the contiguous manifest row (zero
  hot-loop cost, stable across runs). The human-facing `source_id` (string for WDS, int
  for HF) is resolved once at merge time from the manifest — never in the tokenize loop.
- **`build_group_map` assumes one output sequence per source row** (true for
  `image2text` and `sft`). It raises on duplicate rows, which `interleave` can produce by
  splitting one document into several sequences — handle that case separately.
- **SFT keys are per conversation/document**, not per image row; a `merged.provenance.parquet`
  from an SFT run joins against another run at the document level.
- The feature adds no work and no files when `emit_provenance` is off.
