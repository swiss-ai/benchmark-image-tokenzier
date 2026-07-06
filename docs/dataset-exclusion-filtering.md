# Dataset Filtering & Exclusion

Concise but complete reference for excluding specific images from tokenization and
for the other filtering knobs. Aimed at the data team producing exclusion lists and
at engineers wiring up datasets.

---

## 1. The pipeline and where filtering happens

Tokenization is three stages (`CLAUDE.md` → Architecture). Each filter acts at a
specific stage:

| Stage | What it does | Filters that act here |
|-------|--------------|-----------------------|
| **1. Index / scan** (build `manifest.parquet`) | Scanners read shard metadata (never pixels) into a manifest | **Exclusion-ID list** (rows never enter the manifest); automatic validation skips |
| **2. Plan** (`TokenizationPlan`) | Compute batches & rank splits from the manifest | **Pixel-size filter** (`min_pixels`/`max_pixels`, min-dimension floor) |
| **3. Tokenize** | GPU encode → Megatron `.bin/.idx` | none (exclusion is upstream) |

Everything downstream consumes the manifest, so **dropping a row from the manifest
== that image is never tokenized**.

---

## 2. Exclusion-ID lists (the main feature)

Exclude specific source rows by handing the pipeline a plain-text **exclusion-ID
list**. Supported for **HuggingFace** (`.parquet` / `.arrow`) and **WebDataset**
(`.tar`) datasets — see [§6 Limitations](#6-limitations).

A list is tokens separated by commas or whitespace. Pick the format for your storage:
- HF: `stem_row` (general) or `innovator_vl` (special case)
- WebDataset: `wds_key`

> **Validate your list.** Tokens that don't parse for the chosen format (missing `:`,
> non-numeric row for `stem_row`, not `SFT_NNNNNN_NNNNNN` for `innovator_vl`, empty key
> for `wds_key`) are **silently skipped** — they exclude nothing and raise no warning
> (they never become a key, so they don't even show up as "unmatched"). The pipeline
> only errors if *zero* tokens parse. A typo therefore silently under-excludes.

### 2a. `stem_row` (general — recommended)

Token = **`FILE:ROW`** (split on the last `:`).

- **`FILE`** = the shard filename **stem** — the `.parquet`/`.arrow` extension is
  dropped if you include it. Optionally prefix directory parts (`subset/stem`) to
  disambiguate identical stems across subsets/configs.
- **`ROW`** = **0-based row index within that one shard file**, counted across all
  row-groups (parquet) / record batches (arrow). *Not* a global dataset index and
  *not* within a single row-group.

For a multi-image sample, one ID drops the **whole sample/group**.

**How matching works:** for each shard, the pipeline builds candidate keys — the
stem, then progressively longer path suffixes (`stem`, `dir/stem`,
`dir2/dir/stem`, …) — and a list key matches if it equals any of them. So a bare
`stem:row` matches by stem, and `subset/.../stem:row` pins down one subset.

**Example** — files:
```
<root>/mydataset/train-00000-of-00003.parquet
<root>/mydataset/train-00001-of-00003.parquet
<root>/mydataset/train-00002-of-00003.parquet
```
drop rows 5 & 9 of shard 0, row 0 of shard 1, row 120 of shard 2:
```
train-00000-of-00003:5
train-00000-of-00003:9
train-00001-of-00003:0
train-00002-of-00003:120
```

### 2b. `innovator_vl` (special case, default)

Fixed-width tokens **`SFT_<SHARD6>_<ROW6>`**, keyed on file stem `SFT_<SHARD6>`.
Only works when shards are named `SFT_<6 digits>.{parquet,arrow}`.
```
SFT_000099_000001, SFT_000099_000003     # rows 1 and 3 of SFT_000099.parquet
```
`stem_row` can express the same thing (`SFT_000099:1`); prefer it for anything not
already on the `SFT_NNNNNN` convention.

### 2c. `wds_key` (WebDataset)

Token = **`TAR:SAMPLE_KEY`** (split on the last `:`).

- **`TAR`** = the tar shard filename **stem** (`.tar` dropped if you include it).
  Optionally prefix directory parts (`subset/shard_000`) to disambiguate identical
  stems across subdirs — matched exactly like `stem_row`'s file part (stem, then
  progressively longer path suffixes). Easiest rule: the tar's path **relative to
  the `input_pattern` root**, with the extension dropped.
- **`SAMPLE_KEY`** = the sample's WebDataset `__key__`. In a WebDataset tar the
  members of one sample share a basename and differ only by extension —
  `000123.jpg` + `000123.txt` ⇒ key `000123`. So `SAMPLE_KEY` is the tar-member
  filename with its **file extension removed** (only the last `.` segment); a key
  may itself contain dots (`foo.bar.jpg` ⇒ `foo.bar`). For multi-image samples
  scanned with `image_field_pattern="img"`, members look like `000123.img0.jpg`,
  `000123.img1.jpg` and the trailing `.imgN` field is also stripped ⇒ key `000123`.
  `SAMPLE_KEY` is **tar-local** (the same key can recur in other tars), so there is
  **no bare-key form** — you must name the tar.

A multi-image sample (all its images share one `SAMPLE_KEY`) is dropped whole.

**Example** — two tars under the input root; drop two samples from one and one from
the other:
```
shard_000:000123   shard_000:004567
shard_001:000042
```

**For a usable WDS list, give us:** one `TAR:SAMPLE_KEY` token per sample to drop,
where `TAR` is the shard path relative to the input root (no `.tar`) and
`SAMPLE_KEY` is the sample basename (no extension, no `.imgN` field suffix), comma-
or whitespace-separated. The pipeline **aborts** if any `TAR` matches more than one
shard (path-qualify it) and **warns** if a `TAR` matches no shard.

---

## 3. Multiple subsets / configs (shared stems)

HF datasets with several configs/splits reuse shard filenames, so a bare stem
collides. Typical hub layout `<config>/<split>/<shard>.parquet`:
```
<root>/en/train/0000.parquet
<root>/fr/train/0000.parquet
```

| ID | Effect |
|----|--------|
| `en/train/0000:5` | ✅ exactly `en/train/0000.parquet` |
| `train/0000:5`    | ⛔ **rejected** — matches every config's `train/0000` (ambiguous) |
| `en/0000:5`       | ❌ matches nothing — the `train/` split dir is skipped (suffix must be **contiguous**) |
| `0000:5`          | ⛔ **rejected** — matches every `0000` shard (ambiguous) |

**Validation (fail-fast):** at scan time and in the manifest filter, the pipeline
checks the list against the actual input shards and **raises** if any ID resolves
to more than one shard (listing the offending keys), instead of silently
over-excluding. IDs matching **no** shard are warned about, not fatal (a list may
cover files outside one run).

**Safest convention for the data team:** emit each ID as the shard's path
**relative to the directory passed as `input_pattern`**, drop the extension, append
`:<row>` (e.g. `en/train/0000:5`). Always unique, always a valid contiguous suffix,
regardless of nesting depth.

The same rules apply to WebDataset `wds_key` ids: the `TAR` part is matched like a
shard stem, so tar filenames repeated across subdirs are disambiguated by
path-qualifying (`subset/shard_000:KEY`) and ambiguous tar names are rejected.

---

## 4. Applying a list — 3 entry points & required data

| Entry point | When | Input data it needs | Output |
|-------------|------|---------------------|--------|
| **Scan-time** (recommended) | building the manifest | HF `.parquet`/`.arrow` **or** WDS `.tar` shards + ID list | manifest with rows already dropped |
| **`decontaminate_manifest.py`** | after the manifest exists, before tokenizing | an HF manifest (`shard_path`,`chunk_index`,`row_in_chunk`) **or** WDS manifest (`sample_key`,`tar_path`) + ID list | a filtered manifest (point the config at it) |
| **`rebuild_decontaminated.py`** (HF only) | after tokenizing, no re-encode | the preserved per-rank **spill** dir + manifest + tokenizer + ID list | clean `.bin/.idx` with contaminated docs dropped |

```bash
# 1. scan-time — HF
scan_hf_dataset(..., contamination_ids_path="ids.txt", contamination_format="stem_row")
#    scan-time — WebDataset
scan_wds_dataset(..., contamination_ids_path="ids.txt", contamination_format="wds_key")

# 2. post-hoc manifest filter (HF or WDS). --contamination-format defaults to
#    innovator_vl, so pass it explicitly for stem_row / wds_key (--format is an alias).
python scripts/decontaminate_manifest.py \
    --input-manifest manifest.parquet --output-manifest manifest_clean.parquet \
    --contamination-ids ids.txt --contamination-format stem_row

# 3a. post-tokenization rebuild — PREPARE on a head node. Builds & caches plan.pt +
#     reject_doc_ids.json under --output-dir, and runs the guardrail. Pass the SAME
#     plan args as the ORIGINAL tokenize run (see §6): the document-filter args
#     (--mode/--min-pixels/--max-pixels/--max-images-per-doc/--spatial-factor) are
#     enforced by the guardrail (abort on mismatch); copy the remaining plan args
#     (--text-column/--tokenizer-min-pixels/--tokenizer-max-pixels/--window-size/
#     --batch-size/--max-batch-tokens) verbatim too — a mismatch there is NOT detected.
python scripts/rebuild_decontaminated.py \
    --manifest manifest.parquet --contamination-ids ids.txt --contamination-format stem_row \
    --output-dir <tokenized_dir>_decontaminated --tokenizer-path <tok> \
    --mode sft --min-pixels "2048*2048" --max-pixels "2048*2048" --max-images-per-doc 8 \
    --prepare-only

# 3b. post-tokenization rebuild — one slurm array task per rank (0..EXPECTED_RANKS-1),
#     reusing the cached plan.pt / reject_doc_ids.json. Pass --rank + --spill-dir and the
#     SAME plan args as 3a; drop --prepare-only.
python scripts/rebuild_decontaminated.py \
    --manifest manifest.parquet --contamination-ids ids.txt --contamination-format stem_row \
    --spill-dir <tokenized_dir> --output-dir <tokenized_dir>_decontaminated \
    --tokenizer-path <tok> --rank $SLURM_ARRAY_TASK_ID \
    --mode sft --min-pixels "2048*2048" --max-pixels "2048*2048" --max-images-per-doc 8
```

**Auditing:** each drop is verifiable. Scan-time records `contaminated_skipped` (number
of excluded samples) in the manifest metadata; `decontaminate_manifest.py` writes
`manifest_rows_dropped` / `source_docs_dropped` / `unmatched_contamination_ids` (+ the
format and ID path) to a `*_decontamination_meta.json` next to the output manifest.
`rebuild_decontaminated.py` does not write a manifest sidecar — it logs rebuild stats and
writes `reject_doc_ids.json` under `--output-dir` (its length = number of dropped docs).

---

## 5. Other filtering (no list needed)

- **Pixel-size filter** (Stage 2) — `min_pixels` / `max_pixels` in the dataset
  config drop images outside a pixel-count band; a min-dimension floor
  (`spatial_factor`, default 16 px) and failed dimension reads also drop images.
  Tokenizer-side `resize_min_pixels` / `resize_max_pixels` bound the resize.
- **Automatic validation skips** (Stage 1) — rows with missing/corrupt image
  headers or missing columns are skipped (reported as `failed_dims` /
  `skipped_shards` in the manifest metadata).
- **Curated input directory** — point `input_pattern` at a symlink-curated subset
  of shards to exclude whole splits (e.g. an eval split) with no ID list.
- **Content dedup** (alignment mode only) — exact-byte dedup via `media_sha256`;
  automatic and content-based, not list-based.

---

## 6. Limitations

- **HF + WebDataset supported; jsonl-tar not.** Exclusion needs identifying manifest
  columns: HF (`shard_path`,`chunk_index`,`row_in_chunk`) or WDS
  (`sample_key`,`tar_path`). jsonl-tar manifests carry `image_ref`/`tar_path` but no
  exclusion format is wired for them yet.
- **Three ID formats** (`stem_row`, `innovator_vl`, `wds_key`). A content key
  (`sha256`/URL) would need a new parser branch in
  `vision_tokenization/utils/contamination.py` plus the scanner recording that column.
- **`rebuild_decontaminated.py` is HF-only** and its doc_id mapping is position-based,
  assuming no group was dropped by the Stage-2 pixel/min-dimension filter; it
  **aborts** if that assumption breaks (see `CLAUDE.md` → "Known limitations"). Because
  of this, the rebuild run **must be given the same plan args as the original tokenize
  run**, in two tiers:
  - **Document filters** — `--mode`, `--min-pixels`, `--max-pixels`,
    `--max-images-per-doc`, `--spatial-factor`. These change the plan's document set,
    so a mismatch is **caught**: the rebuilt plan's `total_documents` won't match the
    manifest's group count and the guardrail aborts. (`--prepare-only` runs the
    guardrail on the head node, so this fails fast before the slurm array launches.)
  - **Other plan args** — `--text-column`, `--tokenizer-min-pixels`,
    `--tokenizer-max-pixels`, `--window-size`, `--batch-size`, `--max-batch-tokens`.
    Copy these verbatim too so the rebuilt plan is faithful; a mismatch here is **not
    auto-detected**.

  The scan-time and `decontaminate_manifest.py` paths (HF and WDS) operate on raw
  source rows/keys and are unaffected.

---

## 7. Quick reference — data shape per storage

| Storage | Shards | Manifest columns used for exclusion | List support |
|---------|--------|-------------------------------------|--------------|
| **HF parquet/arrow** | `.parquet` / `.arrow` | `shard_path`, `chunk_index`, `row_in_chunk` (+ `group_id` for multi-image) | ✅ `stem_row` / `innovator_vl` |
| **WebDataset** | `.tar` | `sample_key`, `tar_path` (+ `group_id` for multi-image) | ✅ `wds_key` (`tar:sample_key`) — scan-time + `decontaminate_manifest.py` |
| **jsonl + tar** | `.jsonl` + `.tar` | `image_ref`, `tar_path`, `group_id` | ❌ not built in |
