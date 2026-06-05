# Apertus — Image Tokenization at Scale

Production tokenization infrastructure for **discrete image tokenizers**
in the Apertus multimodal stack. The primary workflow here is turning
hundreds of millions of images into Megatron `.bin/.idx` micro-shards
for training; a smaller benchmarking sandbox lives alongside it.

Two workflows:

1. **Distributed dataset tokenization** *(the main thing)* — WebDataset
   and HuggingFace corpora into Megatron shards via `torch.distributed`,
   one independent rank per GPU, no NCCL between ranks. Built and run
   on multi-node Slurm; measured rank-wallclock imbalance under 2 % at
   real run sizes.
2. **Tokenizer benchmarking** *(the sandbox)* — encode → decode → score
   ~14 discrete tokenizers (Emu3, IBQ, FlowMo, OpenMAGViT2, Cosmos,
   TokenFlow, TiTok, UniTok, LlamaGen, Selftok, Seed, VQGAN, …) for
   reconstruction quality, plus qualitative VLM evaluations.

### Scale already in production

- **42 dataset wrappers** under [`scripts/slurm/`](scripts/slurm/) —
  including `pin_200m` (~200 M samples), `llava85m_midtrain` (~85 M),
  `latex_formulas_80m` (~80 M), `blip3_grounding_50m`,
  `innovator_vl_46m`, `facecaption_15m`,
  `megalith_10m_florence2`, `mit_10m_recap`, and dozens of smaller
  SFT / interleave datasets.
- **5 modes**: `image_only`, `image2text`, `text2image`, `sft`,
  `interleave`.
- Outputs feed Apertus 1.5 multimodal training directly — the same
  `.bin/.idx` pair Megatron-LM reads natively, no conversion step.
- Tens of TB of training-ready tokens produced and reused across
  ablations; manifests are reused so re-tokenizing for a different
  batch size or rank count costs *seconds* of replanning.

This is **not a generic pip-installable library**. It targets the CSCS
Clariden cluster (GH200) with a Slurm + container runtime; expect to
adapt paths and the Slurm preamble for any other site.

**Authors:** Yixuan Xu*, Raphael Krest, Nicola Irmiger.

*Major contributor and maintainer.

---

## Quickstart — tokenize one dataset

Submit `sbatch` from a **CSCS login node**; heavy work (Python, GPU, manifest
access) runs on compute nodes inside the container each Slurm script declares.
Each `scripts/slurm/<dataset>.slurm` runs the distributed tokenizer, and
`scripts/slurm/ops/merge.slurm` combines the per-rank shards into a Megatron-LM
`.bin`/`.idx` pair. This repo **consumes** manifests built upstream in the
`multimodal-data` repo — it doesn't build them.

The full step-by-step Quickstart — job submission, the `afterany` merge chain,
resume semantics, and adding a new dataset — is the **single source of truth**
in [`vision_tokenization/README.md`](vision_tokenization/README.md) §2.

---

## Workflows

### 1. Distributed tokenization → `vision_tokenization/`

The production data path. `torch.distributed` with one independent rank
per GPU (no NCCL between ranks), Hydra config composition, deterministic
batch plans, last-rank-out merging. Five modes: `image_only`,
`image2text`, `text2image`, `sft`, `interleave`.

→ See [`vision_tokenization/README.md`](vision_tokenization/README.md)
for architecture, design rationale, and full configuration reference.

### 2. Tokenizer reconstruction benchmarks → `benchmarks/`

Encode-decode 14 tokenizers on a shared image set, score with PSNR /
SSIM / LPIPS, save reconstructions side-by-side under `assets/`.
Best LPIPS in the current sweep: `Emu3VisionTokenizer` at 0.019
(30.14 PSNR, 10.9k tokens).

→ See [`benchmarks/README.md`](benchmarks/README.md) for the layout,
how to run a sweep, and the tokenizer comparison reference card.
Hard numbers in [`benchmarks/metrics/metrics_results.md`](benchmarks/metrics/metrics_results.md).

### 3. Qualitative VLM benchmarks → `vision_tokenization/qualitative_benchmark/`

Evaluate a vision-language model trained on top of these tokenizers:
captioning, VQA, image completion. Publishes a static viewer to
[`docs/index.html`](docs/index.html).

→ See [`vision_tokenization/qualitative_benchmark/README.md`](vision_tokenization/qualitative_benchmark/README.md).

---

## Requirements

| Component | Expectation |
|---|---|
| **Python** | Provided by the container's `/opt/venv` (Python 3.11+). No `pip install -e .` flow — the repo is run in-place. |
| **GPU** | Tested on NVIDIA GH200 120 GB at CSCS Clariden. |
| **Runtime** | Slurm + Pyxis containers. The container TOMLs live under [`scripts/envs/`](scripts/envs/) (`nemo_25_11.toml`, `nemo_26_02.toml`); every Slurm script references one via `#SBATCH --environment=`. |
| **Megatron-LM** | Required at runtime for the `IndexedDatasetBuilder`. The pipeline expects a sibling `Megatron-LM` checkout next to this repo, or `MEGATRON_PATH` set; the merge step will also auto-discover it. |
| **Submodules** | One git submodule (`Tokenizer/submodules/Emu3.5`). Clone with `--recurse-submodules` or run `git submodule update --init --recursive`. |
| **Storage** | Designed for Lustre (CSCS `/capstor` and `/iopsstor`). Manifests are small (≤ MB-scale); shards land under `/capstor/.../tokenized/{mode}/{name}/`. |

There is no `requirements.txt` for production deps — the container is
the source of truth. `requirements-dev.txt` only carries the formatter
toolchain (see [Developer formatting](#developer-formatting) below).

---

## Repository structure

```
.
├── vision_tokenization/    # Distributed tokenization pipeline
│   ├── pipeline/           # Executor, prefetch, writers, merge
│   ├── discrete/           # Tokenizer wrappers (Emu3, Emu3.5; image_only/sft/pair/interleave)
│   ├── indexing/           # Manifest scanners, batch planning, tar reader
│   ├── formats/            # Megatron IndexedDataset (.bin/.idx) I/O
│   ├── configs/            # Hydra config tree (root + per-mode dataset YAMLs)
│   ├── tests/              # 25 tests: format integrity, integration, SFT parsers
│   ├── qualitative_benchmark/  # VLM Q&A, captioning, image completion (own README)
│   └── profile/            # GH200 throughput / OOM profiling notes
├── Tokenizer/              # Vendored tokenizer implementations + submodules (Emu3.5)
├── benchmarks/             # Reconstruction sweeps, metrics, notebooks (own README)
├── scripts/
│   ├── envs/               # Container TOMLs for Slurm
│   └── slurm/              # tokenize/{image2text,interleave,sft}/, ops/, serve/
├── docs/                   # Static viewer for qualitative VLM results
├── conftest.py
├── pyproject.toml          # Black config (not a package definition)
├── requirements-dev.txt    # Formatter toolchain only
└── format.sh               # Wraps black / isort / flake8
```

---

## Tokenizer comparison (summary)

The decision-critical columns. Full reference card and reconstruction
metrics in [`benchmarks/README.md`](benchmarks/README.md) and
[`benchmarks/metrics/metrics_results.md`](benchmarks/metrics/metrics_results.md).

| Model | Token type | # Tokens / image | Codebook | Understanding | Generation |
|---|---|---|---|:-:|:-:|
| Open-MagVit2 | Spatial 2D | 16×16 compression | 262,144 | ✅ | ✅ |
| Emu3-VisionTokenizer | Spatial 2D | 8×8 compression | 32,768 | ✅ | ✅ |
| Cosmos | Spatial 2D | 16×16 or 8×8 | 64,000 | ✅ | ✅ |
| FlowMo Hi | Sequential 1D | 1,024 | 16,384 | — | ✅ |
| TiTok | Sequential 1D | 256 | 4,096 | — | ✅ |
| Selftok | Sequential AR | 512 / 1,024 / 1,536 | 32,768 | ✅ | ✅ |
| UniTok | Sequential 1D | 8 × 256 | 8 × 16,000 | ✅ | ✅ |
| DetailFlow | Sequential AR | 128 / 256 / 512 | 8,192 | — | ✅ |
| TokenFlow | Spatial 2D (next-scale) | 16×16 / 27×27 | 32,768 | ✅ | ✅ |
| VILA-U | Spatial 2D (RQ) | 16×16×4 | 16,384 | ✅ | ✅ |

---

## Developer formatting

Black + isort + flake8, configured to skip submodules:

```bash
pip install -r requirements-dev.txt
./format.sh
```

`pyproject.toml` only carries Black's line length and exclude rules — it
is *not* a package manifest.
