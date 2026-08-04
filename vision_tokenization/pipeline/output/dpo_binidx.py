"""Derive a single-sequence DPO ``.bin/.idx`` + ``index.parquet`` from a published alignment store.

The engine already tokenized each pair's text (``views_tokenized/``, vision-free); this step
only splices the store's deduped vision blocks (``tokens/``) into the prompt slots and
concatenates one contiguous document per pair:

    doc = [ prompt(text + inlined vision) | chosen | rejected ]

plus ``index.parquet`` carrying the slice boundaries + image-token locations. No tokenizer —
pure assembly. The calling phase (``run_dpo_binidx``) registers the outputs and retires the
now-redundant ``tokens/`` + ``views_tokenized/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vision_tokenization.discrete.dpo_pairs import (
    assemble_from_tokenized,
    assemble_prompt_only,
    prompt_from_row,
    tokenized_from_row,
)
from vision_tokenization.formats.megatron import IndexedDatasetBuilder
from vision_tokenization.pipeline.output.alignment_merge import _fsync_file
from vision_tokenization.pipeline.runtime.checkpoint import finalize_shard_writer
from vision_tokenization.utils.json import json_load

SPLITS = ("train", "validation")

INDEX_SCHEMA = pa.schema([
    ("prompt_id", pa.string()),
    ("prompt_len", pa.int32()),
    ("chosen_len", pa.int32()),
    ("rejected_len", pa.int32()),
    ("image_offsets", pa.list_(pa.int32())),
    ("image_lengths", pa.list_(pa.int32())),
    ("seq_chosen_len", pa.int32()),
    ("seq_rejected_len", pa.int32()),
    ("image_tok", pa.int32()),
])

RL_PROMPT_INDEX_SCHEMA = pa.schema([
    ("prompt_id", pa.string()),
    ("prompt_len", pa.int32()),
    ("image_offsets", pa.list_(pa.int32())),
    ("image_lengths", pa.list_(pa.int32())),
    ("image_tok", pa.int32()),
    ("answer", pa.string()),
    ("answer_variants", pa.list_(pa.string())),
    ("enable_thinking", pa.bool_()),
])


def build_dpo_binidx(store_dir: str, *, system: str = "empty", task: str = "preference") -> dict:
    if task == "rl_prompt":
        return _build_rl_prompt_binidx(store_dir, system=system)
    if task != "preference":
        raise ValueError(f"unsupported alignment task: {task!r}")

    store = Path(store_dir)
    _ = json_load(store / "manifest.json")

    out = {"system": system, "task": "preference", "splits": {}}
    for split in SPLITS:
        view_path = store / "views" / f"{split}-00000.parquet"
        tok_views_path = store / "views_tokenized" / f"{split}-00000.parquet"
        tok_path = store / "tokens" / f"{split}-00000.i32"
        if not view_path.exists():
            continue
        views = pq.read_table(view_path).to_pylist()
        if not views:
            continue
        for required in (tok_views_path, tok_path):
            if not required.exists():
                raise FileNotFoundError(
                    f"{required} missing -- retired by a prior binidx run; "
                    "re-run scan -> encode -> merge to rebuild before re-running binidx"
                )
        tok_rows = pq.read_table(tok_views_path).to_pylist()
        if len(tok_rows) != len(views):
            raise RuntimeError(f"{split}: {len(views)} view rows but {len(tok_rows)} tokenized rows")
        tokens = np.memmap(tok_path, dtype="<i4", mode="r")

        bin_tmp, idx_tmp = str(store / f"{split}.bin.tmp"), str(store / f"{split}.idx.tmp")
        builder = IndexedDatasetBuilder(bin_tmp, dtype=np.int32)
        index_rows = []
        for vrow, trow in zip(views, tok_rows):
            if vrow["prompt_id"] != trow["prompt_id"]:
                raise RuntimeError(
                    f"{split}: views/views_tokenized order mismatch "
                    f"({vrow['prompt_id']} vs {trow['prompt_id']})")
            doc, irow = assemble_from_tokenized(
                tokenized_from_row(trow), vrow["images"], tokens, prompt_id=vrow["prompt_id"])
            if (irow["seq_chosen_len"], irow["seq_rejected_len"]) != (
                    vrow["seq_chosen_len"], vrow["seq_rejected_len"]):
                raise RuntimeError(
                    f"{vrow['prompt_id']}: store seq lengths "
                    f"({vrow['seq_chosen_len']}, {vrow['seq_rejected_len']}) disagree with binidx "
                    f"({irow['seq_chosen_len']}, {irow['seq_rejected_len']})")
            builder.add_item(doc)
            builder.end_document()
            index_rows.append(irow)

        declared = np.fromiter(
            (r["prompt_len"] + r["chosen_len"] + r["rejected_len"] for r in index_rows),
            dtype=np.int64, count=len(index_rows),
        )
        if not np.array_equal(np.asarray(builder.sequence_lengths, dtype=np.int64), declared):
            raise RuntimeError(
                f"{split}: doc/index misaligned -- {len(builder.sequence_lengths)} .bin docs vs "
                f"{len(index_rows)} index rows, or a per-doc length disagrees"
            )
        finalize_shard_writer(builder, bin_tmp, idx_tmp, str(store / f"{split}.bin"), str(store / f"{split}.idx"))

        idx_out_tmp = str(store / f"index_{split}.parquet.tmp")
        pq.write_table(pa.Table.from_pylist(index_rows, schema=INDEX_SCHEMA), idx_out_tmp)
        _fsync_file(idx_out_tmp)
        os.replace(idx_out_tmp, str(store / f"index_{split}.parquet"))

        sc = np.array([r["seq_chosen_len"] for r in index_rows])
        spec = out["splits"][split] = {
            "bin": f"{split}.bin",
            "idx": f"{split}.idx",
            "index": f"index_{split}.parquet",
            "n_pairs": len(index_rows),
            "seq_chosen_p50": int(np.percentile(sc, 50)),
            "seq_chosen_p99": int(np.percentile(sc, 99)),
            "seq_chosen_max": int(sc.max()),
        }
        print(f"[{split}] {spec['n_pairs']} docs -> {split}.bin/.idx + index_{split}.parquet  "
              f"(seq_chosen p50={spec['seq_chosen_p50']} max={spec['seq_chosen_max']})")
    return out


def _build_rl_prompt_binidx(store_dir: str, *, system: str = "empty") -> dict:
    """The rl_prompt task branch: one ``[prompt]`` document per sample (no chosen/rejected
    span), so ``sequence_length(doc) == prompt_len``. The index drops the response lengths
    and carries the grading ``answer`` + ``answer_variants`` (from the view) plus the
    ``enable_thinking`` constant. ``prompt_len`` is single-sourced through ``pair_lengths``
    (in ``assemble_prompt_only``) and cross-checked against the merge-stamped view length."""
    store = Path(store_dir)
    _ = json_load(store / "manifest.json")

    out = {"system": system, "task": "rl_prompt", "splits": {}}
    for split in SPLITS:
        view_path = store / "views" / f"{split}-00000.parquet"
        tok_views_path = store / "views_tokenized" / f"{split}-00000.parquet"
        tok_path = store / "tokens" / f"{split}-00000.i32"
        if not view_path.exists():
            continue
        views = pq.read_table(view_path).to_pylist()
        if not views:
            continue
        for required in (tok_views_path, tok_path):
            if not required.exists():
                raise FileNotFoundError(
                    f"{required} missing -- retired by a prior binidx run; "
                    "re-run scan -> encode -> merge to rebuild before re-running binidx"
                )
        tok_rows = pq.read_table(tok_views_path).to_pylist()
        if len(tok_rows) != len(views):
            raise RuntimeError(f"{split}: {len(views)} view rows but {len(tok_rows)} tokenized rows")
        tokens = np.memmap(tok_path, dtype="<i4", mode="r")

        bin_tmp, idx_tmp = str(store / f"{split}.bin.tmp"), str(store / f"{split}.idx.tmp")
        builder = IndexedDatasetBuilder(bin_tmp, dtype=np.int32)
        index_rows = []
        for vrow, trow in zip(views, tok_rows):
            if vrow["prompt_id"] != trow["prompt_id"]:
                raise RuntimeError(
                    f"{split}: views/views_tokenized order mismatch "
                    f"({vrow['prompt_id']} vs {trow['prompt_id']})")
            doc, irow = assemble_prompt_only(
                prompt_from_row(trow), vrow["images"], tokens, prompt_id=vrow["prompt_id"])
            if irow["prompt_len"] != vrow["prompt_len"]:
                raise RuntimeError(
                    f"{vrow['prompt_id']}: store prompt length {vrow['prompt_len']} "
                    f"disagrees with binidx {irow['prompt_len']}")
            irow["answer"] = str(vrow["answer"])
            irow["answer_variants"] = [str(a) for a in (vrow.get("answer_variants") or [])]
            irow["enable_thinking"] = True
            builder.add_item(doc)
            builder.end_document()
            index_rows.append(irow)

        declared = np.fromiter(
            (r["prompt_len"] for r in index_rows), dtype=np.int64, count=len(index_rows))
        if not np.array_equal(np.asarray(builder.sequence_lengths, dtype=np.int64), declared):
            raise RuntimeError(
                f"{split}: doc/index misaligned -- {len(builder.sequence_lengths)} .bin docs vs "
                f"{len(index_rows)} index rows, or a per-doc length disagrees"
            )
        finalize_shard_writer(builder, bin_tmp, idx_tmp, str(store / f"{split}.bin"), str(store / f"{split}.idx"))

        idx_out_tmp = str(store / f"index_{split}.parquet.tmp")
        pq.write_table(pa.Table.from_pylist(index_rows, schema=RL_PROMPT_INDEX_SCHEMA), idx_out_tmp)
        _fsync_file(idx_out_tmp)
        os.replace(idx_out_tmp, str(store / f"index_{split}.parquet"))

        pl = np.array([r["prompt_len"] for r in index_rows])
        spec = out["splits"][split] = {
            "bin": f"{split}.bin",
            "idx": f"{split}.idx",
            "index": f"index_{split}.parquet",
            "n_samples": len(index_rows),
            "prompt_p50": int(np.percentile(pl, 50)),
            "prompt_p99": int(np.percentile(pl, 99)),
            "prompt_max": int(pl.max()),
        }
        print(f"[{split}] {spec['n_samples']} docs -> {split}.bin/.idx + index_{split}.parquet  "
              f"(prompt p50={spec['prompt_p50']} max={spec['prompt_max']})")
    return out
