#!/usr/bin/env python3
"""Reconstruct an alignment document from a binidx alignment store and render it as
markdown — the alignment analogue of ``reconstruct_sequence.py``.

Dispatches on ``manifest["task"]`` (absent ⇒ ``"preference"``):

- ``preference`` — render ``[prompt | chosen | rejected]``: the prompt's text + inlined
  vision, then the chosen and rejected responses.
- ``rl_prompt`` — render ``[prompt]`` only (the whole doc IS the prompt; ``doclen ==
  prompt_len``), then print the grading ``answer`` + ``answer_variants`` +
  ``enable_thinking`` from the index.

Reads ``<store>/{split}.bin`` + ``index_{split}.parquet`` (the slice boundaries) +
``views/{split}`` (raw image refs) + ``media_raw.blob`` (raw bytes). Torch-free: it
decodes nothing on GPU — the prompt's vision is shown by embedding the raw image
sliced from the blob, and the text is detokenized directly.

Usage::

    python -m vision_tokenization.utils.reconstruct_dpo <store_dir> -n 3 -o /tmp/dpo_md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from vision_tokenization.utils.reconstruct_sequence import _parse_sequence


def _render_prompt(prompt, view, stem, tok, vision_token_offset, blob, out_dir) -> tuple[list[str], int]:
    """Render the ``[prompt]`` span (text + inlined vision sliced from the blob) to
    markdown lines, writing each image out next to the markdown. Returns the lines and
    the image count. Shared by both tasks — the prompt layout is task-neutral."""
    segments = _parse_sequence(prompt, tok, vision_token_offset)
    n_img = sum(1 for s in segments if s["type"] == "image")
    lines, img_i = [], 0
    for seg in segments:
        if seg["type"] == "image":
            im = view["images"][img_i]
            raw = bytes(blob[im["raw_offset"]:im["raw_offset"] + im["raw_length"]])
            fname = f"{stem}_img{img_i}.{im['raw_ext']}"
            (out_dir / fname).write_bytes(raw)
            lines.append(
                f"![img]({fname}) *{seg['n_vision_tokens']:,} vision tok, grid={seg['dims']}, "
                f"source {im['width']}x{im['height']}*\n")
            img_i += 1
        elif seg["text"]:
            lines.append("```\n" + seg["text"] + "\n```\n")
    return lines, n_img


def _render_pair(stem, idx_row, doc, view, tok, vision_token_offset, blob, out_dir) -> str:
    pl, cl, rl = idx_row["prompt_len"], idx_row["chosen_len"], idx_row["rejected_len"]
    prompt, chosen, rejected = doc[:pl], doc[pl:pl + cl], doc[pl + cl:pl + cl + rl]
    prompt_lines, n_img = _render_prompt(prompt, view, stem, tok, vision_token_offset, blob, out_dir)

    lines = [
        f"# DPO pair — `{idx_row['prompt_id']}`\n",
        f"- doc **{len(doc):,}** tokens = prompt **{pl}** | chosen **{cl}** | rejected **{rl}**",
        f"- {n_img} image(s), **{idx_row['image_tok']:,}** vision tokens inlined in the prompt",
        f"- `[prompt|chosen]` = first {idx_row['seq_chosen_len']} tok · "
        f"`[prompt|rejected]` = prompt + the rejected slice ({idx_row['seq_rejected_len']} tok)\n",
        "## Prompt (text + inlined vision)\n",
        *prompt_lines,
        "## Chosen\n", "```\n" + tok.decode(chosen, skip_special_tokens=False) + "\n```\n",
        "## Rejected\n", "```\n" + tok.decode(rejected, skip_special_tokens=False) + "\n```\n",
    ]
    return "\n".join(lines)


def _render_rl_prompt(stem, idx_row, doc, view, tok, vision_token_offset, blob, out_dir) -> str:
    pl = idx_row["prompt_len"]
    if len(doc) != pl:
        raise RuntimeError(f"{idx_row['prompt_id']}: doc length {len(doc)} != prompt_len {pl} (whole doc is the prompt)")
    prompt_lines, n_img = _render_prompt(doc, view, stem, tok, vision_token_offset, blob, out_dir)

    variants = idx_row.get("answer_variants") or []
    lines = [
        f"# RL prompt — `{idx_row['prompt_id']}`\n",
        f"- doc **{len(doc):,}** tokens = prompt **{pl}** (the whole doc is the prompt)",
        f"- {n_img} image(s), **{idx_row['image_tok']:,}** vision tokens inlined in the prompt",
        f"- enable_thinking = **{idx_row['enable_thinking']}**\n",
        "## Prompt (text + inlined vision)\n",
        *prompt_lines,
        "## Answer (grading target)\n", "```\n" + str(idx_row["answer"]) + "\n```\n",
    ]
    if variants:
        lines.append("## Answer variants\n")
        lines += ["```\n" + str(v) + "\n```\n" for v in variants]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Reconstruct alignment [prompt|chosen|rejected] / [prompt] docs as markdown")
    p.add_argument("store_dir")
    p.add_argument("--split", default="train")
    p.add_argument("-n", type=int, default=3, help="number of docs, spread across the split")
    p.add_argument("--indices", type=int, nargs="*", help="specific doc indices (overrides -n)")
    p.add_argument("-o", "--output-dir", default=None)
    p.add_argument("--tokenizer", default=None)
    args = p.parse_args()

    store = Path(args.store_dir)
    manifest = json.loads((store / "manifest.json").read_text())
    task = manifest.get("task", "preference")
    if task == "preference":
        render, doclen_of, label = _render_pair, (
            lambda r: r["prompt_len"] + r["chosen_len"] + r["rejected_len"]), "pair"
    elif task == "rl_prompt":
        render, doclen_of, label = _render_rl_prompt, (lambda r: r["prompt_len"]), "prompt"
    else:
        raise ValueError(f"unsupported alignment task: {task!r}")

    tok_dir = Path(args.tokenizer or manifest["tokenizer"]["path"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tok_dir), trust_remote_code=True, use_fast=True)
    vision_token_offset = json.loads((tok_dir / "vision_token_mapping.json").read_text())["vision_token_offset"]

    idx = pq.read_table(store / f"index_{args.split}.parquet").to_pylist()
    binarr = np.memmap(store / f"{args.split}.bin", dtype="<i4", mode="r")
    blob = np.memmap(store / manifest["raw_blob"], dtype=np.uint8, mode="r")
    doclen = [doclen_of(r) for r in idx]
    offs = np.concatenate([[0], np.cumsum(doclen)])

    n = len(idx)
    picks = args.indices if args.indices else [round(k * (n - 1) / max(1, args.n - 1)) for k in range(args.n)]
    # views/ and index_*/ share row order; take only the picked rows, only the image
    # refs (the prompt/chosen/rejected text is decoded from the .bin, not read here).
    views_table = pq.read_table(store / "views" / f"{args.split}-00000.parquet", columns=["prompt_id", "images"])
    picked_views = views_table.take(picks).to_pylist()

    out_dir = Path(args.output_dir or (store / f"recon_{task}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for view, i in zip(picked_views, picks):
        r = idx[i]
        if view["prompt_id"] != r["prompt_id"]:
            raise RuntimeError(f"views/index row {i} desync: {view['prompt_id']} != {r['prompt_id']}")
        doc = np.asarray(binarr[offs[i]:offs[i + 1]])
        (out_dir / f"{label}_{i}.md").write_text(
            render(f"{label}_{i}", r, doc, view, tok, vision_token_offset, blob, out_dir))
        print(f"  {label} {i} ({r['prompt_id'][:40]}) -> {out_dir / f'{label}_{i}.md'}")
    print(f"Written {len(picks)} {label}s to {out_dir}")


if __name__ == "__main__":
    main()
