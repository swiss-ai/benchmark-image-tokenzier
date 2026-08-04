#!/usr/bin/env python3
"""Reconstruct a tokenized sequence from MMIDIDX files and render as markdown.

Usage::

    python -m vision_tokenization.utils.reconstruct_sequence \\
        /path/to/rank_0000_chunk_0000 42 -o /tmp/seq_42.md

    # Decode vision tokens back to images (requires GPU):
    python -m vision_tokenization.utils.reconstruct_sequence \\
        /path/to/rank_0000_chunk_0000 42 -o /tmp/seq_42.md --decode-images
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def _parse_sequence(seq: np.ndarray, tokenizer, vision_token_offset: int):
    """Parse a token sequence into logical segments.

    Structure tokens (img_start, img_end, img_token_start, img_end_of_row,
    img_end_of_frame) are resolved from the tokenizer by name — Apertus 1.5
    appends them above the text vocab, Apertus 2 renames them in place inside it.
    Vision codebook IDs start at *vision_token_offset*.
    """
    IMG_START = tokenizer.encode("<|img_start|>", add_special_tokens=False)[0]
    IMG_END = tokenizer.encode("<|img_end|>", add_special_tokens=False)[0]
    IMG_TOKEN_START = tokenizer.encode("<|img_token_start|>", add_special_tokens=False)[0]

    segments = []
    i = 0
    while i < len(seq):
        tid = int(seq[i])
        if tid == IMG_START:
            i += 1
            # Read dimension text tokens until IMG_TOKEN_START
            dim_text_ids = []
            while i < len(seq) and int(seq[i]) != IMG_TOKEN_START:
                dim_text_ids.append(int(seq[i]))
                i += 1
            dim_text = tokenizer.decode(dim_text_ids).strip()
            if i < len(seq) and int(seq[i]) == IMG_TOKEN_START:
                i += 1
            # Read only actual vision codebook IDs (>= vision_token_offset)
            vision_ids = []
            while i < len(seq):
                t = int(seq[i])
                if t >= vision_token_offset:
                    vision_ids.append(t - vision_token_offset)
                elif t == IMG_END:
                    i += 1  # consume img_end
                    break
                # else: structure token (EOL, EOF) — skip
                i += 1
            segments.append({
                "type": "image",
                "dims": dim_text,
                "n_vision_tokens": len(vision_ids),
                "vision_ids": vision_ids,
            })
        else:
            text_ids = []
            while i < len(seq):
                t = int(seq[i])
                if t == IMG_START:
                    break
                text_ids.append(t)
                i += 1
            text = tokenizer.decode(text_ids, skip_special_tokens=False).strip()
            segments.append({
                "type": "text",
                "text": text,
                "n_tokens": len(text_ids),
            })
    return segments


def render_markdown(
    segments: list[dict],
    seq_len: int,
    prefix: str,
    seq_index: int,
    image_dir: Optional[Path] = None,
) -> str:
    """Render parsed segments as markdown."""
    n_img = sum(1 for s in segments if s["type"] == "image")
    n_text = sum(1 for s in segments if s["type"] == "text")
    img_tokens = sum(s["n_vision_tokens"] for s in segments if s["type"] == "image")
    text_tokens = sum(s["n_tokens"] for s in segments if s["type"] == "text")

    lines = [
        f"# Sequence {seq_index}: {seq_len:,} tokens\n",
        f"- **File**: `{Path(prefix).stem}`",
        f"- **Index**: {seq_index}",
        f"- **Segments**: {n_img} images ({img_tokens:,} vision tokens) + {n_text} text ({text_tokens:,} text tokens)\n",
        "---\n",
    ]

    img_counter = 0
    for seg in segments:
        if seg["type"] == "image":
            dims = seg["dims"]
            n = seg["n_vision_tokens"]
            if image_dir and (image_dir / f"image_{img_counter}.png").exists():
                rel_path = f"{image_dir.name}/image_{img_counter}.png"
                lines.append(f"![image_{img_counter}]({rel_path})\n")
                lines.append(f"*{n:,} vision tokens, grid={dims}*\n")
            else:
                lines.append(f"**[IMAGE: {n:,} vision tokens, grid={dims}]**\n")
            img_counter += 1
        else:
            text = seg["text"]
            if not text:
                continue
            # Escape markdown special chars so raw text renders as-is.
            escaped = text.replace("#", "\\#").replace("<", "\\<").replace(">", "\\>")
            lines.append(f"{escaped}\n")
        lines.append("\n---\n")

    return "\n".join(lines)


def decode_images(segments: list[dict], output_dir: Path, device: str = "cuda:0"):
    """Decode vision tokens back to images using Emu3.5."""
    import torch
    from Tokenizer.Emu3_5_IBQ import Emu3_5_IBQ
    from PIL import Image

    model = Emu3_5_IBQ(
        '/capstor/store/cscs/swissai/infra01/MLLM/Emu3.5-VisionTokenizer',
        device=device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    img_counter = 0
    for seg in segments:
        if seg["type"] != "image":
            continue
        dims = seg["dims"]
        vision_ids = seg["vision_ids"]
        try:
            h, w = [int(x) for x in dims.split("*")]
            indices = torch.tensor(vision_ids, dtype=torch.long, device=device).reshape(1, h, w)
            with torch.inference_mode():
                img = model.decode(indices)
            img_np = img[0].permute(1, 2, 0).cpu().numpy()
            img_np = ((img_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_np).save(output_dir / f"image_{img_counter}.png")
            print(f"  Decoded image_{img_counter}.png ({h}x{w})")
        except Exception as e:
            print(f"  Failed to decode image_{img_counter}: {e}")
        img_counter += 1


def main():
    parser = argparse.ArgumentParser(description="Reconstruct a tokenized sequence as markdown")
    parser.add_argument("prefix", help="MMIDIDX file prefix (without .bin/.idx)")
    parser.add_argument("index", type=int, help="Sequence index within the file")
    parser.add_argument("-o", "--output", default=None, help="Output markdown path (default: recon_examples dir)")
    parser.add_argument("--output-dir", default="/capstor/store/cscs/swissai/infra01/vision-datasets/recon_examples", help="Default output directory")
    parser.add_argument("--decode-images", action="store_true", help="Decode vision tokens to images (requires GPU)")
    parser.add_argument("--tokenizer", default="/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    from transformers import AutoTokenizer

    ds = IndexedDataset(args.prefix)
    if args.index >= len(ds):
        print(f"Error: index {args.index} out of range (dataset has {len(ds)} sequences)", file=sys.stderr)
        sys.exit(1)

    seq = np.array(ds[args.index], dtype=np.int32).copy()
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    from vision_tokenization.discrete.emu.token_layout import vision_band
    from vision_tokenization.utils.json import json_load

    vision_token_offset, _ = vision_band(
        json_load(Path(args.tokenizer) / "tokenizer_config.json"))

    segments = _parse_sequence(seq, tok, vision_token_offset)

    # Resolve output path
    if args.output:
        out_path = Path(args.output)
    else:
        prefix_name = Path(args.prefix).parent.parent.name  # e.g. "pin_200m"
        bucket = Path(args.prefix).parent.name  # e.g. "stage2" or "lct"
        out_dir = Path(args.output_dir) / f"{prefix_name}_{bucket}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"seq_{args.index}.md"

    image_dir = None
    if args.decode_images:
        image_dir = out_path.parent / f"seq_{args.index}_images"
        decode_images(segments, image_dir, device=args.device)

    md = render_markdown(segments, len(seq), args.prefix, args.index, image_dir=image_dir)

    out_path.write_text(md)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
