"""Gate 2: structural integrity of a shard-local alignment payload.

Verifies the per-image block contract (spec invariant 3). For each sampled block:
``<|img_start|>`` H*W ``<|img_token_start|>`` vision/EOL... ``<|img_end_of_frame|>``
``<|img_end|>`` — the EOL count equals ``resize_h // 16``, the vision-token count
equals ``(resize_h // 16) * (resize_w // 16)``, and the only legal place for text
ids is the dims header (before ``<|img_token_start|>``). The dims come from
``views/*.parquet``, so the cross-check is what makes Gate 2 protect invariant 3.
Token ids come from the manifest's ``token_layout`` (derived at build time);
the checker carries no token-id literals.

Usage: ``python vision_tokenization/tests/alignment/check_store.py <root> [--n 64]``
Exits non-zero on any violation.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _iter_view_specs(manifest: dict):
    for split_specs in manifest["views"].values():
        for spec in split_specs:
            yield spec


def check(root: Path, n: int = 64) -> int:
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["token_dtype"] == "<i4", "manifest token_dtype is not <i4"
    assert manifest.get("payload_format") == "alignment_shard_local_v1"
    if manifest.get("schema_version", 3) >= 4:
        raise SystemExit(
            f"{root}: schema {manifest['schema_version']} (post-binidx); Gate 2 checks the "
            "pre-binidx deduped store -- run scripts/_algn_binidx_check.py on the .bin instead"
        )
    tl = manifest["token_layout"]
    img_start, img_end = tl["img_start"], tl["img_end"]
    img_token_start, eol, eof = tl["img_token_start"], tl["eol"], tl["eof"]
    vis_lo, vis_hi = tl["vision_lo"], tl["vision_hi"]

    blob = np.memmap(root / manifest["raw_blob"], dtype=np.uint8, mode="r")
    blocks = []
    for spec in _iter_view_specs(manifest):
        tokens = np.memmap(root / spec["tokens"], dtype=np.dtype(manifest["token_dtype"]), mode="r")
        for row in pq.read_table(root / spec["view"]).to_pylist():
            for img in row["images"]:
                ro, rl = int(img["raw_offset"]), int(img["raw_length"])
                if ro < 0 or ro + rl > blob.size:
                    raise AssertionError(
                        f"raw slice [{ro}:{ro + rl}] out of bounds for media_raw.blob ({blob.size} bytes)"
                    )
                off = int(img["token_offset"])
                ln = int(img["token_length"])
                blocks.append((
                    img["media_id"],
                    int(img["resize_height"]),
                    int(img["resize_width"]),
                    np.asarray(tokens[off:off + ln]),
                ))

    ids = random.Random(0).sample(range(len(blocks)), min(n, len(blocks)))
    bad = 0
    for idx in ids:
        mid, rh, rw, b = blocks[idx]
        errs = []
        if not (b[0] == img_start and b[-1] == img_end):
            errs.append("missing img_start/img_end wrapper")
        if int((b == img_token_start).sum()) != 1:
            errs.append("img_token_start count != 1")
        if int((b == eof).sum()) != 1:
            errs.append("img_end_of_frame count != 1")
        h_rows = int((b == eol).sum())
        if h_rows != rh // 16:
            errs.append(f"EOL rows {h_rows} != resize_h//16 {rh // 16}")
        vis = b[(b >= vis_lo) & (b <= vis_hi)]
        if len(vis) != (rh // 16) * (rw // 16):
            errs.append(
                f"vision count {len(vis)} != (h//16)*(w//16) {(rh // 16) * (rw // 16)}")
        # Body = after the first img_token_start, before the trailing
        # [EOF, img_end]; only vision-range ids and EOL are legal there.
        body_start = int(np.argmax(b == img_token_start)) + 1
        body = b[body_start:-2]
        illegal = body[~(((body >= vis_lo) & (body <= vis_hi)) | (body == eol))]
        if len(illegal):
            errs.append(f"{len(illegal)} non-vision/EOL ids in block body")
        if errs:
            print(f"BAD block {mid[:12]} (h={rh} w={rw} len={len(b)}): {'; '.join(errs)}")
            bad += 1
    print(f"checked {len(ids)} blocks: {len(ids) - bad} ok, {bad} bad")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate 2: alignment media store integrity")
    ap.add_argument("root", type=Path, help="dataset root (contains manifest.json)")
    ap.add_argument("--n", type=int, default=64, help="number of blocks to sample")
    args = ap.parse_args()
    return check(args.root, args.n)


if __name__ == "__main__":
    sys.exit(main())
