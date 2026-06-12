"""Gate 2: structural integrity of an alignment media store.

Verifies the exact-dims contract (spec invariant 3). For each sampled block:
``<|img_start|>`` H*W ``<|img_token_start|>`` vision/EOL... ``<|img_end_of_frame|>``
``<|img_end|>`` — the EOL count equals ``resize_h // 16``, the vision-token count
equals ``(resize_h // 16) * (resize_w // 16)``, and the only legal place for text
ids is the dims header (before ``<|img_token_start|>``). The dims come from
``media.parquet``, so the cross-check is what makes Gate 2 protect invariant 3.

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

# File sits at vision_tokenization/tests/alignment/; add the repo root so the
# media store reader imports as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from vision_tokenization.pipeline.output.media_store import MediaStoreReader  # noqa: E402

IMG_START, IMG_END = 131073, 131074
IMG_TOKEN_START, EOL, EOF = 131075, 131076, 131077
VIS_LO, VIS_HI = 131272, 262343


def check(root: Path, n: int = 64) -> int:
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["token_dtype"] == "<i4", "manifest token_dtype is not <i4"
    reader = MediaStoreReader([root / r for r in manifest["media_roots"]])

    # media_id -> (resize_h, resize_w) from media.parquet (the dims cross-check).
    dims = {}
    for r in manifest["media_roots"]:
        for pq_file in sorted((root / r).glob("media.*.parquet")):
            for m in pq.read_table(pq_file).to_pylist():
                dims[m["media_id"]] = (m["resize_h"], m["resize_w"])

    ids = random.Random(0).sample(sorted(reader.index), min(n, len(reader.index)))
    bad = 0
    for mid in ids:
        b = reader.tokens(mid)
        rh, rw = dims[mid]
        errs = []
        if not (b[0] == IMG_START and b[-1] == IMG_END):
            errs.append("missing img_start/img_end wrapper")
        if int((b == IMG_TOKEN_START).sum()) != 1:
            errs.append("img_token_start count != 1")
        if int((b == EOF).sum()) != 1:
            errs.append("img_end_of_frame count != 1")
        h_rows = int((b == EOL).sum())
        if h_rows != rh // 16:
            errs.append(f"EOL rows {h_rows} != resize_h//16 {rh // 16}")
        vis = b[(b >= VIS_LO) & (b <= VIS_HI)]
        if len(vis) != (rh // 16) * (rw // 16):
            errs.append(
                f"vision count {len(vis)} != (h//16)*(w//16) {(rh // 16) * (rw // 16)}")
        # Body = after the first img_token_start, before the trailing
        # [EOF, img_end]; only vision-range ids and EOL are legal there.
        body_start = int(np.argmax(b == IMG_TOKEN_START)) + 1
        body = b[body_start:-2]
        illegal = body[~(((body >= VIS_LO) & (body <= VIS_HI)) | (body == EOL))]
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
