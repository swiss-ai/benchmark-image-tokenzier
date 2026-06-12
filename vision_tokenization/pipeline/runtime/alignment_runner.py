"""``alignment`` mode: freeze media for preference/RL datasets (views+media spec).

Single-rank. Writes ``<root>/scan.parquet`` (the geometry record, before any
GPU work), ``<root>/media/`` (sealed triple via ``MediaStoreWriter``),
``<root>/views/{train,validation}.parquet``, then ``manifest.json`` LAST (the
commit record), where ``<root> = cfg["output_dir"]`` (already task-keyed to
``.../{alignment|rl}/<output_name>`` by ``run_distributed_pipeline``).

The mode is task-neutral; ``cfg["task"]`` (``preference`` today) selects the row
adapter + view schema inside ``ingest_parquet`` via ``ROW_ADAPTERS``. The runner,
planner, media store, and manifest never branch on task.

Geometry comes from the scan (pipeline contract: scan before plan): ingest
decodes width/height once per unique media and skips corrupt/sub-16px images;
this runner only opens images to feed the GPU encoder.
"""

from __future__ import annotations

import hashlib
import io
import logging
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from vision_tokenization.indexing.alignment.ingest import (
    SPATIAL_FACTOR,
    ingest_parquet,
    write_scan_parquet,
)
from vision_tokenization.indexing.alignment.planning import plan_exact_dim_batches
from vision_tokenization.pipeline.output.media_store import (
    MediaStoreWriter,
    atomic_write_json,
)
from vision_tokenization.utils.image_geometry import smart_resize_dims_batch

logger = logging.getLogger(__name__)


def run_alignment_mode(cfg: dict) -> dict:
    from vision_tokenization.discrete.emu import create_tokenizer

    out = Path(cfg["output_dir"])
    res = ingest_parquet(Path(cfg["input_parquet"]), task=cfg["task"])
    if res.n_skipped_media:
        logger.warning("scan skipped %d corrupt/sub-%dpx images (and their pairs)",
                       res.n_skipped_media, SPATIAL_FACTOR)
    out.mkdir(parents=True, exist_ok=True)
    scan_size = write_scan_parquet(out / "scan.parquet", res.unique_media)

    tokenizer = create_tokenizer(
        mode="alignment",
        text_tokenizer_path=cfg["tokenizer_path"],
        device=f"cuda:{cfg['local_rank']}",
        min_pixels=cfg["tokenizer_min_pixels"],
        max_pixels=cfg["tokenizer_max_pixels"],
        max_encode_pixels=cfg.get("max_encode_pixels"),
        **(cfg.get("tokenizer_kwargs", {})),
    )

    # Exact smart-resize dims from the scan geometry (runner never decodes).
    resize_h, resize_w = smart_resize_dims_batch(
        np.array([um.height for um in res.unique_media], dtype=np.int64),
        np.array([um.width for um in res.unique_media], dtype=np.int64),
        min_pixels=cfg["tokenizer_min_pixels"],
        max_pixels=cfg["tokenizer_max_pixels"], factor=SPATIAL_FACTOR)
    dims = list(zip(range(len(res.unique_media)),
                    resize_h.tolist(), resize_w.tolist()))

    writer = MediaStoreWriter(out / "media")
    for batch in plan_exact_dim_batches(dims, batch_size=cfg["encode_batch_size"]):
        images = [Image.open(io.BytesIO(res.unique_media[i].raw)).convert("RGB")
                  for i in batch.member_indices]
        # [B, L] int64 CPU, rows INCLUDE outer BOS/EOS (encapsulate_batch);
        # tokenize_images is already @torch.inference_mode-decorated.
        batched = tokenizer.tokenize_images(
            images, (batch.resize_height, batch.resize_width))
        for i, row in zip(batch.member_indices, batched):
            um = res.unique_media[i]
            assert int(row[0]) == tokenizer.bos_id and int(row[-1]) == tokenizer.eos_id
            block = row[1:-1].numpy().astype(np.int32)   # <|img_start|>...<|img_end|>
            writer.add(um.media_id, tokens=block, raw=um.raw,
                       resize_h=batch.resize_height, resize_w=batch.resize_width,
                       kind="image", source=um.source, raw_ext=um.raw_ext)
    media_files = writer.seal()

    # Views: exact media token stats (skipped-media rows already dropped at scan).
    length_of = {r["media_id"]: r["length_elems"] for r in
                 pq.read_table(out / "media" / "media.000000.parquet").to_pylist()}
    rows = res.view_rows
    for row in rows:
        row["media_tokens_total"] = sum(length_of[m] for m in row["prompt_media_refs"])
        row["text_chars"] = (sum(len(m["content"]) for m in row["prompt"])
                             + len(row["chosen"]) + len(row["rejected"]))

    rng = random.Random(42)
    rng.shuffle(rows)
    n_val = min(cfg["val_rows"], max(1, len(rows) // 50))
    (out / "views").mkdir(parents=True, exist_ok=True)
    view_files = {}
    for name, part in (("validation", rows[:n_val]), ("train", rows[n_val:])):
        p = out / "views" / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(part), p)
        view_files[f"views/{name}.parquet"] = p.stat().st_size

    tok_sha = hashlib.sha256(
        (Path(cfg["tokenizer_path"]) / "tokenizer.json").read_bytes()).hexdigest()
    atomic_write_json(out / "manifest.json", {
        "schema_version": 1,
        "tokenizer": {"path": cfg["tokenizer_path"], "sha256": tok_sha},
        "vision_tokenizer": {"version": "Emu3.5",
                             "min_pixels": cfg["tokenizer_min_pixels"],
                             "max_pixels": cfg["tokenizer_max_pixels"]},
        "token_dtype": "<i4",
        "expected_min_model_vocab": 266440,
        "media_roots": ["media/"],
        "store_raw": True,
        "files": {"scan.parquet": scan_size,
                  **{f"media/{k}": v for k, v in media_files.items()}, **view_files},
        "source_input": str(cfg["input_parquet"]),
        "n_pairs": len(rows),
        "n_unique_media": len(dims),
        "n_skipped_media": res.n_skipped_media,
    })

    logger.info(
        "alignment mode done: %d pairs, %d unique media (%d skipped) -> %s",
        len(rows), len(dims), res.n_skipped_media, out)
    return {
        "output_dir": str(out),
        "samples_processed": len(rows),
        "tokens_generated": sum(length_of.values()),
        "n_pairs": len(rows),
        "n_unique_media": len(dims),
        "n_skipped_media": res.n_skipped_media,
    }
