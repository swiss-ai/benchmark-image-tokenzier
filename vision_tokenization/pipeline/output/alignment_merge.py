"""Merge per-rank alignment spills into the single views/tokens store.

This module owns the alignment merge end to end: the per-media payload writer
(``AlignmentPayloadBackend``) and the merge that drives it
(``materialize_alignment``). It is torch-free — the writer lays stripped int32
token blocks into ``tokens/`` and ``views/`` with pyarrow + numpy only, so the
merge runs inline on the head node (no GPU, no torch), exactly like the bin/idx
``merge.py``. Raw image bytes are not rewritten here: they stay in the scan's
flat ``media_raw.blob``, sliced by each image ref's ``raw_offset``/``raw_length``.

Each rank GPU-encodes a disjoint slice of the unique media — the plan gives one
document per media, so ``split_image_batches_for_workers`` partitions media
across ranks with no overlap — and spills the stripped blocks via
``ComponentSpillWriter``. ``materialize_alignment`` replays every rank's blocks
through ``AlignmentPayloadBackend.add_media`` in deterministic source-parquet
order (so each media's lazily-read raw bytes sweep their source row group once
instead of thrashing in media_id-hash order): a single running token cursor
assigns the offsets, so there is no cross-rank rebasing — the offsets are
recorded in the views, leaving the token file's physical order free.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from vision_tokenization.discrete.emu.token_layout import (
    STRUCTURE_TOKENS,
    resolve_token_ids_from_dir,
    vision_band,
)
from vision_tokenization.discrete.dpo_pairs import seq_lengths
from vision_tokenization.indexing.alignment.ingest import MARKER
from vision_tokenization.indexing.alignment.payload import (
    VIEW_SCHEMA,
    _alignment_media_refs,
    split_payload_rows,
    tokenized_views_dir,
)
from vision_tokenization.indexing.planning.tokenization_plan import IMAGE
from vision_tokenization.indexing.scanners.parquet_media_scan import load_media_inventory
from vision_tokenization.pipeline.output.spill import ComponentSpillReader
from vision_tokenization.pipeline.runtime.checkpoint import load_rank_manifests
from vision_tokenization.utils.json import json_dump_atomic, json_load

logger = logging.getLogger(__name__)

IMAGE_KIND = int(IMAGE)


class EncodeIncompleteError(RuntimeError):
    """A view row references a media that was never encoded.

    The alignment publish-stage completeness gate, raised from
    ``AlignmentPayloadBackend.finalize``. The executor only calls finalize on a
    clean loop, so this surfaces a genuine encode gap — never a masked loop
    error.
    """


def _fsync_file(path) -> None:
    """fsync a closed file's contents to stable storage (by path) before its
    atomic os.replace, matching the token-file fsync."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_tokenized_views(spill_dir, world_size: int) -> pa.Table:
    """Concatenate every rank's spilled tokenized pairs. Gate on the same rank count
    as the GPU spill — a missing shard means a rank's text pass did not complete, and
    must fail here, not later as a per-pair KeyError in the merge."""
    tdir = tokenized_views_dir(spill_dir)
    files = sorted(tdir.glob("rank_*.parquet"))
    if len(files) != world_size:
        raise FileNotFoundError(
            f"alignment merge: {len(files)} tokenized-view shards in {tdir}, expected {world_size} "
            f"(the engine text pass did not complete on every rank)")
    return pa.concat_tables([pq.read_table(f) for f in files])


def _tokenized_maps(table: pa.Table) -> tuple[dict, dict]:
    """``(prompt_id -> row index, prompt_id -> (prompt_text_len, chosen_len, rejected_len))``
    from the tokenized table. The lengths feed the store's seq lengths without
    materializing the id arrays; the index reorders the table to view order at write."""
    pids = table.column("prompt_id").to_pylist()
    ptl = pc.list_value_length(table.column("prompt_text_ids")).to_pylist()
    cl = pc.list_value_length(table.column("chosen_ids")).to_pylist()
    rl = pc.list_value_length(table.column("rejected_ids")).to_pylist()
    index, seq_inputs = {}, {}
    for i, (pid, ptl_i, cl_i, rl_i) in enumerate(zip(pids, ptl, cl, rl)):
        index[pid] = i
        seq_inputs[pid] = (ptl_i, cl_i, rl_i)
    if len(index) != len(pids):
        raise ValueError(
            f"alignment merge: {len(pids) - len(index)} duplicate prompt_id(s) in tokenized views — "
            f"the per-pair text<->vision join keys on prompt_id and requires them unique")
    return index, seq_inputs


class AlignmentPayloadBackend:
    """Write the final alignment ``views/`` + ``tokens/`` payload. Raw bytes stay
    in the scan's ``media_raw.blob``; each image ref carries a ``raw_offset`` /
    ``raw_length`` slice into it.

    The merge drives this writer: ranks GPU-encode the unique media and spill
    their stripped blocks; ``materialize_alignment`` replays them through
    ``add_media`` in source-parquet order. It does not create a content-addressed
    media store on disk.
    """

    def __init__(
        self,
        view_rows: List[dict],
        *,
        public_output_dir: str | Path,
        requested_validation_rows: int,
        tokenized: pa.Table,
        split_key: str = "prompt_id",
        seed: int = 42,
    ):
        self._public_dir = Path(public_output_dir)
        train_rows, validation_rows = split_payload_rows(
            view_rows,
            requested_validation_rows=requested_validation_rows,
            split_key=split_key,
            seed=seed,
        )
        self._rows_by_split: dict[str, list[dict]] = {"train": train_rows}
        if validation_rows:
            self._rows_by_split["validation"] = validation_rows

        self._media_splits: dict[str, set[str]] = {}
        self._image_ref_counts: dict[str, int] = {split: 0 for split in self._rows_by_split}
        for split, rows in self._rows_by_split.items():
            for row in rows:
                for media_id in _alignment_media_refs(row):
                    self._image_ref_counts[split] += 1
                    self._media_splits.setdefault(media_id, set()).add(split)

        self._token_files: dict[str, Any] = {}
        self._token_tmp: dict[str, Path] = {}
        self._token_final: dict[str, Path] = {}
        self._token_offsets: dict[str, int] = {}
        self._locations: dict[str, dict[str, dict]] = {
            split: {} for split in self._rows_by_split
        }
        self._tokenized = tokenized
        self._tokenized_index, self._seq_inputs = _tokenized_maps(tokenized)
        self.result: dict | None = None

    def open(self, output_dir: str, rank: int, writer_state: Optional[dict] = None) -> None:
        if rank != 0:
            raise ValueError("alignment payload backend is single-rank")
        if writer_state:
            raise ValueError("alignment payload backend is seal-at-end; resume is unsupported")

        self._public_dir.mkdir(parents=True, exist_ok=True)
        # No upfront wipe — a failed re-run keeps the prior store intact; the
        # old payload and manifest survive until finalize's atomic os.replace.
        for split in self._rows_by_split:
            token_rel = f"tokens/{split}-00000.i32"
            token_final = self._public_dir / token_rel
            token_final.parent.mkdir(parents=True, exist_ok=True)

            token_tmp = token_final.with_suffix(token_final.suffix + ".tmp")
            if token_tmp.exists():
                token_tmp.unlink()

            self._token_tmp[split] = token_tmp
            self._token_final[split] = token_final
            self._token_files[split] = open(token_tmp, "wb")
            self._token_offsets[split] = 0

    def add_media(
        self,
        media: Any,
        block: np.ndarray,
        *,
        resize_height: int,
        resize_width: int,
    ) -> None:
        """Materialize one media's stripped token block into each split it
        appears in (deduped per split). The single per-media write op, driven by
        the merge replaying each rank's spilled blocks in source-parquet order."""
        for split in sorted(self._media_splits.get(media.media_id, ())):
            if media.media_id in self._locations[split]:
                continue
            self._write_media_to_split(
                split, media, block,
                resize_height=resize_height, resize_width=resize_width,
            )

    def _write_media_to_split(
        self,
        split: str,
        media: Any,
        tokens: np.ndarray,
        *,
        resize_height: int,
        resize_width: int,
    ) -> None:
        token_offset = self._token_offsets[split]
        self._token_files[split].write(tokens.tobytes())
        self._token_offsets[split] += int(tokens.size)

        self._locations[split][media.media_id] = {
            "media_id": media.media_id,
            "width": int(media.width),
            "height": int(media.height),
            "resize_height": int(resize_height),
            "resize_width": int(resize_width),
            "token_offset": token_offset,
            "token_length": int(tokens.size),
            "raw_offset": int(media.raw_offset),
            "raw_length": int(media.raw_length_bytes),
            "raw_ext": str(media.raw_ext),
        }

    def _build_view_rows(self, split: str) -> tuple[list[dict], int]:
        locations = self._locations[split]
        view_rows = []
        dropped = 0
        for row in self._rows_by_split[split]:
            media_ids = _alignment_media_refs(row)
            missing = next((mid for mid in media_ids if mid not in locations), None)
            if missing is not None:
                dropped += 1
                logger.warning(
                    "dropping pair %s: media %s did not encode (undecodable source image)",
                    row.get("prompt_id", ""), missing[:12],
                )
                continue
            images = [dict(locations[mid]) for mid in media_ids]
            prompt_id = str(row.get("prompt_id", ""))
            media_tokens_total = sum(int(img["token_length"]) for img in images)
            si = self._seq_inputs.get(prompt_id)
            if si is None:
                raise KeyError(f"{prompt_id}: no tokenized text spilled for this pair")
            _, seq_chosen, seq_rejected = seq_lengths(si[0], media_tokens_total, si[1], si[2])

            prompt = row.get("prompt") or []
            chosen = str(row.get("chosen", ""))
            rejected = str(row.get("rejected", ""))
            text_chars = row.get("text_chars")
            if text_chars is None:
                text_chars = (
                    sum(len(str(m.get("content", ""))) for m in prompt)
                    + len(chosen)
                    + len(rejected)
                )
            view_rows.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "prompt_id": prompt_id,
                "text_chars": int(text_chars),
                "media_tokens_total": media_tokens_total,
                "seq_chosen_len": seq_chosen,
                "seq_rejected_len": seq_rejected,
                "images": images,
            })
        if self._rows_by_split[split] and not view_rows:
            raise EncodeIncompleteError(
                f"all {len(self._rows_by_split[split])} {split} pairs dropped — every "
                f"referenced media failed to encode (systematic, not sporadic corruption)"
            )
        return view_rows, dropped

    def _write_tokenized_split(self, prompt_ids: list[str], final_path: Path) -> None:
        """Persist this split's tokenized pieces (binidx input) in view-row order."""
        table = self._tokenized.take([self._tokenized_index[pid] for pid in prompt_ids])
        tmp = final_path.with_suffix(final_path.suffix + ".tmp")
        pq.write_table(table, tmp)
        _fsync_file(tmp)
        os.replace(tmp, final_path)

    def finalize(self) -> None:
        files: dict[str, int] = {}
        views: dict[str, list[dict]] = {}
        n_dropped_pairs = 0

        for split in self._rows_by_split:
            token_file = self._token_files.pop(split)
            token_file.flush()
            os.fsync(token_file.fileno())
            token_file.close()

            view_rel = f"views/{split}-00000.parquet"
            token_rel = f"tokens/{split}-00000.i32"
            view_final = self._public_dir / view_rel
            view_tmp = view_final.with_suffix(view_final.suffix + ".tmp")
            view_final.parent.mkdir(parents=True, exist_ok=True)

            split_rows, n_dropped = self._build_view_rows(split)
            n_dropped_pairs += n_dropped
            pq.write_table(
                pa.Table.from_pylist(split_rows, schema=VIEW_SCHEMA),
                view_tmp,
            )
            _fsync_file(view_tmp)
            os.replace(self._token_tmp[split], self._token_final[split])
            os.replace(view_tmp, view_final)

            tok_rel = f"views_tokenized/{split}-00000.parquet"
            tok_final = self._public_dir / tok_rel
            tok_final.parent.mkdir(parents=True, exist_ok=True)
            self._write_tokenized_split([vr["prompt_id"] for vr in split_rows], tok_final)

            for rel in (view_rel, token_rel, tok_rel):
                files[rel] = (self._public_dir / rel).stat().st_size
            spec = {
                "view": view_rel,
                "tokens": token_rel,
                "tokenized": tok_rel,
                "n_rows": len(split_rows),
                "n_image_refs": self._image_ref_counts[split],
                "n_media": len(self._locations[split]),
                "token_elements": self._token_offsets[split],
                "token_dtype": "<i4",
            }
            views[split] = [spec]

        if n_dropped_pairs:
            logger.warning(
                "dropped %d pair(s) whose source image failed to decode; published "
                "the rest", n_dropped_pairs,
            )
        self.result = {"files": files, "views": views, "n_dropped_pairs": n_dropped_pairs}


def _gate(spill_dir: Path) -> int:
    """Refuse to merge until every rank that participated has a completion
    manifest agreeing on world size and plan fingerprint — the same gate the
    bin/idx merge uses, minus the Megatron-shard file check (alignment reads the
    spills directly). The expected world size is the value the ranks record from
    their launch; deriving it here means the inline merge needs no world-size
    argument threaded from the launcher. Returns that world size."""
    manifests = load_rank_manifests(str(spill_dir))
    if not manifests:
        raise RuntimeError(f"alignment merge: no rank manifests in {spill_dir}")
    sizes = {int(m["world_size"]) for m in manifests}
    if len(sizes) != 1:
        raise RuntimeError(f"alignment merge: ranks disagree on world_size {sizes}")
    world_size = sizes.pop()
    present = {int(m["rank"]) for m in manifests}
    missing = set(range(world_size)) - present
    if missing:
        raise RuntimeError(
            f"alignment merge: rank manifest(s) {sorted(missing)} missing in {spill_dir}"
        )
    fingerprints = {repr(m.get("plan")) for m in manifests}
    if len(fingerprints) != 1:
        raise RuntimeError(
            f"alignment merge: ranks disagree on plan fingerprint: {fingerprints}"
        )
    return world_size


def _read_spilled_blocks(spill_dir: Path) -> dict:
    """document_id -> (stripped int32 block, resize_height, resize_width).

    Each unique media is spilled by exactly one rank (disjoint plan split), so a
    doc id appearing twice means that invariant broke and we fail loud.
    """
    blocks: dict[int, tuple] = {}
    for rank_dir in sorted(d for d in spill_dir.glob("rank_*") if d.is_dir()):
        if not (rank_dir / "_SUCCESS").exists():
            raise RuntimeError(f"alignment merge: {rank_dir} has no _SUCCESS marker")
        for comp_file in sorted(rank_dir.glob("components.*.parquet")):
            shard_id = int(comp_file.name.split(".")[1])
            for comp in pq.read_table(comp_file).to_pylist():
                if int(comp["kind"]) != IMAGE_KIND:
                    continue
                doc_id = int(comp["document_id"])
                if doc_id in blocks:
                    raise RuntimeError(
                        f"alignment merge: media doc {doc_id} spilled by >1 rank "
                        f"(disjoint-encode invariant broken)"
                    )
                block = ComponentSpillReader.load_tokens(
                    rank_dir, shard_id,
                    int(comp["token_offset"]), int(comp["token_length"]),
                )
                blocks[doc_id] = (block, int(comp["resize_height"]), int(comp["resize_width"]))
    return blocks


def materialize_alignment(
    spill_dir,
    *,
    inventory: list,
    view_rows: list,
    public_output_dir,
    requested_validation_rows: int,
    split_key: str = "prompt_id",
    seed: int = 42,
) -> dict:
    """Gate on rank completion, read every rank's spilled media blocks, and
    materialize the single views/tokens store. Returns the backend result."""
    spill_dir = Path(spill_dir)
    world_size = _gate(spill_dir)
    blocks = _read_spilled_blocks(spill_dir)
    tokenized = _read_tokenized_views(spill_dir, world_size)
    logger.info("alignment merge: %d media blocks, %d tokenized pairs across %d ranks",
                len(blocks), tokenized.num_rows, world_size)

    backend = AlignmentPayloadBackend(
        view_rows,
        public_output_dir=public_output_dir,
        requested_validation_rows=requested_validation_rows,
        tokenized=tokenized,
        split_key=split_key,
        seed=seed,
    )
    backend.open(str(public_output_dir), rank=0)
    # replay in source-parquet order for a stable, reproducible token layout
    order = sorted(
        (d for d in range(len(inventory)) if d in blocks),
        key=lambda d: (inventory[d].source_path, inventory[d].row_group,
                       inventory[d].row_index, inventory[d].image_index),
    )
    for doc_id in order:
        block, resize_height, resize_width = blocks[doc_id]
        backend.add_media(
            inventory[doc_id], block,
            resize_height=resize_height, resize_width=resize_width,
        )
    backend.finalize()
    return backend.result


def _token_layout(tokenizer_dir, tokenizer_config: dict) -> dict:
    """Manifest ``token_layout``, derived from the tokenizer's static files alone.

    Ids come from the snapshot's ``added_tokens``, bands from its
    ``omnimodal_config`` — consumers read ids from the manifest, never from
    literals, and publish never loads the tokenizer.
    """
    ids = resolve_token_ids_from_dir(
        tokenizer_dir, {"image_marker": MARKER, **STRUCTURE_TOKENS})
    vision_lo, vision_hi = vision_band(tokenizer_config)
    return {"image_marker": MARKER, "image_marker_id": ids.pop("image_marker"),
            **ids, "vision_lo": vision_lo, "vision_hi": vision_hi}


def publish_alignment_store(output_dir, *, keep_intermediates: bool = False) -> dict:
    """Alignment MERGE + publish (inline, CPU, torch-free): gate on rank
    completion, materialize the views/tokens store from every rank's spill, and
    write ``manifest.json`` LAST (the commit record). Raw bytes stay in the scan's
    ``media_raw.blob`` (indexed by per-image ``raw_offset``), which survives the
    intermediate cleanup. The scan's ``publish_meta.json`` carries the
    config-derived manifest fields, so the only argument is the store dir — exactly
    like the bin/idx ``merge``. Removes the spill + scan intermediates unless kept."""
    out = Path(output_dir)
    meta = json_load(out / "publish_meta.json")
    inventory = load_media_inventory(out / "media_unique.parquet")
    blob_path = out / "media_raw.blob"
    if not blob_path.exists():
        raise FileNotFoundError(
            f"{out}: alignment publish requires media_raw.blob "
            f"(set materialize_raw_store=true in the scan)"
        )
    view_rows = pq.read_table(out / "views.raw.parquet").to_pylist()
    n_pairs, n_unique_media = len(view_rows), len(inventory)
    n_skipped_media = meta["n_skipped_media"]

    result = materialize_alignment(
        out / "_spill",
        inventory=inventory,
        view_rows=view_rows,
        public_output_dir=out,
        requested_validation_rows=int(meta["val_rows"]),
    )

    tokenizer_path = Path(meta["tokenizer_path"])
    tokenizer_config = json_load(tokenizer_path / "tokenizer_config.json")
    tok_sha = hashlib.sha256((tokenizer_path / "tokenizer.json").read_bytes()).hexdigest()
    views = result["views"]
    json_dump_atomic({
        "schema_version": 3,
        "payload_format": "alignment_shard_local_v1",
        "tokenizer": {"path": meta["tokenizer_path"], "sha256": tok_sha},
        "vision_tokenizer": {"version": tokenizer_config["vision_tokenizer"]["type"],
                             "min_pixels": meta["tokenizer_min_pixels"],
                             "max_pixels": meta["tokenizer_max_pixels"]},
        "token_dtype": "<i4",
        "token_layout": _token_layout(tokenizer_path, tokenizer_config),
        "expected_min_model_vocab": max(
            m["offset"] + m["vocab_size"]
            for m in tokenizer_config["omnimodal_config"]["modalities"]),
        "views": views,
        "raw_blob": "media_raw.blob",
        "default_train_view": "train" if "train" in views else None,
        "default_validation_view": "validation" if "validation" in views else None,
        "files": {**result["files"], "media_raw.blob": blob_path.stat().st_size},
        "source_input": meta["source_input"],
        "n_pairs": n_pairs,
        "n_unique_media": n_unique_media,
        "n_skipped_media": n_skipped_media,
        "n_dropped_pairs": result["n_dropped_pairs"],
    }, out / "manifest.json")

    if not keep_intermediates:
        shutil.rmtree(out / "_spill", ignore_errors=True)
        shutil.rmtree(out / "_scan_build", ignore_errors=True)
        for name in ("scan.parquet", "media_unique.parquet", "row_media_refs.parquet",
                     "views.raw.parquet", "publish_meta.json"):
            (out / name).unlink(missing_ok=True)

    logger.info(
        "alignment publish: %d pairs, %d unique media (%d skipped, %d dropped) -> %s",
        n_pairs, n_unique_media, n_skipped_media, result["n_dropped_pairs"], out)
    return {**result, "output_dir": str(out),
            "n_pairs": n_pairs, "n_unique_media": n_unique_media,
            "n_skipped_media": n_skipped_media}


def stamp_dpo_section(output_dir, dpo: dict, *, delete_tokens: bool = False) -> None:
    """Register the dpo binidx outputs into the published manifest (schema 3 -> 4): record the
    ``dpo`` section, swap each split's deduped ``tokens/`` entry for its ``.bin/.idx/index``, and
    retire the now-redundant ``tokens/`` + ``views_tokenized/`` (both are inlined into the
    ``.bin``). The manifest's sole writer, beside ``publish_alignment_store``."""
    out = Path(output_dir)
    m = json_load(out / "manifest.json")
    m["schema_version"] = 4
    m["dpo"] = dpo
    for split, view_specs in m.get("views", {}).items():
        m["files"].pop(f"tokens/{split}-00000.i32", None)
        m["files"].pop(f"views_tokenized/{split}-00000.parquet", None)
        for spec in view_specs:
            spec.pop("tokens", None)
            spec.pop("tokenized", None)
    for spec in dpo["splits"].values():
        for rel in (spec["bin"], spec["idx"], spec["index"]):
            m["files"][rel] = (out / rel).stat().st_size
    json_dump_atomic(m, out / "manifest.json")
    if delete_tokens:
        shutil.rmtree(out / "tokens", ignore_errors=True)
        shutil.rmtree(out / "views_tokenized", ignore_errors=True)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: ``python -m vision_tokenization.pipeline.output.alignment_merge <store_dir>``."""
    parser = argparse.ArgumentParser(description="Merge alignment spills into the views/tokens store.")
    parser.add_argument("output_dir", help="Alignment store dir (holds _spill/, publish_meta.json, views.raw.parquet)")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="Keep _spill/ + scan artifacts after publish instead of removing them")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = publish_alignment_store(args.output_dir, keep_intermediates=args.keep_intermediates)
    print(f"published {result['n_pairs']:,} pairs, {result['n_unique_media']:,} media "
          f"({result['n_dropped_pairs']} dropped) -> {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
