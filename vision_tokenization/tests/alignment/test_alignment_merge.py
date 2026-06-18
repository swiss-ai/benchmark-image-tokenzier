from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vision_tokenization.indexing.alignment.payload import TOKENIZED_SCHEMA, tokenized_views_dir
from vision_tokenization.indexing.planning.tokenization_plan import IMAGE
from vision_tokenization.pipeline.output.alignment_merge import materialize_alignment
from vision_tokenization.pipeline.output.spill import ComponentSpillWriter
from vision_tokenization.pipeline.runtime.checkpoint import write_rank_manifest


def _tok_row(prompt_id, n_images, ptl=2, cl=1, rl=2):
    return {
        "prompt_id": prompt_id, "prompt_text_ids": list(range(ptl)),
        "image_insert_positions": [0] * n_images,
        "chosen_ids": list(range(cl)), "rejected_ids": list(range(rl)), "enable_thinking": False,
    }


def _spill_tokenized(spill_dir, world_size, rows):
    """Rank 0 carries the pairs; the rest spill empty shards, as a real run does."""
    d = tokenized_views_dir(spill_dir)
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=TOKENIZED_SCHEMA), d / "rank_0000.parquet")
    for rank in range(1, world_size):
        pq.write_table(pa.Table.from_pylist([], schema=TOKENIZED_SCHEMA), d / f"rank_{rank:04d}.parquet")


def _media(media_id, raw, width, height, raw_ext, source, raw_offset=0):
    return SimpleNamespace(
        media_id=media_id, raw=raw, width=width, height=height,
        raw_ext=raw_ext, source=source,
        source_path=source, row_group=0, row_index=0, image_index=0,
        raw_offset=raw_offset, raw_length_bytes=len(raw),
    )


def _row(prompt_id, refs):
    return {
        "prompt": [{"role": "user", "content": "<|image|>\nq"}],
        "chosen": "good", "rejected": "bad",
        "prompt_media_refs": refs, "chosen_media_refs": [], "rejected_media_refs": [],
        "prompt_id": prompt_id, "text_chars": 16, "media_tokens_total": 0,
    }


def _spill_rank(spill_dir, rank, world_size, items):
    """items: list of (doc_id, int32 block, resize_h, resize_w)."""
    w = ComponentSpillWriter(str(spill_dir), rank=rank)
    w.open()
    for doc_id, block, rh, rw in items:
        w.add_component(
            document_id=doc_id, component_index=0, kind=int(IMAGE),
            tokens=block, resize_height=rh, resize_width=rw,
        )
    w.finalize()
    (spill_dir / f"rank_{rank:04d}" / "_SUCCESS").touch()
    write_rank_manifest(str(spill_dir), rank, world_size, {"fp": 1}, backend="spill", files=[])


def test_materialize_alignment_two_ranks_rebases_cross_rank_pair(tmp_path):
    """A pair references media encoded by two DIFFERENT ranks; the merge must
    concatenate in doc-id order and resolve each image to the right global
    offset in the merged token stream."""
    spill = tmp_path / "spill"
    _spill_rank(spill, 0, 2, [(0, np.array([11, 12], dtype=np.int32), 128, 256)])
    _spill_rank(spill, 1, 2, [(1, np.array([21, 22, 23], dtype=np.int32), 128, 256)])
    _spill_tokenized(spill, 2, [_tok_row("p0", 2)])
    inventory = [
        _media("m1", b"raw-one", 64, 48, "png", "s0", raw_offset=0),
        _media("m2", b"raw-two", 32, 32, "jpg", "s1", raw_offset=7),
    ]
    view_rows = [_row("p0", ["m1", "m2"])]

    materialize_alignment(
        spill, inventory=inventory, view_rows=view_rows,
        public_output_dir=tmp_path / "public", requested_validation_rows=0,
    )

    pub = tmp_path / "public"
    tokens = np.fromfile(pub / "tokens" / "train-00000.i32", dtype="<i4")
    np.testing.assert_array_equal(tokens, np.array([11, 12, 21, 22, 23], dtype=np.int32))

    view = pq.read_table(pub / "views" / "train-00000.parquet").to_pylist()[0]
    imgs = view["images"]
    assert [im["media_id"] for im in imgs] == ["m1", "m2"]
    assert (imgs[0]["token_offset"], imgs[0]["token_length"]) == (0, 2)
    assert (imgs[1]["token_offset"], imgs[1]["token_length"]) == (2, 3)

    assert not (pub / "raw").exists()
    assert [(im["raw_offset"], im["raw_length"], im["raw_ext"]) for im in imgs] == [
        (0, 7, "png"), (7, 7, "jpg"),
    ]

    # seq lengths = prompt_text(2) + vision(2+3) + chosen(1) / rejected(2)
    assert (view["seq_chosen_len"], view["seq_rejected_len"]) == (8, 9)
    tok = pq.read_table(pub / "views_tokenized" / "train-00000.parquet").to_pylist()
    assert [r["prompt_id"] for r in tok] == ["p0"]


def test_materialize_alignment_gates_on_missing_rank(tmp_path):
    spill = tmp_path / "spill"
    _spill_rank(spill, 0, 2, [(0, np.array([11, 12], dtype=np.int32), 128, 256)])
    with pytest.raises(RuntimeError, match="manifest"):
        materialize_alignment(
            spill, inventory=[_media("m1", b"r", 64, 48, "png", "s0")],
            view_rows=[_row("p0", ["m1"])],
            public_output_dir=tmp_path / "public", requested_validation_rows=0,
        )
