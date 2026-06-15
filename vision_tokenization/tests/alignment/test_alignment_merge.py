from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from vision_tokenization.indexing.planning.tokenization_plan import IMAGE
from vision_tokenization.pipeline.output.alignment_merge import materialize_alignment
from vision_tokenization.pipeline.output.spill import ComponentSpillWriter
from vision_tokenization.pipeline.runtime.checkpoint import write_rank_manifest


def _media(media_id, raw, width, height, raw_ext, source):
    return SimpleNamespace(
        media_id=media_id, raw=raw, width=width, height=height,
        raw_ext=raw_ext, source=source,
        source_path=source, row_group=0, row_index=0, image_index=0,
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
    inventory = [
        _media("m1", b"raw-one", 64, 48, "png", "s0"),
        _media("m2", b"raw-two", 32, 32, "jpg", "s1"),
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

    raw = pq.read_table(pub / "raw" / "train-00000.parquet").to_pylist()
    assert [(r["media_id"], r["raw_bytes"]) for r in raw] == [("m1", b"raw-one"), ("m2", b"raw-two")]


def test_materialize_alignment_gates_on_missing_rank(tmp_path):
    spill = tmp_path / "spill"
    _spill_rank(spill, 0, 2, [(0, np.array([11, 12], dtype=np.int32), 128, 256)])
    with pytest.raises(RuntimeError, match="manifest"):
        materialize_alignment(
            spill, inventory=[_media("m1", b"r", 64, 48, "png", "s0")],
            view_rows=[_row("p0", ["m1"])],
            public_output_dir=tmp_path / "public", requested_validation_rows=0,
        )
