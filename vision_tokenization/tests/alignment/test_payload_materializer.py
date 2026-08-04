from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vision_tokenization.indexing.alignment.payload import TOKENIZED_SCHEMA
from vision_tokenization.pipeline.output.alignment_merge import (
    AlignmentPayloadBackend,
    EncodeIncompleteError,
)


def _tok_row(prompt_id, n_images, ptl=2, cl=1, rl=2):
    return {
        "prompt_id": prompt_id,
        "prompt_text_ids": list(range(ptl)),
        "image_insert_positions": [0] * n_images,
        "chosen_ids": list(range(cl)),
        "rejected_ids": list(range(rl)),
        "enable_thinking": False,
    }


def _tok_table(rows):
    return pa.Table.from_pylist(rows, schema=TOKENIZED_SCHEMA)


def _media(media_id, raw, width, height, raw_ext, source, raw_offset=0):
    return SimpleNamespace(
        media_id=media_id, raw=raw, width=width, height=height,
        raw_ext=raw_ext, source=source,
        raw_offset=raw_offset, raw_length_bytes=len(raw),
    )


def _row(prompt_id, refs):
    return {
        "prompt": [{"role": "user", "content": "<|image|>\nquestion"}],
        "chosen": "good",
        "rejected": "bad",
        "prompt_media_refs": refs,
        "chosen_media_refs": [],
        "rejected_media_refs": [],
        "prompt_id": prompt_id,
        "text_chars": 16,
        "media_tokens_total": 0,
    }


def test_alignment_payload_backend_writes_final_payload_without_media_staging(tmp_path):
    public = tmp_path / "public"
    work = tmp_path / "work"
    rows = [_row("p0", ["m1", "m1", "m2"])]
    inventory = [
        _media("m1", b"raw-one", 64, 48, "png", "s0", raw_offset=0),
        _media("m2", b"raw-two", 32, 32, "jpg", "s1", raw_offset=7),
    ]

    backend = AlignmentPayloadBackend(
        rows,
        public_output_dir=public,
        requested_validation_rows=0,
        tokenized=_tok_table([_tok_row("p0", 3)]),
    )
    backend.open(str(work), rank=0)
    backend.add_media(inventory[0], np.array([11, 12], dtype=np.int32),
                      resize_height=128, resize_width=256)
    backend.add_media(inventory[1], np.array([21, 22, 23], dtype=np.int32),
                      resize_height=128, resize_width=256)
    backend.finalize()

    assert not (public / "media").exists()
    assert not (work / "media").exists()
    assert sorted(p.name for p in public.iterdir()) == ["tokens", "views", "views_tokenized"]

    tokens = np.fromfile(public / "tokens" / "train-00000.i32", dtype="<i4")
    np.testing.assert_array_equal(tokens, np.array([11, 12, 21, 22, 23], dtype=np.int32))

    view_row = pq.read_table(public / "views" / "train-00000.parquet").to_pylist()[0]
    assert view_row["images"] == [
        {
            "media_id": "m1",
            "width": 64,
            "height": 48,
            "resize_height": 128,
            "resize_width": 256,
            "token_offset": 0,
            "token_length": 2,
            "raw_offset": 0,
            "raw_length": 7,
            "raw_ext": "png",
        },
        {
            "media_id": "m1",
            "width": 64,
            "height": 48,
            "resize_height": 128,
            "resize_width": 256,
            "token_offset": 0,
            "token_length": 2,
            "raw_offset": 0,
            "raw_length": 7,
            "raw_ext": "png",
        },
        {
            "media_id": "m2",
            "width": 32,
            "height": 32,
            "resize_height": 128,
            "resize_width": 256,
            "token_offset": 2,
            "token_length": 3,
            "raw_offset": 7,
            "raw_length": 7,
            "raw_ext": "jpg",
        },
    ]
    assert view_row["media_tokens_total"] == 7
    assert view_row["seq_chosen_len"] == 2 + 7 + 1
    assert view_row["seq_rejected_len"] == 2 + 7 + 2

    tok_views = pq.read_table(public / "views_tokenized" / "train-00000.parquet").to_pylist()
    assert [r["prompt_id"] for r in tok_views] == ["p0"]

    assert backend.result["views"]["train"][0]["n_rows"] == 1
    assert backend.result["views"]["train"][0]["n_image_refs"] == 3
    assert backend.result["views"]["train"][0]["n_media"] == 2


def test_alignment_payload_backend_keeps_prior_output_on_open(tmp_path):
    """A re-run must not wipe prior output up front; old files survive open()
    and are only atomically replaced at finalize."""
    public = tmp_path / "public"
    (public / "tokens").mkdir(parents=True)
    stale = public / "tokens" / "train-00000.i32"
    stale.write_bytes(b"OLD")
    backend = AlignmentPayloadBackend(
        [_row("p0", ["m1"])],
        public_output_dir=public, requested_validation_rows=0,
        tokenized=_tok_table([_tok_row("p0", 1)]),
    )
    backend.open(str(tmp_path / "work"), rank=0)
    assert stale.read_bytes() == b"OLD"


def test_alignment_payload_backend_drops_pair_with_undecodable_media(tmp_path):
    """An un-encoded media (source image failed to decode) drops only its pair;
    the rest of the store still publishes."""
    inventory = [
        _media("m1", b"raw-one", 64, 48, "png", "s0"),
        _media("m2", b"raw-two", 32, 32, "jpg", "s1"),
    ]
    backend = AlignmentPayloadBackend(
        [_row("keep", ["m1"]), _row("drop", ["m2"])],
        public_output_dir=tmp_path / "public", requested_validation_rows=0,
        tokenized=_tok_table([_tok_row("keep", 1)]),
    )
    backend.open(str(tmp_path / "work"), rank=0)
    backend.add_media(inventory[0], np.array([11, 12], dtype=np.int32),
                      resize_height=128, resize_width=256)
    backend.finalize()
    assert backend.result["n_dropped_pairs"] == 1
    assert backend.result["views"]["train"][0]["n_rows"] == 1


def test_alignment_payload_backend_finalize_raises_when_all_pairs_dropped(tmp_path):
    """If every pair references an un-encoded media (systematic, not sporadic),
    finalize fails loud rather than publishing an empty store."""
    backend = AlignmentPayloadBackend(
        [_row("p0", ["m2"])],
        public_output_dir=tmp_path / "public", requested_validation_rows=0,
        tokenized=_tok_table([]),
    )
    backend.open(str(tmp_path / "work"), rank=0)
    with pytest.raises(EncodeIncompleteError, match="dropped"):
        backend.finalize()
