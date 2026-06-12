from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from vision_tokenization.pipeline.output.media_store import (
    MediaStoreReader,
    MediaStoreWriter,
)


def _block(n, seed):
    rng = np.random.default_rng(seed)
    # plausible encapsulated block: img_start ... img_end
    body = rng.integers(131272, 262344, size=n - 2, dtype=np.int32)
    return np.concatenate([[131073], body, [131074]]).astype(np.int32)


def test_write_then_read_roundtrip(tmp_path):
    w = MediaStoreWriter(tmp_path / "media")
    b1, b2 = _block(100, 1), _block(64, 2)
    w.add("a" * 64, tokens=b1, raw=b"\xff\xd8raw1", resize_h=160, resize_w=160,
          kind="image", source="ds/sample-0", raw_ext="jpg")
    w.add("b" * 64, tokens=b2, raw=b"\xff\xd8raw22", resize_h=128, resize_w=128,
          kind="image", source="ds/sample-1", raw_ext="jpg")
    files = w.seal()  # returns {relpath: byte_size} for the manifest

    r = MediaStoreReader([tmp_path / "media"], token_dtype="<i4")
    np.testing.assert_array_equal(r.tokens("a" * 64), b1)
    np.testing.assert_array_equal(r.tokens("b" * 64), b2)
    assert r.raw("b" * 64) == b"\xff\xd8raw22"
    assert set(files) == {"media.000000.parquet", "tokens.000000.bin", "raw.000000.bin"}


def test_units_are_token_elements(tmp_path):
    w = MediaStoreWriter(tmp_path / "media")
    b1, b2 = _block(10, 3), _block(7, 4)
    w.add("a" * 64, tokens=b1, raw=b"x", resize_h=0, resize_w=0,
          kind="image", source="s", raw_ext="jpg")
    w.add("b" * 64, tokens=b2, raw=b"y", resize_h=0, resize_w=0,
          kind="image", source="s", raw_ext="jpg")
    w.seal()
    t = pq.read_table(tmp_path / "media" / "media.000000.parquet")
    rows = {r["media_id"]: r for r in t.to_pylist()}
    assert rows["b" * 64]["offset_elems"] == 10      # elements, not bytes (40)
    assert rows["b" * 64]["length_elems"] == 7
    assert rows["a" * 64]["raw_offset_bytes"] == 0
    assert rows["a" * 64]["raw_length_bytes"] == 1


def test_duplicate_media_id_rejected(tmp_path):
    w = MediaStoreWriter(tmp_path / "media")
    w.add("a" * 64, tokens=_block(8, 5), raw=b"x", resize_h=0, resize_w=0,
          kind="image", source="s", raw_ext="jpg")
    with pytest.raises(ValueError, match="duplicate media_id"):
        w.add("a" * 64, tokens=_block(8, 6), raw=b"x", resize_h=0, resize_w=0,
              kind="image", source="s", raw_ext="jpg")


def test_reader_refuses_unknown_dtype(tmp_path):
    w = MediaStoreWriter(tmp_path / "media")
    w.add("a" * 64, tokens=_block(8, 7), raw=b"x", resize_h=0, resize_w=0,
          kind="image", source="s", raw_ext="jpg")
    w.seal()
    with pytest.raises(ValueError, match="token_dtype"):
        MediaStoreReader([tmp_path / "media"], token_dtype="<i8")


def _backend_fixtures():
    from vision_tokenization.pipeline.runtime.checkpoint import WorkerStats

    inventory = [
        SimpleNamespace(media_id="a" * 64, raw=b"\xff\xd8raw1", source="s/0", raw_ext="jpg"),
        SimpleNamespace(media_id="b" * 64, raw=b"\xff\xd8raw22", source="s/1", raw_ext="png"),
    ]
    plan = SimpleNamespace(components=SimpleNamespace(source_ref=np.array([0, 1])))
    tokenizer = SimpleNamespace(bos_id=1, eos_id=2)
    return inventory, plan, tokenizer, WorkerStats()


def test_media_store_backend_strips_wrapper_and_seals(tmp_path):
    from vision_tokenization.pipeline.output.backend import MediaStoreBackend

    inventory, plan, tokenizer, stats = _backend_fixtures()
    blocks = [_block(10, 8), _block(7, 9)]
    rows = [torch.tensor(np.concatenate([[1], blk, [2]]), dtype=torch.long)
            for blk in blocks]

    backend = MediaStoreBackend(inventory)
    backend.open(str(tmp_path), rank=0)
    backend.write_batch(
        image_tokens=rows, texts=None, component_indices=np.array([0, 1]),
        group_slices=None, resize_height=160, resize_width=160,
        plan=plan, tokenizer=tokenizer, stats=stats)
    backend.finalize()

    r = MediaStoreReader([tmp_path / "media"], token_dtype="<i4")
    np.testing.assert_array_equal(r.tokens("a" * 64), blocks[0])  # BOS/EOS stripped
    np.testing.assert_array_equal(r.tokens("b" * 64), blocks[1])
    assert r.raw("b" * 64) == b"\xff\xd8raw22"
    assert stats.samples_processed == 2 and stats.tokens_generated == 17
    files = backend.completed_files()
    assert sum(f["sequences"] for f in files) == 2
    assert sum(f["tokens"] for f in files) == 17


def test_media_store_backend_refuses_unwrapped_block(tmp_path):
    from vision_tokenization.pipeline.output.backend import MediaStoreBackend

    inventory, plan, tokenizer, stats = _backend_fixtures()
    backend = MediaStoreBackend(inventory)
    backend.open(str(tmp_path), rank=0)
    with pytest.raises(ValueError, match="BOS..EOS"):
        backend.write_batch(
            image_tokens=[torch.tensor(_block(8, 10), dtype=torch.long)],
            texts=None, component_indices=np.array([0]), group_slices=None,
            resize_height=160, resize_width=160,
            plan=plan, tokenizer=tokenizer, stats=stats)
