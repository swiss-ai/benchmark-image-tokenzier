import numpy as np
import pyarrow.parquet as pq
import pytest

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
