"""The materialized raw store is a torch-free flat ``np.memmap`` + ``raw_offset``,
read via ``LazyRawMedia.raw`` instead of a Megatron ``.bin``/``.idx`` (whose
builder/reader pull in torch and would poison the torch-free scan). This test
pins that the flat-memmap read is byte-identical to the canonical Megatron
``IndexedDataset`` reader for the same documents — so the substitute is faithful.
"""

import numpy as np
import pytest

from vision_tokenization.indexing.scanners.parquet_media_scan import LazyRawMedia


def test_raw_blob_memmap_matches_indexed_dataset(tmp_path):
    pytest.importorskip("megatron.core.datasets.indexed_dataset")
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    from vision_tokenization.formats.megatron import IndexedDatasetBuilder
    from vision_tokenization.pipeline.runtime.checkpoint import finalize_shard_writer

    rng = np.random.default_rng(0)
    blobs = [rng.integers(0, 256, size=int(n), dtype=np.uint8).tobytes()
             for n in rng.integers(1, 4096, size=64)]

    # canonical: one uint8 document per blob in a Megatron .bin/.idx
    pre = str(tmp_path / "ref")
    builder = IndexedDatasetBuilder(pre + ".bin.tmp", dtype=np.uint8)
    for b in blobs:
        builder.add_item(np.frombuffer(b, dtype=np.uint8))
        builder.end_document()
    finalize_shard_writer(builder, pre + ".bin.tmp", pre + ".idx.tmp", pre + ".bin", pre + ".idx")
    ds = IndexedDataset(pre)

    # ours: flat blob + offsets, read via LazyRawMedia.raw
    blob_path = tmp_path / "media_raw.blob"
    offsets, cursor = [], 0
    with open(blob_path, "wb") as fh:
        for b in blobs:
            fh.write(b)
            offsets.append(cursor)
            cursor += len(b)
    mm = np.memmap(blob_path, dtype=np.uint8, mode="r")

    for i, b in enumerate(blobs):
        ref = np.asarray(ds[i]).tobytes()
        mine = LazyRawMedia(
            media_id="m", raw_length_bytes=len(b), raw_ext="jpg", width=1, height=1,
            source="s", source_path="p", row_group=0, row_index=0, image_index=0,
            raw_offset=offsets[i], blob=mm,
        ).raw
        assert mine == b, f"flat memmap != original at doc {i}"
        assert mine == ref, f"flat memmap != IndexedDataset at doc {i}"
