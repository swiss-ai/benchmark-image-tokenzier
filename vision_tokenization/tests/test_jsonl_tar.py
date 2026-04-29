"""Tests for the generic JSONL+tar scanner and loader."""

from __future__ import annotations

import io
import json
import tarfile

import numpy as np
import pytest
from PIL import Image


def _make_image(width: int, height: int, color=(255, 0, 0)) -> Image.Image:
    return Image.new("RGB", (width, height), color)


def _image_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _create_content_tar(tar_path: str, members: dict[str, tuple[int, int]]):
    with tarfile.open(tar_path, "w") as tf:
        for name, (width, height) in members.items():
            data = _image_bytes(_make_image(width, height))
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_scan_jsonl_tar_dataset_resolves_global_tar_scope_and_prefix_strip(tmp_path):
    pytest.importorskip("orjson")

    from vision_tokenization.indexing.manifest import load_interleave_manifest
    from vision_tokenization.indexing.scanners.jsonl_tar import scan_jsonl_tar_dataset

    dataset_dir = tmp_path / "dataset"
    jsonl_dir = dataset_dir / "jsonl"
    tar_dir = dataset_dir / "images"
    jsonl_dir.mkdir(parents=True)
    tar_dir.mkdir()

    jsonl_path = jsonl_dir / "data.jsonl"
    tar_path = tar_dir / "germany.tar"
    manifest_path = tmp_path / "manifest.parquet"

    _create_content_tar(str(tar_path), {"germany/Q183.jpg": (32, 24)})

    row = {
        "conversations": [{"role": "user", "content": "Who is shown here?"}],
        "image": "Wikidata_images_v3/germany/Q183.jpg",
    }
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(row))
        fh.write("\n")

    scan_jsonl_tar_dataset(
        input_pattern=str(jsonl_path),
        output_manifest=str(manifest_path),
        image_field="image",
        tar_pattern="*.tar",
        tar_scope="global",
        tar_root="images",
        image_path_prefix_strip="Wikidata_images_v3/",
        num_workers=1,
    )

    table = load_interleave_manifest(manifest_path)
    assert len(table) == 1
    assert table.column("group_id").to_pylist() == [0]
    assert table.column("image_index").to_pylist() == [0]
    assert table.column("image_ref").to_pylist() == ["germany/Q183.jpg"]
    assert table.column("tar_path").to_pylist() == [str(tar_path)]


def test_jsonl_tar_loader_sft_loads_grouped_and_flat_texts(tmp_path):
    pytest.importorskip("orjson")

    from vision_tokenization.indexing.scanners.jsonl_tar import scan_jsonl_tar_dataset
    from vision_tokenization.pipeline.runtime.data import JSONLTarLoader, create_loader

    part_dir = tmp_path / "part00000"
    part_dir.mkdir()

    jsonl_path = part_dir / "data.jsonl"
    tar_path = part_dir / "imgs.tar"
    manifest_path = tmp_path / "manifest.parquet"

    _create_content_tar(
        str(tar_path),
        {
            "imgs/1.png": (40, 30),
            "imgs/2.png": (64, 48),
        },
    )

    conversations = [
        {"role": "user", "content": "<image><image> Compare these."},
        {"role": "assistant", "content": "They differ in size."},
    ]
    row = {
        "conversations": conversations,
        "image": ["./imgs/1.png", "./imgs/2.png"],
    }
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(row))
        fh.write("\n")

    scan_jsonl_tar_dataset(
        input_pattern=str(jsonl_path),
        output_manifest=str(manifest_path),
        image_field="image",
        tar_pattern="imgs.tar*",
        tar_scope="parent_dir",
        num_workers=1,
    )

    loader = create_loader(
        {
            "dataset_type": "jsonl_tar",
            "manifest_path": str(manifest_path),
            "mode": "sft",
            "text_column": "conversations",
            "max_open_files": 8,
        }
    )
    assert isinstance(loader, JSONLTarLoader)

    images, texts = loader.load_batch(
        np.array([0, 1], dtype=np.int64),
        group_slices=np.array([[0, 2]], dtype=np.int64),
    )
    flat_texts = loader.load_text_batch(np.array([0, 1], dtype=np.int64), group_slices=None)
    loader.close()

    assert len(images) == 2
    assert all(img is not None for img in images)
    assert len(texts) == 1
    assert texts[0] == conversations
    assert flat_texts == [conversations, conversations]


def test_create_loader_rejects_parser_backed_jsonl_tar(tmp_path):
    from vision_tokenization.pipeline.runtime.data import create_loader

    manifest_path = tmp_path / "manifest.parquet"
    manifest_path.write_bytes(b"")

    with pytest.raises(ValueError, match="do not support parser-backed loading yet"):
        create_loader(
            {
                "dataset_type": "jsonl_tar",
                "manifest_path": str(manifest_path),
                "mode": "sft",
                "text_column": "conversations",
                "parser": "conversation",
            }
        )
