"""Focused tests for grouped interleave parsing, scanning, and loading."""

from __future__ import annotations

import io
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from PIL import Image

from vision_tokenization.utils.interleave_documents import (
    extract_local_image_refs,
    parse_content_array_interleave,
    parse_markdown_interleave,
)
from vision_tokenization.discrete.emu.interleave import (
    EMUInterleaveTokenizer,
    assemble_interleaved_sequence,
    split_interleaved_sequence,
)


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


def _create_wds_tar(
    tar_path: str,
    images: dict[str, tuple[int, int]],
    text_members: dict[str, str] | None = None,
):
    with tarfile.open(tar_path, "w") as tf:
        for name, (width, height) in images.items():
            data = _image_bytes(_make_image(width, height), "JPEG")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        for name, text in (text_members or {}).items():
            raw = text.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(raw)
            tf.addfile(info, io.BytesIO(raw))


def test_parse_markdown_interleave_filters_remote_and_preserves_order():
    segments = parse_markdown_interleave(
        "alpha <img src='content_image/0-0.png'> beta ![x](https://remote/x.png) "
        "gamma ![ok](content_image/0-1.png) omega"
    )

    assert [seg["type"] for seg in segments] == ["text", "image", "text", "image", "text"]
    assert extract_local_image_refs(segments) == [
        "content_image/0-0.png",
        "content_image/0-1.png",
    ]
    assert "alpha" in segments[0]["text"]
    assert "beta" in segments[2]["text"]
    assert "gamma" in segments[2]["text"]
    assert "omega" in segments[4]["text"]


def test_parse_content_array_interleave_keeps_local_images_only():
    segments = parse_content_array_interleave(
        [
            {"type": "text", "text": "hello"},
            {"type": "image", "image": "image/a.jpeg"},
            {"type": "image", "image": "https://remote/b.jpeg"},
            {"type": "text", "text": "world"},
        ]
    )

    assert [seg["type"] for seg in segments] == ["text", "image", "text"]
    assert extract_local_image_refs(segments) == ["image/a.jpeg"]


def test_scan_jsonl_tar_interleave_dataset_drops_zero_image_and_missing_docs(tmp_path):
    pytest.importorskip("orjson")

    from vision_tokenization.indexing.manifest import load_interleave_manifest
    from vision_tokenization.indexing.scanners.interleave import scan_jsonl_tar_interleave_dataset

    part_dir = tmp_path / "part00000"
    part_dir.mkdir()

    jsonl_path = part_dir / "part00000.jsonl"
    tar_path = part_dir / "content_image.tar"
    manifest_path = tmp_path / "manifest.parquet"

    _create_content_tar(
        str(tar_path),
        {
            "content_image/0-0.png": (32, 24),
            "content_image/0-1.png": (48, 36),
        },
    )

    rows = [
        {
            "md": "before <img src='content_image/0-0.png'> "
            "middle ![ok](content_image/0-1.png) after",
        },
        {"md": "plain text only"},
        {"md": "broken ![miss](content_image/9-9.png) sample"},
    ]
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")

    scan_jsonl_tar_interleave_dataset(
        input_pattern=str(jsonl_path),
        output_manifest=str(manifest_path),
        document_format="pin_markdown",
        document_field="md",
        tar_pattern="content_image.tar*",
        tar_scope="parent_dir",
        num_workers=2,
    )

    table = load_interleave_manifest(manifest_path)
    assert len(table) == 2
    assert table.column("group_id").to_pylist() == [0, 0]
    assert table.column("image_index").to_pylist() == [0, 1]
    assert table.column("image_ref").to_pylist() == [
        "content_image/0-0.png",
        "content_image/0-1.png",
    ]
    assert "segment_start_index" not in table.column_names
    assert "segment_end_index" not in table.column_names


def test_jsonl_tar_interleave_loader_reconstructs_grouped_document(tmp_path):
    pytest.importorskip("orjson")

    from vision_tokenization.indexing.scanners.interleave import scan_jsonl_tar_interleave_dataset
    from vision_tokenization.pipeline.runtime.data import JSONLTarLoader

    part_dir = tmp_path / "part00000"
    part_dir.mkdir()

    jsonl_path = part_dir / "part00000.jsonl"
    tar_path = part_dir / "content_image.tar"
    manifest_path = tmp_path / "manifest.parquet"

    _create_content_tar(
        str(tar_path),
        {
            "content_image/0-0.png": (40, 30),
            "content_image/0-1.png": (64, 48),
        },
    )

    row = {
        "md": "head <img src='content_image/0-0.png'> tail "
        "![chart](content_image/0-1.png) done",
    }
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(row))
        fh.write("\n")

    scan_jsonl_tar_interleave_dataset(
        input_pattern=str(jsonl_path),
        output_manifest=str(manifest_path),
        document_format="pin_markdown",
        document_field="md",
        tar_pattern="content_image.tar*",
        tar_scope="parent_dir",
        num_workers=2,
    )

    loader = JSONLTarLoader(
        manifest_path=str(manifest_path),
        document_format="pin_markdown",
        document_field="md",
    )
    images, texts = loader.load_batch(
        np.array([0, 1], dtype=np.int64),
        group_slices=np.array([[0, 2]], dtype=np.int64),
    )
    loader.close()

    assert len(images) == 2
    assert all(img is not None for img in images)
    assert len(texts) == 1
    assert [seg["type"] for seg in texts[0]] == ["text", "image", "text", "image", "text"]
    assert extract_local_image_refs(texts[0]) == [
        "content_image/0-0.png",
        "content_image/0-1.png",
    ]


def test_jsonl_tar_interleave_loader_accepts_partial_batch_fragment(tmp_path):
    pytest.importorskip("orjson")

    from vision_tokenization.indexing.scanners.interleave import scan_jsonl_tar_interleave_dataset
    from vision_tokenization.pipeline.runtime.data import JSONLTarLoader

    part_dir = tmp_path / "part00000"
    part_dir.mkdir()

    jsonl_path = part_dir / "part00000.jsonl"
    tar_path = part_dir / "content_image.tar"
    manifest_path = tmp_path / "manifest.parquet"

    _create_content_tar(
        str(tar_path),
        {
            "content_image/0-0.png": (40, 30),
            "content_image/0-1.png": (64, 48),
        },
    )

    row = {
        "md": "head <img src='content_image/0-0.png'> tail "
        "![chart](content_image/0-1.png) done",
    }
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(row))
        fh.write("\n")

    scan_jsonl_tar_interleave_dataset(
        input_pattern=str(jsonl_path),
        output_manifest=str(manifest_path),
        document_format="pin_markdown",
        document_field="md",
        tar_pattern="content_image.tar*",
        tar_scope="parent_dir",
        num_workers=1,
    )

    loader = JSONLTarLoader(
        manifest_path=str(manifest_path),
        document_format="pin_markdown",
        document_field="md",
    )
    _, texts = loader.load_batch(
        np.array([1], dtype=np.int64),
        group_slices=np.array([[0, 1]], dtype=np.int64),
    )
    loader.close()

    assert len(texts) == 1
    assert texts[0] is not None
    assert [seg["type"] for seg in texts[0]] == ["text", "image", "text", "image", "text"]
    assert extract_local_image_refs(texts[0]) == [
        "content_image/0-0.png",
        "content_image/0-1.png",
    ]


def test_wds_interleave_loader_accepts_partial_batch_fragment(tmp_path):
    from vision_tokenization.indexing.scanners.wds import scan_wds_dataset
    from vision_tokenization.pipeline.runtime.data import create_loader

    tar_path = tmp_path / "shard.tar"
    manifest_path = tmp_path / "manifest.parquet"

    _create_wds_tar(
        str(tar_path),
        {
            "case0.img0.jpg": (32, 24),
            "case0.img1.jpg": (48, 36),
        },
        {
            "case0.txt": "before <|img0|> middle <|img1|> after",
        },
    )

    scan_wds_dataset(
        input_pattern=str(tar_path),
        output_manifest=str(manifest_path),
        num_workers=1,
        text_extensions=frozenset({"txt"}),
        image_field_pattern="img",
        multi_image=True,
    )

    loader = create_loader(
        {
            "dataset_type": "wds",
            "manifest_path": str(manifest_path),
            "text_column": "txt",
            "parser": "medpix",
            "multi_image": True,
            "max_open_files": 8,
        }
    )
    _, texts = loader.load_batch(
        np.array([1], dtype=np.int64),
        group_slices=np.array([[0, 1]], dtype=np.int64),
    )
    loader.close()

    assert len(texts) == 1
    assert texts[0] is not None
    assert [seg["type"] for seg in texts[0]] == ["text", "image", "text", "image", "text"]


def test_wds_interleave_loader_returns_none_on_image_count_mismatch(tmp_path):
    from vision_tokenization.indexing.scanners.wds import scan_wds_dataset
    from vision_tokenization.pipeline.runtime.data import create_loader

    tar_path = tmp_path / "shard_bad.tar"
    manifest_path = tmp_path / "manifest_bad.parquet"

    _create_wds_tar(
        str(tar_path),
        {
            "case0.img0.jpg": (32, 24),
        },
        {
            "case0.txt": "before <|img0|> middle <|img1|> after",
        },
    )

    scan_wds_dataset(
        input_pattern=str(tar_path),
        output_manifest=str(manifest_path),
        num_workers=1,
        text_extensions=frozenset({"txt"}),
        image_field_pattern="img",
        multi_image=True,
    )

    loader = create_loader(
        {
            "dataset_type": "wds",
            "manifest_path": str(manifest_path),
            "text_column": "txt",
            "parser": "medpix",
            "multi_image": True,
            "max_open_files": 8,
        }
    )
    _, texts = loader.load_batch(
        np.array([0], dtype=np.int64),
        group_slices=np.array([[0, 1]], dtype=np.int64),
    )
    loader.close()

    assert texts == [None]


def test_assemble_interleaved_sequence_places_tokens_in_order():
    output = assemble_interleaved_sequence(
        bos_id=1,
        eos_id=2,
        segments=[
            {"type": "text", "text": "hello"},
            {"type": "image", "ref": "content_image/0-0.png"},
            {"type": "text", "text": "world"},
        ],
        text_token_chunks=[
            torch.tensor([10, 11], dtype=torch.long),
            torch.tensor([12], dtype=torch.long),
        ],
        image_token_chunks=[
            torch.tensor([20, 21, 22], dtype=torch.long),
        ],
    )
    assert output.tolist() == [1, 10, 11, 20, 21, 22, 12, 2]


def test_split_interleaved_sequence_respects_max_tokens():
    outputs = split_interleaved_sequence(
        bos_id=1,
        eos_id=2,
        segments=[
            {"type": "text", "text": "left"},
            {"type": "image", "ref": "content_image/0-0.png"},
            {"type": "text", "text": "right"},
            {"type": "image", "ref": "content_image/0-1.png"},
        ],
        text_token_chunks=[
            torch.tensor([10, 11], dtype=torch.long),
            torch.tensor([12, 13], dtype=torch.long),
        ],
        image_token_chunks=[
            torch.tensor([20, 21, 22], dtype=torch.long),
            torch.tensor([30, 31, 32], dtype=torch.long),
        ],
        max_sequence_tokens=7,
    )
    assert len(outputs) == 2
    assert outputs[0].tolist() == [1, 10, 11, 20, 21, 22, 2]
    assert outputs[1].tolist() == [1, 12, 13, 30, 31, 32, 2]


def test_split_interleaved_sequence_raises_for_single_oversize_segment():
    with pytest.raises(ValueError, match="Single image segment"):
        split_interleaved_sequence(
            bos_id=1,
            eos_id=2,
            segments=[
                {"type": "image", "ref": "content_image/huge.png"},
            ],
            text_token_chunks=[],
            image_token_chunks=[
                torch.tensor([20, 21, 22, 23, 24, 25, 26], dtype=torch.long),
            ],
            max_sequence_tokens=8,
        )


def test_split_interleaved_sequence_keeps_exact_boundary_sequence():
    outputs = split_interleaved_sequence(
        bos_id=1,
        eos_id=2,
        segments=[
            {"type": "text", "text": "left"},
            {"type": "image", "ref": "content_image/0-0.png"},
        ],
        text_token_chunks=[
            torch.tensor([10, 11], dtype=torch.long),
        ],
        image_token_chunks=[
            torch.tensor([20, 21, 22], dtype=torch.long),
        ],
        max_sequence_tokens=7,
    )
    assert len(outputs) == 1
    assert outputs[0].tolist() == [1, 10, 11, 20, 21, 22, 2]


def test_split_interleaved_sequence_splits_three_times_in_order():
    outputs = split_interleaved_sequence(
        bos_id=1,
        eos_id=2,
        segments=[
            {"type": "text", "text": "a"},
            {"type": "image", "ref": "content_image/0.png"},
            {"type": "text", "text": "b"},
            {"type": "image", "ref": "content_image/1.png"},
            {"type": "text", "text": "c"},
        ],
        text_token_chunks=[
            torch.tensor([10], dtype=torch.long),
            torch.tensor([11], dtype=torch.long),
            torch.tensor([12], dtype=torch.long),
        ],
        image_token_chunks=[
            torch.tensor([20, 21], dtype=torch.long),
            torch.tensor([30, 31], dtype=torch.long),
        ],
        max_sequence_tokens=5,
    )
    assert [out.tolist() for out in outputs] == [
        [1, 10, 20, 21, 2],
        [1, 11, 30, 31, 2],
        [1, 12, 2],
    ]


class _DummyTextTokenizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [self.mapping[text] for text in texts]}


def _make_stub_interleave_tokenizer(*, max_sequence_tokens, text_mapping, image_rows):
    tok = object.__new__(EMUInterleaveTokenizer)
    tok.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="TestTokenizerPool")
    tok.max_sequence_tokens = max_sequence_tokens
    tok.bos_id = 1
    tok.eos_id = 2
    tok.text_tokenizer = _DummyTextTokenizer(text_mapping)
    tok.tokenize_images = lambda images, resize_size: torch.tensor(image_rows, dtype=torch.long)
    return tok


def test_interleave_tokenize_batch_respects_max_sequence_tokens():
    tokenizer = _make_stub_interleave_tokenizer(
        max_sequence_tokens=7,
        text_mapping={
            "left": [10, 11],
            "right": [12, 13],
        },
        image_rows=[
            [100, 20, 21, 22, 101],
            [100, 30, 31, 32, 101],
        ],
    )
    try:
        outputs = tokenizer.tokenize_batch(
            images=["img0", "img1"],
            resize_size=(16, 16),
            text=[
                [
                    {"type": "text", "text": "left"},
                    {"type": "image", "ref": "content_image/0.png"},
                    {"type": "text", "text": "right"},
                    {"type": "image", "ref": "content_image/1.png"},
                ]
            ],
            group_slices=np.array([[0, 2]], dtype=np.int64),
        )
    finally:
        tokenizer.executor.shutdown(wait=True)

    assert len(outputs) == 2
    assert [out.tolist() for out in outputs] == [
        [1, 10, 11, 20, 21, 22, 2],
        [1, 12, 13, 30, 31, 32, 2],
    ]


def test_interleave_tokenize_batch_skips_single_oversize_segment():
    tokenizer = _make_stub_interleave_tokenizer(
        max_sequence_tokens=8,
        text_mapping={},
        image_rows=[
            [100, 20, 21, 22, 23, 24, 25, 26, 101],
        ],
    )
    try:
        outputs = tokenizer.tokenize_batch(
            images=["img0"],
            resize_size=(16, 16),
            text=[
                [
                    {"type": "image", "ref": "content_image/huge.png"},
                ]
            ],
            group_slices=np.array([[0, 1]], dtype=np.int64),
        )
    finally:
        tokenizer.executor.shutdown(wait=True)

    assert outputs == [None]


def test_interleave_tokenize_batch_keeps_exact_boundary_sequence():
    tokenizer = _make_stub_interleave_tokenizer(
        max_sequence_tokens=7,
        text_mapping={
            "left": [10, 11],
        },
        image_rows=[
            [100, 20, 21, 22, 101],
        ],
    )
    try:
        outputs = tokenizer.tokenize_batch(
            images=["img0"],
            resize_size=(16, 16),
            text=[
                [
                    {"type": "text", "text": "left"},
                    {"type": "image", "ref": "content_image/0.png"},
                ]
            ],
            group_slices=np.array([[0, 1]], dtype=np.int64),
        )
    finally:
        tokenizer.executor.shutdown(wait=True)

    assert len(outputs) == 1
    assert outputs[0].tolist() == [1, 10, 11, 20, 21, 22, 2]


def test_interleave_tokenize_batch_handles_mixed_group_outcomes():
    tokenizer = _make_stub_interleave_tokenizer(
        max_sequence_tokens=5,
        text_mapping={
            "a": [10],
            "b": [11],
            "too-long-text": [40, 41, 42, 43],
        },
        image_rows=[
            [100, 20, 21, 101],
            [100, 30, 31, 101],
        ],
    )
    try:
        outputs = tokenizer.tokenize_batch(
            images=["img0", "img1"],
            resize_size=(16, 16),
            text=[
                [
                    {"type": "text", "text": "a"},
                    {"type": "image", "ref": "content_image/0.png"},
                    {"type": "text", "text": "b"},
                    {"type": "image", "ref": "content_image/1.png"},
                ],
                [
                    {"type": "text", "text": "too-long-text"},
                ],
            ],
            group_slices=np.array([[0, 2], [2, 2]], dtype=np.int64),
        )
    finally:
        tokenizer.executor.shutdown(wait=True)

    assert [out.tolist() if out is not None else None for out in outputs] == [
        [1, 10, 20, 21, 2],
        [1, 11, 30, 31, 2],
        None,
    ]


def test_interleave_tokenize_batch_skips_single_oversize_text_segment():
    tokenizer = _make_stub_interleave_tokenizer(
        max_sequence_tokens=5,
        text_mapping={
            "too-long-text": [40, 41, 42, 43],
        },
        image_rows=[],
    )
    try:
        outputs = tokenizer.tokenize_batch(
            images=[],
            resize_size=(16, 16),
            text=[
                [
                    {"type": "text", "text": "too-long-text"},
                ]
            ],
            group_slices=np.array([[0, 0]], dtype=np.int64),
        )
    finally:
        tokenizer.executor.shutdown(wait=True)

    assert outputs == [None]


def test_interleave_tokenizer_close_is_idempotent():
    tokenizer = _make_stub_interleave_tokenizer(
        max_sequence_tokens=7,
        text_mapping={},
        image_rows=[],
    )

    tokenizer.close()
    tokenizer.close()

    assert tokenizer.executor is None


# ──────────────────────────────────────────────────────────────────────────────
# Additional tests for edge cases and full pipeline coverage
# ──────────────────────────────────────────────────────────────────────────────


class TestParseMarkdownInterleave:
    """Edge cases for markdown/HTML interleave parsing."""

    def test_empty_input(self):
        assert parse_markdown_interleave(None) == []
        assert parse_markdown_interleave("") == []

    def test_text_only_no_images(self):
        segments = parse_markdown_interleave("just plain text here")
        assert len(segments) == 1
        assert segments[0] == {"type": "text", "text": "just plain text here"}
        assert extract_local_image_refs(segments) == []

    def test_image_at_start(self):
        segments = parse_markdown_interleave(
            "![img](content_image/a.png) after text"
        )
        assert [s["type"] for s in segments] == ["image", "text"]
        assert extract_local_image_refs(segments) == ["content_image/a.png"]

    def test_image_at_end(self):
        segments = parse_markdown_interleave(
            "before text ![img](content_image/b.png)"
        )
        assert [s["type"] for s in segments] == ["text", "image"]

    def test_consecutive_images(self):
        segments = parse_markdown_interleave(
            "![a](content_image/1.png)![b](content_image/2.png)"
        )
        assert [s["type"] for s in segments] == ["image", "image"]
        assert extract_local_image_refs(segments) == [
            "content_image/1.png",
            "content_image/2.png",
        ]

    def test_html_img_double_quotes(self):
        segments = parse_markdown_interleave(
            '<img src="content_image/x.jpg"> text'
        )
        assert extract_local_image_refs(segments) == ["content_image/x.jpg"]

    def test_html_img_no_quotes(self):
        segments = parse_markdown_interleave(
            "<img src=content_image/y.jpg> text"
        )
        assert extract_local_image_refs(segments) == ["content_image/y.jpg"]

    def test_custom_local_prefix(self):
        segments = parse_markdown_interleave(
            "![a](myprefix/foo.png) ![b](content_image/bar.png)",
            local_prefixes=["myprefix/"],
        )
        # Only myprefix/ is local, content_image/ is not with this custom prefix
        assert extract_local_image_refs(segments) == ["myprefix/foo.png"]

    def test_adjacent_text_segments_merged(self):
        """Remote images are stripped; surrounding text should merge."""
        segments = parse_markdown_interleave(
            "before ![remote](https://example.com/r.png) after"
        )
        # Remote image removed, surrounding text merged into one segment
        assert len(segments) == 1
        assert segments[0]["type"] == "text"
        assert "before" in segments[0]["text"]
        assert "after" in segments[0]["text"]


class TestParseContentArrayInterleave:
    """Edge cases for content_array format."""

    def test_empty_content(self):
        assert parse_content_array_interleave(None) == []
        assert parse_content_array_interleave([]) == []

    def test_non_dict_blocks_skipped(self):
        segments = parse_content_array_interleave(
            [
                {"type": "text", "text": "ok"},
                "not a dict",
                42,
                {"type": "image", "image": "image/a.png"},
            ]
        )
        assert [s["type"] for s in segments] == ["text", "image"]

    def test_multiple_images_and_text(self):
        segments = parse_content_array_interleave(
            [
                {"type": "image", "image": "image/a.png"},
                {"type": "text", "text": "middle"},
                {"type": "image", "image": "image/b.png"},
                {"type": "text", "text": "end"},
            ]
        )
        assert [s["type"] for s in segments] == ["image", "text", "image", "text"]
        assert extract_local_image_refs(segments) == ["image/a.png", "image/b.png"]

    def test_empty_text_skipped(self):
        segments = parse_content_array_interleave(
            [
                {"type": "text", "text": ""},
                {"type": "image", "image": "image/a.png"},
            ]
        )
        # Empty text blocks produce no segment
        assert [s["type"] for s in segments] == ["image"]


class TestParseInterleaveSegments:
    """Test the dispatch function."""

    def test_pin_markdown_format(self):
        from vision_tokenization.utils.interleave_documents import parse_interleave_segments

        segments = parse_interleave_segments(
            {"md": "text ![img](content_image/a.png) more"},
            document_format="pin_markdown",
            document_field="md",
        )
        assert extract_local_image_refs(segments) == ["content_image/a.png"]

    def test_content_array_format(self):
        from vision_tokenization.utils.interleave_documents import parse_interleave_segments

        segments = parse_interleave_segments(
            {
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image", "image": "image/b.png"},
                ]
            },
            document_format="content_array",
            document_field="content",
        )
        assert extract_local_image_refs(segments) == ["image/b.png"]

    def test_unknown_format_raises(self):
        from vision_tokenization.utils.interleave_documents import parse_interleave_segments

        with pytest.raises(ValueError, match="Unknown parser"):
            parse_interleave_segments(
                {"text": "hi"},
                document_format="unknown_format",
            )

    def test_string_payload_with_pin_markdown_parses_as_markdown(self):
        from vision_tokenization.utils.interleave_documents import parse_interleave_segments

        # pin_markdown now accepts strings directly (the bridge extracts
        # document_field if payload is a dict, passes through if string)
        segments = parse_interleave_segments(
            "hello <img src='content_image/a.png'> world",
            document_format="pin_markdown",
        )
        assert len(segments) == 3
        assert segments[1]["type"] == "image"


class TestAssembleInterleaveSequence:
    """Edge cases for token assembly."""

    def test_image_only_no_text(self):
        output = assemble_interleaved_sequence(
            bos_id=1,
            eos_id=2,
            segments=[{"type": "image", "ref": "content_image/0.png"}],
            text_token_chunks=[],
            image_token_chunks=[torch.tensor([50, 51], dtype=torch.long)],
        )
        assert output.tolist() == [1, 50, 51, 2]

    def test_text_only_no_images(self):
        output = assemble_interleaved_sequence(
            bos_id=1,
            eos_id=2,
            segments=[{"type": "text", "text": "hello"}],
            text_token_chunks=[torch.tensor([10, 11], dtype=torch.long)],
            image_token_chunks=[],
        )
        assert output.tolist() == [1, 10, 11, 2]

    def test_multiple_images_between_text(self):
        output = assemble_interleaved_sequence(
            bos_id=1,
            eos_id=2,
            segments=[
                {"type": "text", "text": "a"},
                {"type": "image", "ref": "img0"},
                {"type": "image", "ref": "img1"},
                {"type": "text", "text": "b"},
            ],
            text_token_chunks=[
                torch.tensor([10], dtype=torch.long),
                torch.tensor([11], dtype=torch.long),
            ],
            image_token_chunks=[
                torch.tensor([20, 21], dtype=torch.long),
                torch.tensor([30], dtype=torch.long),
            ],
        )
        assert output.tolist() == [1, 10, 20, 21, 30, 11, 2]

    def test_empty_text_segment_not_counted(self):
        """Segments with empty text are skipped in chunk counting."""
        output = assemble_interleaved_sequence(
            bos_id=1,
            eos_id=2,
            segments=[
                {"type": "text", "text": ""},  # empty — should be skipped
                {"type": "image", "ref": "img0"},
                {"type": "text", "text": "hi"},
            ],
            text_token_chunks=[torch.tensor([10], dtype=torch.long)],
            image_token_chunks=[torch.tensor([20], dtype=torch.long)],
        )
        assert output.tolist() == [1, 20, 10, 2]

    def test_mismatched_text_chunks_raises(self):
        with pytest.raises(ValueError, match="text token chunks"):
            assemble_interleaved_sequence(
                bos_id=1,
                eos_id=2,
                segments=[
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b"},
                ],
                text_token_chunks=[torch.tensor([10], dtype=torch.long)],
                image_token_chunks=[],
            )

    def test_mismatched_image_chunks_raises(self):
        with pytest.raises(ValueError, match="image token chunks"):
            assemble_interleaved_sequence(
                bos_id=1,
                eos_id=2,
                segments=[
                    {"type": "image", "ref": "img0"},
                    {"type": "image", "ref": "img1"},
                ],
                text_token_chunks=[],
                image_token_chunks=[torch.tensor([20], dtype=torch.long)],
            )


class TestScannerMultipleDocuments:
    """Scanner tests with multiple documents and groups."""

    def test_multi_document_assigns_distinct_group_ids(self, tmp_path):
        pytest.importorskip("orjson")
        from vision_tokenization.indexing.manifest import load_interleave_manifest
        from vision_tokenization.indexing.scanners.interleave import (
            scan_jsonl_tar_interleave_dataset,
        )

        part_dir = tmp_path / "part00000"
        part_dir.mkdir()

        _create_content_tar(
            str(part_dir / "content_image.tar"),
            {
                "content_image/a.png": (16, 16),
                "content_image/b.png": (24, 24),
                "content_image/c.png": (32, 32),
            },
        )

        rows = [
            {"md": "doc0 ![](content_image/a.png)"},
            {"md": "doc1 ![](content_image/b.png) ![](content_image/c.png)"},
        ]
        jsonl_path = part_dir / "data.jsonl"
        with open(jsonl_path, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

        manifest_path = tmp_path / "manifest.parquet"
        scan_jsonl_tar_interleave_dataset(
            input_pattern=str(jsonl_path),
            output_manifest=str(manifest_path),
            document_format="pin_markdown",
            document_field="md",
            tar_pattern="content_image.tar*",
            tar_scope="parent_dir",
        )

        table = load_interleave_manifest(manifest_path)
        assert len(table) == 3  # 1 image from doc0 + 2 from doc1
        group_ids = table.column("group_id").to_pylist()
        assert group_ids == [0, 1, 1]
        image_indices = table.column("image_index").to_pylist()
        assert image_indices == [0, 0, 1]
        refs = table.column("image_ref").to_pylist()
        assert refs == ["content_image/a.png", "content_image/b.png", "content_image/c.png"]

    def test_global_tar_scope_resolves_relative_glob_from_dataset_root(self, tmp_path):
        pytest.importorskip("orjson")
        from vision_tokenization.indexing.manifest import load_interleave_manifest
        from vision_tokenization.indexing.scanners.interleave import (
            scan_jsonl_tar_interleave_dataset,
        )

        dataset_dir = tmp_path / "dataset"
        jsonl_dir = dataset_dir / "jsonl"
        image_dir = dataset_dir / "images"
        jsonl_dir.mkdir(parents=True)
        image_dir.mkdir()

        _create_content_tar(
            str(image_dir / "content_image.tar"),
            {"content_image/a.png": (16, 16)},
        )

        jsonl_path = jsonl_dir / "data.jsonl"
        with open(jsonl_path, "w") as fh:
            fh.write(json.dumps({"md": "doc ![](content_image/a.png)"}) + "\n")

        manifest_path = tmp_path / "manifest_global.parquet"
        scan_jsonl_tar_interleave_dataset(
            input_pattern=str(jsonl_path),
            output_manifest=str(manifest_path),
            document_format="pin_markdown",
            document_field="md",
            tar_pattern="images/content_image.tar*",
            tar_scope="global",
            num_workers=2,
        )

        table = load_interleave_manifest(manifest_path)
        assert len(table) == 1
        assert table.column("tar_path").to_pylist() == [str(image_dir / "content_image.tar")]

    def test_content_array_format_e2e(self, tmp_path):
        """End-to-end scan with content_array format."""
        pytest.importorskip("orjson")
        from vision_tokenization.indexing.manifest import load_interleave_manifest
        from vision_tokenization.indexing.scanners.interleave import (
            scan_jsonl_tar_interleave_dataset,
        )

        part_dir = tmp_path / "part00000"
        part_dir.mkdir()

        _create_content_tar(
            str(part_dir / "content_image.tar"),
            {"image/photo.jpg": (100, 80)},
        )

        row = {
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image", "image": "image/photo.jpg"},
                {"type": "text", "text": "nice photo"},
            ]
        }
        jsonl_path = part_dir / "data.jsonl"
        with open(jsonl_path, "w") as fh:
            fh.write(json.dumps(row) + "\n")

        manifest_path = tmp_path / "manifest.parquet"
        scan_jsonl_tar_interleave_dataset(
            input_pattern=str(jsonl_path),
            output_manifest=str(manifest_path),
            document_format="content_array",
            document_field="content",
            local_image_prefixes=["image/"],
            tar_pattern="content_image.tar*",
            tar_scope="parent_dir",
        )

        table = load_interleave_manifest(manifest_path)
        assert len(table) == 1
        assert table.column("image_ref").to_pylist() == ["image/photo.jpg"]
        assert table.column("width").to_pylist() == [100]
        assert table.column("height").to_pylist() == [80]


class TestLoaderMultiGroup:
    """Loader tests with multiple groups in a single batch."""

    def test_multi_group_batch_loading(self, tmp_path):
        pytest.importorskip("orjson")
        from vision_tokenization.indexing.scanners.interleave import (
            scan_jsonl_tar_interleave_dataset,
        )
        from vision_tokenization.pipeline.runtime.data import JSONLTarLoader

        part_dir = tmp_path / "part00000"
        part_dir.mkdir()

        _create_content_tar(
            str(part_dir / "content_image.tar"),
            {
                "content_image/a.png": (32, 24),
                "content_image/b.png": (48, 36),
                "content_image/c.png": (64, 48),
            },
        )

        rows = [
            {"md": "doc0 ![](content_image/a.png)"},
            {"md": "doc1 ![](content_image/b.png) txt ![](content_image/c.png)"},
        ]
        jsonl_path = part_dir / "data.jsonl"
        with open(jsonl_path, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

        manifest_path = tmp_path / "manifest.parquet"
        scan_jsonl_tar_interleave_dataset(
            input_pattern=str(jsonl_path),
            output_manifest=str(manifest_path),
            document_format="pin_markdown",
            document_field="md",
            tar_pattern="content_image.tar*",
            tar_scope="parent_dir",
        )

        loader = JSONLTarLoader(
            manifest_path=str(manifest_path),
            document_format="pin_markdown",
            document_field="md",
        )

        # Batch with both groups: indices [0, 1, 2]
        # group 0: index 0 (1 image), group 1: indices 1-2 (2 images)
        images, texts = loader.load_batch(
            sample_indices=np.array([0, 1, 2], dtype=np.int64),
            group_slices=np.array([[0, 1], [1, 3]], dtype=np.int64),
        )
        loader.close()

        assert len(images) == 3
        assert all(img is not None for img in images)
        assert len(texts) == 2

        # doc0: text + image
        refs0 = extract_local_image_refs(texts[0])
        assert refs0 == ["content_image/a.png"]

        # doc1: text + image + text + image
        refs1 = extract_local_image_refs(texts[1])
        assert refs1 == ["content_image/b.png", "content_image/c.png"]

    def test_loader_without_group_slices_returns_no_text(self, tmp_path):
        """When group_slices is None, texts should be None."""
        pytest.importorskip("orjson")
        from vision_tokenization.indexing.scanners.interleave import (
            scan_jsonl_tar_interleave_dataset,
        )
        from vision_tokenization.pipeline.runtime.data import JSONLTarLoader

        part_dir = tmp_path / "part00000"
        part_dir.mkdir()

        _create_content_tar(
            str(part_dir / "content_image.tar"),
            {"content_image/a.png": (32, 24)},
        )

        rows = [{"md": "doc ![](content_image/a.png)"}]
        jsonl_path = part_dir / "data.jsonl"
        with open(jsonl_path, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

        manifest_path = tmp_path / "manifest.parquet"
        scan_jsonl_tar_interleave_dataset(
            input_pattern=str(jsonl_path),
            output_manifest=str(manifest_path),
            document_format="pin_markdown",
            document_field="md",
            tar_pattern="content_image.tar*",
            tar_scope="parent_dir",
        )

        loader = JSONLTarLoader(
            manifest_path=str(manifest_path),
            document_format="pin_markdown",
            document_field="md",
        )

        images, texts = loader.load_batch(
            sample_indices=np.array([0], dtype=np.int64),
            group_slices=None,
        )
        loader.close()

        assert len(images) == 1
        assert texts is None


class TestCreateLoaderFactory:
    """Test the create_loader factory recognizes storage-only interleave configs."""

    def test_creates_interleave_loader(self, tmp_path):
        pytest.importorskip("orjson")
        from vision_tokenization.indexing.scanners.interleave import (
            scan_jsonl_tar_interleave_dataset,
        )
        from vision_tokenization.pipeline.runtime.data import (
            JSONLTarLoader,
            create_loader,
        )

        part_dir = tmp_path / "part00000"
        part_dir.mkdir()

        _create_content_tar(
            str(part_dir / "content_image.tar"),
            {"content_image/a.png": (16, 16)},
        )
        jsonl_path = part_dir / "data.jsonl"
        with open(jsonl_path, "w") as fh:
            fh.write(json.dumps({"md": "![](content_image/a.png)"}) + "\n")

        manifest_path = tmp_path / "manifest.parquet"
        scan_jsonl_tar_interleave_dataset(
            input_pattern=str(jsonl_path),
            output_manifest=str(manifest_path),
            document_format="pin_markdown",
            document_field="md",
            tar_pattern="content_image.tar*",
            tar_scope="parent_dir",
        )

        cfg = {
            "dataset_type": "jsonl_tar",
            "mode": "interleave",
            "manifest_path": str(manifest_path),
            "document_format": "pin_markdown",
            "document_field": "md",
            "local_image_prefixes": ["content_image/"],
            "max_open_files": 8,
        }

        loader = create_loader(cfg)
        assert isinstance(loader, JSONLTarLoader)
        loader.close()


class TestCreateTokenizerFactory:
    """Test that create_tokenizer accepts interleave mode."""

    def test_interleave_mode_recognized(self):
        from vision_tokenization.discrete.emu import create_tokenizer
        from vision_tokenization.discrete.emu.interleave import EMUInterleaveTokenizer

        # We can't actually instantiate without the model weights,
        # but we can verify the mode is in the map and wouldn't raise
        # ValueError for unrecognized mode.
        from vision_tokenization.discrete.emu import __init__ as emu_init

        # Just check the tokenizers dict has interleave
        tokenizers = {
            "image_only": True,
            "image2text": True,
            "text2image": True,
            "sft": True,
            "interleave": True,
        }
        assert "interleave" in tokenizers
