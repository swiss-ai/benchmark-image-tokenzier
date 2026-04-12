from __future__ import annotations

import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from vision_tokenization.indexing.scanners.hf import scan_hf_dataset
from vision_tokenization.parsers import parse_sft_messages
from vision_tokenization.pipeline.runtime.data import HFImageLoader, create_loader


_HF_IMAGE_TYPE = pa.struct(
    [
        pa.field("bytes", pa.binary()),
        pa.field("path", pa.string()),
    ]
)


def _make_image(width: int, height: int, color=(255, 0, 0)) -> Image.Image:
    return Image.new("RGB", (width, height), color)


def _image_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _hf_image_cell(width: int, height: int) -> dict:
    return {
        "bytes": _image_bytes(_make_image(width, height)),
        "path": None,
    }


def _write_hf_parquet_shard(
    shard_path: str,
    rows: list,
    *,
    column_name: str = "image",
    multi_image: bool = False,
    extra_columns: dict[str, list] | None = None,
):
    if multi_image:
        image_array = pa.array(
            [
                [_hf_image_cell(width, height) for width, height in sample]
                for sample in rows
            ],
            type=pa.list_(_HF_IMAGE_TYPE),
        )
    else:
        image_array = pa.array(
            [_hf_image_cell(width, height) for width, height in rows],
            type=_HF_IMAGE_TYPE,
        )

    columns = {column_name: image_array}
    for key, values in (extra_columns or {}).items():
        columns[key] = pa.array(values)
    pq.write_table(pa.table(columns), shard_path)


def test_conversation_parser_accepts_conversation_field():
    row = {
        "conversation": [
            {"from": "human", "value": "<image>\nDescribe the scene."},
            {"from": "gpt", "value": "A snowy field."},
        ]
    }

    messages = parse_sft_messages(row, parser="conversation")

    assert messages == row["conversation"]


def test_conversation_parser_accepts_configured_conversation_column():
    row = {
        "dialog": [
            {"from": "human", "value": "<image>\nDescribe the scene."},
            {"from": "gpt", "value": "A snowy field."},
        ]
    }

    messages = parse_sft_messages(
        row,
        parser="conversation",
        parser_args={"conversation_column": "dialog"},
    )

    assert messages == row["dialog"]


def test_qa_parser_prepends_expected_number_of_placeholders():
    row = {"input": "Which regions are visible?", "output": "forest and water"}

    messages = parse_sft_messages(row, parser="qa", num_images=2)

    assert messages == [
        {"role": "user", "content": "<image>\n<image>\nWhich regions are visible?"},
        {"role": "assistant", "content": "forest and water"},
    ]


def test_qa_parser_rejects_marker_count_mismatch():
    row = {"question": "<image>\nCompare the two panels.", "answer": "left is brighter"}

    with pytest.raises(ValueError, match="Prompt contains 1 image marker\\(s\\), expected 2"):
        parse_sft_messages(row, parser="qa", num_images=2)


def test_molmo_multi_image_qa_parser_builds_multi_turn_conversation():
    row = {
        "image_urls": ["a", "b", "c"],
        "qa_pairs": {
            "question": ["Which image is the warmest?", "Which image is the brightest?"],
            "answer": ["The third.", "The first."],
        },
    }

    messages = parse_sft_messages(row, parser="molmo_multi_image_qa", num_images=3)

    assert messages == [
        {"role": "user", "content": "<image>\n<image>\n<image>\nWhich image is the warmest?"},
        {"role": "assistant", "content": "The third."},
        {"role": "user", "content": "Which image is the brightest?"},
        {"role": "assistant", "content": "The first."},
    ]


def test_hf_loader_single_image_sft_parser_returns_canonical_messages(tmp_path):
    shard_path = str(tmp_path / "single.parquet")
    _write_hf_parquet_shard(
        shard_path,
        [(32, 48), (64, 96)],
        extra_columns={
            "question": ["What organ is shown?", "<image>\nWhat tissue is shown?"],
            "answer": ["heart", "epithelium"],
        },
    )

    loader = HFImageLoader(
        input_pattern=str(tmp_path / "*.parquet"),
        parser="qa",
        parser_columns=["question", "answer"],
        parser_kind="sft",
    )
    images, texts = loader.load_batch(np.array([0, 1], dtype=np.int64))
    loader.close()

    assert [img.size for img in images] == [(32, 48), (64, 96)]
    assert texts == [
        [
            {"role": "user", "content": "<image>\nWhat organ is shown?"},
            {"role": "assistant", "content": "heart"},
        ],
        [
            {"role": "user", "content": "<image>\nWhat tissue is shown?"},
            {"role": "assistant", "content": "epithelium"},
        ],
    ]


def test_hf_loader_single_image_conversation_parser_honors_parser_args(tmp_path):
    shard_path = str(tmp_path / "single_dialog.parquet")
    _write_hf_parquet_shard(
        shard_path,
        [(28, 36)],
        extra_columns={
            "dialog": [[
                {"from": "human", "value": "<image>\nDescribe the scene."},
                {"from": "gpt", "value": "A snowy field."},
            ]],
        },
    )

    loader = HFImageLoader(
        input_pattern=str(tmp_path / "*.parquet"),
        parser="conversation",
        parser_columns=["dialog"],
        parser_args={"conversation_column": "dialog"},
        parser_kind="sft",
    )
    images, texts = loader.load_batch(np.array([0], dtype=np.int64))
    loader.close()

    assert [img.size for img in images] == [(28, 36)]
    assert texts == [[
        {"from": "human", "value": "<image>\nDescribe the scene."},
        {"from": "gpt", "value": "A snowy field."},
    ]]


def test_hf_loader_multi_image_sft_parser_uses_group_image_count(tmp_path):
    shard_path = str(tmp_path / "multi.parquet")
    _write_hf_parquet_shard(
        shard_path,
        [
            [(11, 21), (31, 41)],
            [(51, 61)],
        ],
        column_name="images",
        multi_image=True,
        extra_columns={
            "question": ["Compare the two images.", "What is visible?"],
            "answer": ["The left one is darker.", "A mountain."],
        },
    )

    manifest_path = str(tmp_path / "multi_manifest.parquet")
    scan_hf_dataset(
        input_pattern=str(tmp_path / "*.parquet"),
        output_manifest=manifest_path,
        image_list_column="images",
        num_workers=1,
    )

    loader = HFImageLoader(
        input_pattern=str(tmp_path / "*.does_not_matter"),
        manifest_path=manifest_path,
        image_list_column="images",
        parser="qa",
        parser_columns=["question", "answer"],
        parser_kind="sft",
    )
    images, texts = loader.load_batch(
        np.array([0, 1, 2], dtype=np.int64),
        group_slices=np.array([[0, 2], [2, 3]], dtype=np.int64),
    )
    loader.close()

    assert [img.size for img in images] == [(11, 21), (31, 41), (51, 61)]
    assert texts == [
        [
            {"role": "user", "content": "<image>\n<image>\nCompare the two images."},
            {"role": "assistant", "content": "The left one is darker."},
        ],
        [
            {"role": "user", "content": "<image>\nWhat is visible?"},
            {"role": "assistant", "content": "A mountain."},
        ],
    ]


def test_parse_row_dicts_handles_sparse_batch_positions(tmp_path):
    shard_path = str(tmp_path / "single.parquet")
    _write_hf_parquet_shard(
        shard_path,
        [(24, 24)],
        extra_columns={"question": ["unused"], "answer": ["unused"]},
    )

    loader = HFImageLoader(
        input_pattern=str(tmp_path / "*.parquet"),
        parser="qa",
        parser_columns=["question", "answer"],
        parser_kind="sft",
    )
    try:
        texts = loader._parse_row_dicts(
            {1: {"question": "What is shown?", "answer": "cell"}},
            batch_size=2,
        )
    finally:
        loader.close()

    assert texts == [
        None,
        [
            {"role": "user", "content": "<image>\nWhat is shown?"},
            {"role": "assistant", "content": "cell"},
        ],
    ]


def test_hf_loader_fragmented_multi_image_sft_parser_uses_full_document_image_count(tmp_path):
    shard_path = str(tmp_path / "multi.parquet")
    _write_hf_parquet_shard(
        shard_path,
        [[(11, 21), (31, 41), (51, 61)]],
        column_name="images",
        multi_image=True,
        extra_columns={
            "question": ["Compare all three images."],
            "answer": ["The first is darkest."],
        },
    )

    manifest_path = str(tmp_path / "multi_manifest.parquet")
    scan_hf_dataset(
        input_pattern=str(tmp_path / "*.parquet"),
        output_manifest=manifest_path,
        image_list_column="images",
        num_workers=1,
    )

    loader = HFImageLoader(
        input_pattern=str(tmp_path / "*.does_not_matter"),
        manifest_path=manifest_path,
        image_list_column="images",
        parser="qa",
        parser_columns=["question", "answer"],
        parser_kind="sft",
    )
    try:
        images, texts = loader.load_batch(
            np.array([0, 1], dtype=np.int64),
            group_slices=np.array([[0, 2]], dtype=np.int64),
        )
        text_only = loader.load_text_batch(
            np.array([0, 1], dtype=np.int64),
            group_slices=np.array([[0, 2]], dtype=np.int64),
        )
    finally:
        loader.close()

    assert [img.size for img in images] == [(11, 21), (31, 41)]
    expected_messages = [
        {"role": "user", "content": "<image>\n<image>\n<image>\nCompare all three images."},
        {"role": "assistant", "content": "The first is darkest."},
    ]
    assert texts == [expected_messages]
    assert text_only == [expected_messages]


def test_create_loader_enables_sft_parser_for_hf_dataset(tmp_path):
    shard_path = str(tmp_path / "single.parquet")
    _write_hf_parquet_shard(
        shard_path,
        [(24, 24)],
        extra_columns={"question": ["What is shown?"], "answer": ["cell"]},
    )

    loader = create_loader(
        {
            "dataset_type": "hf",
            "mode": "sft",
            "input_pattern": str(tmp_path / "*.parquet"),
            "image_column": "image",
            "parser": "qa",
            "parser_columns": ["question", "answer"],
        }
    )
    try:
        images, texts = loader.load_batch(np.array([0], dtype=np.int64))
    finally:
        loader.close()

    assert [img.size for img in images] == [(24, 24)]
    assert texts == [
        [
            {"role": "user", "content": "<image>\nWhat is shown?"},
            {"role": "assistant", "content": "cell"},
        ]
    ]


def test_create_loader_requires_parser_columns_for_hf_sft_parser(tmp_path):
    shard_path = str(tmp_path / "single.parquet")
    _write_hf_parquet_shard(
        shard_path,
        [(24, 24)],
        extra_columns={"question": ["What is shown?"], "answer": ["cell"]},
    )

    with pytest.raises(ValueError, match="requires parser_columns to be set"):
        create_loader(
            {
                "dataset_type": "hf",
                "mode": "sft",
                "input_pattern": str(tmp_path / "*.parquet"),
                "image_column": "image",
                "parser": "qa",
            }
        )
