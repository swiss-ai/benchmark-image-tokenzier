from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from vision_tokenization.discrete.sft_segments import render_sft_segments
from vision_tokenization.indexing.manifest import HF_SCHEMA_PHYSICAL_MULTI_IMAGE
from vision_tokenization.indexing.scanners.hf import scan_hf_dataset
from vision_tokenization.indexing.scanners._workers import hf_common
from vision_tokenization.indexing.scanners._workers.hf_common import (
    build_hf_output_columns,
    build_hf_output_table,
    scan_hf_image_map_batch_columns,
)
from vision_tokenization.indexing.scanners._workers.hf_parquet import scan_single_hf_parquet_shard
from vision_tokenization.parsers import parse_sft_messages
from vision_tokenization.pipeline.runtime.data import HFImageLoader, create_loader
from vision_tokenization.utils import image_map_parquet


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


class _ApertusShapeChatTokenizer:
    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, **kwargs):
        assert tokenize is False
        rendered = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "user":
                rendered.append("<|user_start|>")
                if isinstance(content, str):
                    rendered.append(content)
                elif isinstance(content, dict) and "parts" in content:
                    for part in content["parts"]:
                        if part["type"] == "image":
                            rendered.append("<|image|>")
                        elif part["type"] == "text":
                            rendered.append(part["text"])
                        else:
                            raise ValueError(f"Invalid user part: {part['type']}")
                else:
                    raise ValueError(f"Invalid user message: {role}")
                rendered.append("<|user_end|>")
            elif role == "assistant":
                if not isinstance(content, str):
                    raise ValueError("Invalid assistant content")
                rendered.append("<|assistant_start|>")
                rendered.append(content)
                rendered.append("<|assistant_end|>")
            else:
                raise ValueError(f"Invalid role: {role}")
        return "".join(rendered)


def _write_image_map_sft_parquet_shard(shard_path: str, *, cell_shape: str = "raw_bytes"):
    img_a = _image_bytes(_make_image(17, 23, (255, 0, 0)))
    img_b = _image_bytes(_make_image(31, 37, (0, 255, 0)))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "a.png"},
                "Compare the images.",
                {"type": "image", "image": "b.png"},
            ],
        },
        {"role": "assistant", "content": ["The first is red."]},
    ]
    if cell_shape == "raw_bytes":
        images = pa.array(
            [[("b.png", img_b), ("a.png", img_a)]],
            type=pa.map_(pa.string(), pa.binary()),
        )
    elif cell_shape == "struct_with_bytes":
        images = pa.array(
            [[
                ("b.png", {"bytes": img_b, "path": None}),
                ("a.png", {"bytes": img_a, "path": None}),
            ]],
            type=pa.map_(pa.string(), _HF_IMAGE_TYPE),
        )
    elif cell_shape == "struct_with_path":
        shard_parent = Path(shard_path).parent
        img_a_path = shard_parent / "a.png"
        img_b_path = shard_parent / "b.png"
        _make_image(17, 23, (255, 0, 0)).save(img_a_path)
        _make_image(31, 37, (0, 255, 0)).save(img_b_path)
        images = pa.array(
            [[
                ("b.png", {"bytes": None, "path": str(img_b_path)}),
                ("a.png", {"bytes": None, "path": str(img_a_path)}),
            ]],
            type=pa.map_(pa.string(), _HF_IMAGE_TYPE),
        )
    else:
        raise ValueError(f"Unsupported cell_shape: {cell_shape}")

    table = pa.table(
        {
            "messages": pa.array([json.dumps(messages)]),
            "images": images,
        }
    )
    pq.write_table(table, shard_path)


def _write_two_row_image_map_sft_parquet_shard(shard_path: str):
    img_a = _image_bytes(_make_image(17, 23, (255, 0, 0)))
    img_b = _image_bytes(_make_image(31, 37, (0, 255, 0)))
    messages = [
        [
            {
                "role": "user",
                "content": [{"type": "image", "image": "a.png"}, "First?"],
            },
            {"role": "assistant", "content": ["red"]},
        ],
        [
            {
                "role": "user",
                "content": [{"type": "image", "image": "b.png"}, "Second?"],
            },
            {"role": "assistant", "content": ["green"]},
        ],
    ]
    table = pa.table(
        {
            "messages": pa.array([json.dumps(msg) for msg in messages]),
            "images": pa.array(
                [[("a.png", img_a)], [("b.png", img_b)]],
                type=pa.map_(pa.string(), pa.binary()),
            ),
        }
    )
    pq.write_table(table, shard_path)


def test_image_map_conversation_parser_normalizes_json_string_messages():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "img0.png"},
                "What is shown?",
            ],
        },
        {"role": "assistant", "content": ["A chart."]},
    ]

    parsed = parse_sft_messages(
        {"messages": json.dumps(messages)},
        parser="image_map_conversation",
        parser_args={"conversation_column": "messages"},
    )

    assert parsed == [
        {
            "role": "user",
            "content": {
                "parts": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown?"},
                ],
            },
        },
        {
            "role": "assistant",
            "content": "A chart.",
        },
    ]


def test_image_map_conversation_parser_renders_through_apertus_chat_template_shape():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "img0.png"},
                "How were the drivers able to park here?",
            ],
        },
        {"role": "assistant", "content": ["<think>Reasoning</think>\nB"]},
    ]

    parsed = parse_sft_messages(
        {"messages": json.dumps(messages)},
        parser="image_map_conversation",
        parser_args={"conversation_column": "messages"},
    )
    rendered = render_sft_segments(
        parsed,
        text_tokenizer=_ApertusShapeChatTokenizer(),
        image_marker_candidates=("<|image|>",),
        expected_num_images=1,
    )

    assert rendered.rendered_text == (
        "<|user_start|><|image|>How were the drivers able to park here?<|user_end|>"
        "<|assistant_start|><think>Reasoning</think>\nB<|assistant_end|>"
    )
    assert rendered.segments == [
        {"type": "text", "text": "<|user_start|>"},
        {"type": "image"},
        {
            "type": "text",
            "text": (
                "How were the drivers able to park here?<|user_end|>"
                "<|assistant_start|><think>Reasoning</think>\nB<|assistant_end|>"
            ),
        },
    ]


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


def test_hf_loader_image_map_sft_uses_message_order(tmp_path):
    shard_path = str(tmp_path / "image_map.parquet")
    _write_image_map_sft_parquet_shard(shard_path)

    manifest_path = str(tmp_path / "image_map_manifest.parquet")
    scan_hf_dataset(
        input_pattern=shard_path,
        output_manifest=manifest_path,
        image_map_column="images",
        message_column="messages",
        num_workers=1,
    )

    loader = HFImageLoader(
        input_pattern=str(tmp_path / "*.does_not_matter"),
        manifest_path=manifest_path,
        image_map_column="images",
        message_column="messages",
        parser="image_map_conversation",
        parser_columns=["messages"],
        parser_args={"conversation_column": "messages"},
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

    assert [img.size for img in images] == [(17, 23), (31, 37)]
    expected_messages = [
        {
            "role": "user",
            "content": {
                "parts": [
                    {"type": "image"},
                    {"type": "text", "text": "Compare the images."},
                    {"type": "image"},
                ],
            },
        },
        {
            "role": "assistant",
            "content": "The first is red.",
        },
    ]
    assert texts == [expected_messages]
    assert text_only == [expected_messages]


def test_create_loader_enables_image_map_sft_loader(tmp_path):
    shard_path = str(tmp_path / "image_map.parquet")
    _write_image_map_sft_parquet_shard(shard_path)

    manifest_path = str(tmp_path / "image_map_manifest.parquet")
    scan_hf_dataset(
        input_pattern=shard_path,
        output_manifest=manifest_path,
        image_map_column="images",
        message_column="messages",
        num_workers=1,
    )

    loader = create_loader(
        {
            "dataset_type": "hf",
            "mode": "sft",
            "input_pattern": str(tmp_path / "*.does_not_matter"),
            "manifest_path": manifest_path,
            "image_map_column": "images",
            "message_column": "messages",
            "parser": "image_map_conversation",
            "parser_columns": ["messages"],
            "parser_args": {"conversation_column": "messages"},
        }
    )
    try:
        images, texts = loader.load_batch(
            np.array([0, 1], dtype=np.int64),
            group_slices=np.array([[0, 2]], dtype=np.int64),
        )
    finally:
        loader.close()

    assert [img.size for img in images] == [(17, 23), (31, 37)]
    assert texts[0][0]["content"] == {
        "parts": [
            {"type": "image"},
            {"type": "text", "text": "Compare the images."},
            {"type": "image"},
        ],
    }


@pytest.mark.parametrize("cell_shape", ["raw_bytes", "struct_with_bytes", "struct_with_path"])
def test_hf_loader_image_map_sft_cell_shapes_scan_load_and_render(tmp_path, cell_shape):
    shard_path = str(tmp_path / "image_map.parquet")
    _write_image_map_sft_parquet_shard(shard_path, cell_shape=cell_shape)

    manifest_path = str(tmp_path / "image_map_manifest.parquet")
    scan_hf_dataset(
        input_pattern=shard_path,
        output_manifest=manifest_path,
        image_map_column="images",
        message_column="messages",
        num_workers=1,
    )

    manifest = pq.read_table(manifest_path)
    assert manifest.column("width").to_pylist() == [17, 31]
    assert manifest.column("height").to_pylist() == [23, 37]

    loader = HFImageLoader(
        input_pattern=str(tmp_path / "*.does_not_matter"),
        manifest_path=manifest_path,
        image_map_column="images",
        message_column="messages",
        parser="image_map_conversation",
        parser_columns=["messages"],
        parser_args={"conversation_column": "messages"},
        parser_kind="sft",
    )
    try:
        images, texts = loader.load_batch(
            np.array([0, 1], dtype=np.int64),
            group_slices=np.array([[0, 2]], dtype=np.int64),
        )
    finally:
        loader.close()

    assert [img.size for img in images] == [(17, 23), (31, 37)]
    rendered = render_sft_segments(
        texts[0],
        text_tokenizer=_ApertusShapeChatTokenizer(),
        image_marker_candidates=("<|image|>",),
        expected_num_images=2,
    )
    assert rendered.segments.count({"type": "image"}) == 2


def test_hf_image_map_scan_uses_arrow_map_arrays_without_python_map_materialization(
    tmp_path,
    monkeypatch,
):
    shard_path = str(tmp_path / "image_map.parquet")
    _write_image_map_sft_parquet_shard(shard_path)
    batch = pq.read_table(shard_path, columns=["images", "messages"])

    def _fail_image_map_as_dict(*args, **kwargs):
        raise AssertionError("scanner must not materialize each image map as a Python dict")

    monkeypatch.setattr(hf_common, "image_map_as_dict", _fail_image_map_as_dict)

    out = build_hf_output_columns(is_multi=True)
    out, source_rows, failed_dims, failed_messages, failed_image_maps = (
        scan_hf_image_map_batch_columns(
            out,
            batch.column("images"),
            batch.column("messages"),
            chunk_index=0,
            source_rows=0,
            failed_dims=0,
            failed_messages=0,
            failed_image_maps=0,
        )
    )
    table = build_hf_output_table(out, is_multi=True)

    assert source_rows == 1
    assert failed_dims == 0
    assert failed_messages == 0
    assert failed_image_maps == 0
    assert table.column("width").to_pylist() == [17, 31]
    assert table.column("height").to_pylist() == [23, 37]
    assert table.column("image_index").to_pylist() == [0, 1]


def test_hf_image_map_scan_keeps_single_image_rows_on_python_path(tmp_path, monkeypatch):
    shard_path = str(tmp_path / "image_map.parquet")
    _write_two_row_image_map_sft_parquet_shard(shard_path)
    batch = pq.read_table(shard_path, columns=["images", "messages"])

    def _fail_binary_header_array(*args, **kwargs):
        raise AssertionError("single-image map rows should not pay Arrow header slicing cost")

    monkeypatch.setattr(hf_common, "_binary_header_array", _fail_binary_header_array)

    out = build_hf_output_columns(is_multi=True)
    out, source_rows, failed_dims, failed_messages, failed_image_maps = (
        scan_hf_image_map_batch_columns(
            out,
            batch.column("images"),
            batch.column("messages"),
            chunk_index=0,
            source_rows=0,
            failed_dims=0,
            failed_messages=0,
            failed_image_maps=0,
        )
    )
    table = build_hf_output_table(out, is_multi=True)

    assert source_rows == 2
    assert failed_dims == 0
    assert failed_messages == 0
    assert failed_image_maps == 0
    assert table.column("width").to_pylist() == [17, 31]


def test_hf_image_map_scan_streams_batches_without_read_row_group(tmp_path, monkeypatch):
    shard_path = str(tmp_path / "image_map.parquet")
    _write_two_row_image_map_sft_parquet_shard(shard_path)
    monkeypatch.setattr(image_map_parquet, "DEFAULT_IMAGE_MAP_BATCH_SIZE", 1)

    def _fail_read_row_group(*args, **kwargs):
        raise AssertionError("image-map scan must not materialize full row groups")

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", _fail_read_row_group)

    table, source_rows, failed_dims, failed_messages, failed_image_maps, skip_reason = (
        scan_single_hf_parquet_shard(
            shard_path,
            image_map_column="images",
            message_column="messages",
        )
    )

    assert skip_reason is None
    assert source_rows == 2
    assert failed_dims == 0
    assert failed_messages == 0
    assert failed_image_maps == 0
    assert table.column("row_in_chunk").to_pylist() == [0, 1]


def test_hf_image_map_loader_streams_rows_without_read_row_group(tmp_path, monkeypatch):
    shard_path = tmp_path / "image_map.parquet"
    _write_two_row_image_map_sft_parquet_shard(str(shard_path))
    manifest_path = tmp_path / "image_map_manifest.parquet"
    manifest = pa.table(
        {
            "sample_index": pa.array([0, 1], type=pa.int64()),
            "width": pa.array([17, 31], type=pa.int32()),
            "height": pa.array([23, 37], type=pa.int32()),
            "group_id": pa.array([0, 1], type=pa.int64()),
            "image_index": pa.array([0, 0], type=pa.int16()),
            "shard_path": pa.array([str(shard_path), str(shard_path)]).dictionary_encode(),
            "chunk_index": pa.array([0, 0], type=pa.int32()),
            "row_in_chunk": pa.array([0, 1], type=pa.int32()),
        },
        schema=HF_SCHEMA_PHYSICAL_MULTI_IMAGE,
    )
    pq.write_table(manifest, manifest_path)

    def _fail_read_row_group(*args, **kwargs):
        raise AssertionError("image-map loader must not materialize full row groups")

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", _fail_read_row_group)

    loader = HFImageLoader(
        input_pattern=str(tmp_path / "*.does_not_matter"),
        manifest_path=manifest_path,
        image_map_column="images",
        message_column="messages",
        parser="image_map_conversation",
        parser_columns=["messages"],
        parser_args={"conversation_column": "messages"},
        parser_kind="sft",
    )
    try:
        images, texts = loader.load_batch(
            np.array([0, 1], dtype=np.int64),
            group_slices=np.array([[0, 1], [1, 2]], dtype=np.int64),
        )
    finally:
        loader.close()

    assert [img.size for img in images] == [(17, 23), (31, 37)]
    assert [text[1]["content"] for text in texts] == ["red", "green"]


@pytest.mark.parametrize(
    "bad_messages",
    [
        "{not json",
        json.dumps([{"role": "user", "content": [{"type": "image"}, "Q"]}]),
    ],
)
def test_hf_image_map_scan_reports_invalid_messages(tmp_path, bad_messages):
    img_a = _image_bytes(_make_image(17, 23, (255, 0, 0)))
    valid_messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": "a.png"}, "Q"],
        },
        {"role": "assistant", "content": ["A"]},
    ]
    shard_path = tmp_path / "image_map.parquet"
    table = pa.table(
        {
            "messages": pa.array([json.dumps(valid_messages), bad_messages]),
            "images": pa.array(
                [
                    [("a.png", img_a)],
                    [("a.png", img_a)],
                ],
                type=pa.map_(pa.string(), pa.binary()),
            ),
        }
    )
    pq.write_table(table, shard_path)

    manifest_path = tmp_path / "image_map_manifest.parquet"
    scan_hf_dataset(
        input_pattern=str(shard_path),
        output_manifest=str(manifest_path),
        image_map_column="images",
        message_column="messages",
        num_workers=1,
    )

    manifest = pq.read_table(manifest_path)
    assert len(manifest) == 1

    meta_path = manifest_path.with_name(manifest_path.stem + "_meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["total_source_rows"] == 2
    assert meta["failed_messages"] == 1
    assert meta["failed_image_maps"] == 0


def test_hf_image_map_scan_reports_invalid_image_maps(tmp_path):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": "a.png"}, "Q"],
        },
        {"role": "assistant", "content": ["A"]},
    ]
    shard_path = tmp_path / "image_map.parquet"
    pq.write_table(
        pa.table(
            {
                "messages": pa.array([json.dumps(messages)]),
                "images": pa.array(["not a map"]),
            }
        ),
        shard_path,
    )

    manifest_path = tmp_path / "image_map_manifest.parquet"
    scan_hf_dataset(
        input_pattern=str(shard_path),
        output_manifest=str(manifest_path),
        image_map_column="images",
        message_column="messages",
        num_workers=1,
    )

    manifest = pq.read_table(manifest_path)
    assert len(manifest) == 1
    assert manifest.column("width").to_pylist() == [-1]
    assert manifest.column("height").to_pylist() == [-1]

    meta_path = manifest_path.with_name(manifest_path.stem + "_meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["total_source_rows"] == 1
    assert meta["failed_messages"] == 0
    assert meta["failed_image_maps"] == 1
    assert meta["failed_dims"] == 1
