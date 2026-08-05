from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from vision_tokenization.common.assembly import StructureTokenIds
from vision_tokenization.discrete.conversation import ConversationPolicy
from vision_tokenization.discrete.emu.sft import EMUSftTokenizer
from vision_tokenization.discrete.sft_segments import ChatTemplateSFTDocumentRenderer
from vision_tokenization.indexing.planning.tokenization_plan import (
    IMAGE,
    TEXT,
    ComponentIndex,
    DocumentIndex,
    ExecutionPlan,
    PlanMetadata,
    TokenizationPlan,
)
from vision_tokenization.pipeline.output.backend import SpillBackend
from vision_tokenization.pipeline.output.rebuild import _assemble_document, rebuild_rank
from vision_tokenization.pipeline.output.spill import ComponentSpillWriter
from vision_tokenization.pipeline.runtime.checkpoint import WorkerStats


def _messages_key(messages):
    return tuple((msg["role"], msg["content"]) for msg in messages)


class _DummyChatTokenizer:
    """Small tokenizer stub for rendered SFT segment tests."""

    def __init__(self, render_map, token_map):
        self.render_map = render_map
        self.token_map = token_map
        self.batch_calls = []

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, **kwargs):
        assert tokenize is False
        assert add_generation_prompt is False
        return self.render_map[_messages_key(messages)]

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        self.batch_calls.append(list(texts))
        return {"input_ids": [self.token_map[text] for text in texts]}

    @staticmethod
    def convert_tokens_to_ids(token):
        return 99 if token == "<|image|>" else -1

    @staticmethod
    def convert_ids_to_tokens(token_id):
        return "<|image|>" if token_id == 99 else "<unk>"

    @staticmethod
    def decode(token_ids, skip_special_tokens=False):
        return "<|image|>" if list(token_ids) == [99] else ""


def _make_stub_sft_tokenizer(*, render_map, token_map, image_rows):
    text_tokenizer = _DummyChatTokenizer(render_map, token_map)
    tokenizer = object.__new__(EMUSftTokenizer)
    tokenizer.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="TestSFTPool")
    tokenizer.bos_id = 1
    tokenizer.eos_id = 2
    tokenizer.text_tokenizer = text_tokenizer
    tokenizer.conversation_policy = ConversationPolicy()
    tokenizer._sft_renderer = ChatTemplateSFTDocumentRenderer(
        text_tokenizer=text_tokenizer,
        conversation_policy=tokenizer.conversation_policy,
    )
    tokenizer.tokenize_images = lambda images, resize_size: torch.tensor(image_rows, dtype=torch.long)
    return tokenizer, text_tokenizer


class _FakeSpillWriter:
    def __init__(self):
        self.components = []

    def add_component(self, **kwargs):
        self.components.append(kwargs)


class _FakeSpillSFTTokenizer:
    """Stub tokenizer exposing only the SFT spill contract (render_sft_document)."""

    bos_id = 1
    eos_id = 2

    def __init__(self, text_tokenizer):
        self.text_tokenizer = text_tokenizer
        self.expected_num_images_calls: list = []
        self._renderer = ChatTemplateSFTDocumentRenderer(
            text_tokenizer=text_tokenizer,
            conversation_policy=ConversationPolicy(),
        )

    def render_sft_document(self, raw_text, *, expected_num_images=None):
        self.expected_num_images_calls.append(expected_num_images)
        return self._renderer.render_document(
            raw_text, expected_num_images=expected_num_images,
        )


def test_sft_tokenize_batch_batches_all_text_spans_once():
    raw_texts = [
        [
            {"from": "human", "value": "doc0 prompt"},
            {"from": "gpt", "value": "doc0 answer"},
        ],
        [
            {"from": "human", "value": "doc1 prompt"},
            {"from": "gpt", "value": "doc1 answer"},
        ],
    ]
    render_map = {
        _messages_key(
            [
                {"role": "user", "content": "doc0 prompt"},
                {"role": "assistant", "content": "doc0 answer"},
            ]
        ): "A<|image|>B<|image|>C",
        _messages_key(
            [
                {"role": "user", "content": "doc1 prompt"},
                {"role": "assistant", "content": "doc1 answer"},
            ]
        ): "D<|image|>E",
    }
    token_map = {
        "A": [1, 10],
        "B": [11],
        "C": [12, 2],
        "D": [20],
        "E": [21],
    }
    tokenizer, text_tokenizer = _make_stub_sft_tokenizer(
        render_map=render_map,
        token_map=token_map,
        image_rows=[
            [1, 200, 201, 2],
            [1, 210, 211, 2],
            [1, 220, 221, 2],
        ],
    )
    try:
        outputs = tokenizer.tokenize_batch(
            images=["img0", "img1", "img2"],
            resize_size=(16, 16),
            text=raw_texts,
            group_slices=np.array([[0, 2], [2, 3]], dtype=np.int64),
        )
    finally:
        tokenizer.executor.shutdown(wait=True)

    assert text_tokenizer.batch_calls == [["A", "B", "C", "D", "E"]]
    assert [out.tolist() if out is not None else None for out in outputs] == [
        [1, 10, 200, 201, 11, 210, 211, 12, 2],
        [1, 20, 220, 221, 21, 2],
    ]


def test_sft_tokenize_batch_skips_group_on_image_slot_mismatch():
    raw_texts = [
        [
            {"from": "human", "value": "doc prompt"},
            {"from": "gpt", "value": "doc answer"},
        ]
    ]
    render_map = {
        _messages_key(
            [
                {"role": "user", "content": "doc prompt"},
                {"role": "assistant", "content": "doc answer"},
            ]
        ): "A<|image|>B",
    }
    token_map = {"A": [10], "B": [11]}
    tokenizer, text_tokenizer = _make_stub_sft_tokenizer(
        render_map=render_map,
        token_map=token_map,
        image_rows=[
            [1, 200, 2],
            [1, 210, 2],
        ],
    )
    try:
        outputs = tokenizer.tokenize_batch(
            images=["img0", "img1"],
            resize_size=(16, 16),
            text=raw_texts,
            group_slices=np.array([[0, 2]], dtype=np.int64),
        )
    finally:
        tokenizer.executor.shutdown(wait=True)

    assert outputs == [None]
    assert text_tokenizer.batch_calls == []


def test_spill_backend_writes_segmented_sft_components_in_runtime_order():
    raw_texts = [
        [
            {"from": "human", "value": "doc prompt"},
            {"from": "gpt", "value": "doc answer"},
        ]
    ]
    render_map = {
        _messages_key(
            [
                {"role": "user", "content": "doc prompt"},
                {"role": "assistant", "content": "doc answer"},
            ]
        ): "A<|image|>B<|image|>C",
    }
    token_map = {
        "A": [10],
        "B": [11],
        "C": [12],
    }
    text_tokenizer = _DummyChatTokenizer(render_map, token_map)

    plan = TokenizationPlan(
        documents=DocumentIndex(
            document_id=np.array([0], dtype=np.int64),
            output_order=np.array([0], dtype=np.int64),
            num_images=np.array([2], dtype=np.int16),
        ),
        components=ComponentIndex(
            document_id=np.array([0, 0], dtype=np.int64),
            component_index=np.array([0, 1], dtype=np.int16),
            kind=np.array([IMAGE, IMAGE], dtype=np.int8),
            source_kind=np.array([0, 0], dtype=np.int8),
            source_ref=np.array([0, 1], dtype=np.int64),
            image_index=np.array([0, 1], dtype=np.int16),
        ),
        execution=ExecutionPlan(),
        metadata=PlanMetadata(manifest_path="x", manifest_fingerprint="y", mode="sft"),
    )

    backend = SpillBackend()
    backend._writer = _FakeSpillWriter()
    stats = WorkerStats()

    backend._write_sft_batch(
        image_tokens=torch.tensor([[1, 200, 2], [1, 210, 2]], dtype=torch.long),
        texts=raw_texts,
        component_indices=np.array([0, 1], dtype=np.int64),
        group_slices=np.array([[0, 2]], dtype=np.int64),
        resize_height=32,
        resize_width=32,
        plan=plan,
        tokenizer=_FakeSpillSFTTokenizer(text_tokenizer),
        stats=stats,
    )

    written = sorted(
        backend._writer.components,
        key=lambda row: (row["component_index"], row["kind"]),
    )
    assert [
        (row["component_index"], row["kind"], row["tokens"].tolist())
        for row in written
    ] == [
        (0, int(TEXT), [10]),
        (1, int(IMAGE), [200]),
        (2, int(TEXT), [11]),
        (3, int(IMAGE), [210]),
        (4, int(TEXT), [12]),
    ]
    assert stats.samples_processed == 2
    assert stats.text_tokens == 3
    assert stats.image_tokens == 2
    assert stats.tokens_generated == 5


def test_spill_backend_renders_sft_with_full_document_image_count_for_fragments():
    raw_texts = [
        [
            {"from": "human", "value": "doc prompt"},
            {"from": "gpt", "value": "doc answer"},
        ]
    ]
    render_map = {
        _messages_key(
            [
                {"role": "user", "content": "doc prompt"},
                {"role": "assistant", "content": "doc answer"},
            ]
        ): "A<|image|>B<|image|>C<|image|>D<|image|>E",
    }
    token_map = {
        "A": [10],
        "B": [11],
        "C": [12],
        "D": [13],
        "E": [14],
    }
    text_tokenizer = _DummyChatTokenizer(render_map, token_map)

    plan = TokenizationPlan(
        documents=DocumentIndex(
            document_id=np.array([0], dtype=np.int64),
            output_order=np.array([0], dtype=np.int64),
            num_images=np.array([4], dtype=np.int16),
        ),
        components=ComponentIndex(
            document_id=np.array([0, 0], dtype=np.int64),
            component_index=np.array([0, 1], dtype=np.int16),
            kind=np.array([IMAGE, IMAGE], dtype=np.int8),
            source_kind=np.array([0, 0], dtype=np.int8),
            source_ref=np.array([0, 1], dtype=np.int64),
            image_index=np.array([0, 1], dtype=np.int16),
        ),
        execution=ExecutionPlan(),
        metadata=PlanMetadata(manifest_path="x", manifest_fingerprint="y", mode="sft"),
    )

    backend = SpillBackend()
    backend._writer = _FakeSpillWriter()
    stats = WorkerStats()
    tokenizer = _FakeSpillSFTTokenizer(text_tokenizer)

    backend._write_sft_batch(
        image_tokens=torch.tensor([[1, 200, 2], [1, 210, 2]], dtype=torch.long),
        texts=raw_texts,
        component_indices=np.array([0, 1], dtype=np.int64),
        group_slices=np.array([[0, 2]], dtype=np.int64),
        resize_height=32,
        resize_width=32,
        plan=plan,
        tokenizer=tokenizer,
        stats=stats,
    )

    assert tokenizer.expected_num_images_calls == [4]
    written = sorted(
        backend._writer.components,
        key=lambda row: (row["component_index"], row["kind"]),
    )
    assert [
        (row["component_index"], row["kind"], row["tokens"].tolist())
        for row in written
    ] == [
        (0, int(TEXT), [10]),
        (1, int(IMAGE), [200]),
        (2, int(TEXT), [11]),
        (3, int(IMAGE), [210]),
        (4, int(TEXT), [12]),
        (6, int(TEXT), [13]),
        (8, int(TEXT), [14]),
    ]
    assert stats.samples_processed == 2
    assert stats.samples_skipped == 0
    assert stats.text_tokens == 5
    assert stats.image_tokens == 2
    assert stats.tokens_generated == 7


def test_spill_backend_drops_sft_doc_after_text_owning_fragment_is_lost():
    raw_texts = [
        [
            {"from": "human", "value": "doc prompt"},
            {"from": "gpt", "value": "doc answer"},
        ]
    ]
    render_map = {
        _messages_key(
            [
                {"role": "user", "content": "doc prompt"},
                {"role": "assistant", "content": "doc answer"},
            ]
        ): "A<|image|>B<|image|>C<|image|>D<|image|>E",
    }
    token_map = {
        "A": [10],
        "B": [11],
        "C": [12],
        "D": [13],
        "E": [14],
    }
    text_tokenizer = _DummyChatTokenizer(render_map, token_map)

    plan = TokenizationPlan(
        documents=DocumentIndex(
            document_id=np.array([0], dtype=np.int64),
            output_order=np.array([0], dtype=np.int64),
            num_images=np.array([4], dtype=np.int16),
        ),
        components=ComponentIndex(
            document_id=np.array([0, 0, 0, 0], dtype=np.int64),
            component_index=np.array([0, 1, 2, 3], dtype=np.int16),
            kind=np.array([IMAGE, IMAGE, IMAGE, IMAGE], dtype=np.int8),
            source_kind=np.array([0, 0, 0, 0], dtype=np.int8),
            source_ref=np.array([0, 1, 2, 3], dtype=np.int64),
            image_index=np.array([0, 1, 2, 3], dtype=np.int16),
        ),
        execution=ExecutionPlan(),
        metadata=PlanMetadata(manifest_path="x", manifest_fingerprint="y", mode="sft"),
    )

    backend = SpillBackend()
    backend._writer = _FakeSpillWriter()
    stats = WorkerStats()
    tokenizer = _FakeSpillSFTTokenizer(text_tokenizer)

    backend._write_sft_batch(
        image_tokens=torch.tensor([[1, 200, 2], [1, 210, 2]], dtype=torch.long),
        texts=[None],
        component_indices=np.array([0, 1], dtype=np.int64),
        group_slices=np.array([[0, 2]], dtype=np.int64),
        resize_height=32,
        resize_width=32,
        plan=plan,
        tokenizer=tokenizer,
        stats=stats,
    )

    backend._write_sft_batch(
        image_tokens=torch.tensor([[1, 220, 2], [1, 230, 2]], dtype=torch.long),
        texts=raw_texts,
        component_indices=np.array([2, 3], dtype=np.int64),
        group_slices=np.array([[0, 2]], dtype=np.int64),
        resize_height=32,
        resize_width=32,
        plan=plan,
        tokenizer=tokenizer,
        stats=stats,
    )

    assert backend._writer.components == []
    assert stats.samples_processed == 0
    assert stats.samples_skipped == 1
    assert stats.text_tokens == 0
    assert stats.image_tokens == 0
    assert stats.tokens_generated == 0


def test_rebuild_assemble_document_sft_uses_runtime_component_order():
    token_ids = StructureTokenIds(
        bos_id=1,
        eos_id=2,
        img_start_id=10,
        img_end_id=11,
        img_token_start_id=12,
        eol_id=13,
        eof_id=14,
        vision_token_offset=100,
        image_token_id=50,
    )
    components = [
        ({"kind": int(TEXT)}, torch.tensor([1, 10], dtype=torch.long)),
        ({"kind": int(IMAGE)}, torch.tensor([20, 21], dtype=torch.long)),
        ({"kind": int(TEXT)}, torch.tensor([30], dtype=torch.long)),
        ({"kind": int(IMAGE)}, torch.tensor([40], dtype=torch.long)),
        ({"kind": int(TEXT)}, torch.tensor([60, 2], dtype=torch.long)),
    ]

    sequences = _assemble_document(
        mode="sft",
        components=components,
        token_ids=token_ids,
        expected_num_images=2,
    )

    assert len(sequences) == 1
    assert sequences[0].tolist() == [1, 10, 20, 21, 30, 40, 60, 2]


def test_rebuild_assemble_document_sft_skips_incomplete_doc_without_text():
    token_ids = StructureTokenIds(
        bos_id=1,
        eos_id=2,
        img_start_id=10,
        img_end_id=11,
        img_token_start_id=12,
        eol_id=13,
        eof_id=14,
        vision_token_offset=100,
        image_token_id=50,
    )
    components = [
        ({"kind": int(IMAGE)}, torch.tensor([20, 21], dtype=torch.long)),
        ({"kind": int(IMAGE)}, torch.tensor([40], dtype=torch.long)),
    ]

    sequences = _assemble_document(
        mode="sft",
        components=components,
        token_ids=token_ids,
        expected_num_images=2,
    )

    assert sequences == []


def test_rebuild_assemble_document_sft_skips_incomplete_doc_missing_later_images():
    token_ids = StructureTokenIds(
        bos_id=1,
        eos_id=2,
        img_start_id=10,
        img_end_id=11,
        img_token_start_id=12,
        eol_id=13,
        eof_id=14,
        vision_token_offset=100,
        image_token_id=50,
    )
    components = [
        ({"kind": int(TEXT)}, torch.tensor([101], dtype=torch.long)),
        ({"kind": int(IMAGE)}, torch.tensor([20, 21], dtype=torch.long)),
        ({"kind": int(TEXT)}, torch.tensor([102], dtype=torch.long)),
        ({"kind": int(IMAGE)}, torch.tensor([40], dtype=torch.long)),
        ({"kind": int(TEXT)}, torch.tensor([103], dtype=torch.long)),
        ({"kind": int(TEXT)}, torch.tensor([104], dtype=torch.long)),
        ({"kind": int(TEXT)}, torch.tensor([105], dtype=torch.long)),
    ]

    sequences = _assemble_document(
        mode="sft",
        components=components,
        token_ids=token_ids,
        expected_num_images=4,
    )

    assert sequences == []


def test_rebuild_rank_skips_incomplete_sft_spill_document(tmp_path):
    token_ids = StructureTokenIds(
        bos_id=1,
        eos_id=2,
        img_start_id=10,
        img_end_id=11,
        img_token_start_id=12,
        eol_id=13,
        eof_id=14,
        vision_token_offset=100,
        image_token_id=50,
    )
    plan = TokenizationPlan(
        documents=DocumentIndex(
            document_id=np.array([0], dtype=np.int64),
            output_order=np.array([0], dtype=np.int64),
            num_images=np.array([2], dtype=np.int16),
        ),
        components=ComponentIndex(
            document_id=np.array([0, 0], dtype=np.int64),
            component_index=np.array([0, 1], dtype=np.int16),
            kind=np.array([IMAGE, IMAGE], dtype=np.int8),
            source_kind=np.array([0, 0], dtype=np.int8),
            source_ref=np.array([0, 1], dtype=np.int64),
            image_index=np.array([0, 1], dtype=np.int16),
        ),
        execution=ExecutionPlan(),
        metadata=PlanMetadata(manifest_path="x", manifest_fingerprint="y", mode="sft"),
    )

    output_dir = tmp_path / "spill_rebuild_sft"
    writer = ComponentSpillWriter(str(output_dir), rank=0, token_dtype=np.int32)
    writer.open()
    writer.add_component(
        document_id=0,
        component_index=1,
        kind=int(IMAGE),
        tokens=np.array([200, 201], dtype=np.int32),
    )
    writer.add_component(
        document_id=0,
        component_index=3,
        kind=int(IMAGE),
        tokens=np.array([210, 211], dtype=np.int32),
    )
    writer.finalize()
    # _SUCCESS is owned by the backend layer; rebuild refuses rank dirs without it.
    (output_dir / "rank_0000" / "_SUCCESS").touch()

    result = rebuild_rank(
        plan,
        rank=0,
        spill_dir=output_dir,
        token_ids=token_ids,
        vocab_size=200000,
    )

    assert result["rank"] == 0
    assert result["sequences"] == 0
    assert result["tokens"] == 0
    assert result["rejected_documents"] == 1
    assert not (output_dir / "rank_0000_chunk_0000.bin").exists()
    assert not (output_dir / "rank_0000_chunk_0000.idx").exists()


def test_rebuild_rank_skips_sft_spill_document_missing_later_fragment_images(tmp_path):
    token_ids = StructureTokenIds(
        bos_id=1,
        eos_id=2,
        img_start_id=10,
        img_end_id=11,
        img_token_start_id=12,
        eol_id=13,
        eof_id=14,
        vision_token_offset=100,
        image_token_id=50,
    )
    plan = TokenizationPlan(
        documents=DocumentIndex(
            document_id=np.array([0], dtype=np.int64),
            output_order=np.array([0], dtype=np.int64),
            num_images=np.array([4], dtype=np.int16),
        ),
        components=ComponentIndex(
            document_id=np.array([0, 0, 0, 0], dtype=np.int64),
            component_index=np.array([0, 1, 2, 3], dtype=np.int16),
            kind=np.array([IMAGE, IMAGE, IMAGE, IMAGE], dtype=np.int8),
            source_kind=np.array([0, 0, 0, 0], dtype=np.int8),
            source_ref=np.array([0, 1, 2, 3], dtype=np.int64),
            image_index=np.array([0, 1, 2, 3], dtype=np.int16),
        ),
        execution=ExecutionPlan(),
        metadata=PlanMetadata(manifest_path="x", manifest_fingerprint="y", mode="sft"),
    )

    output_dir = tmp_path / "spill_rebuild_sft_missing_late_fragment"
    writer = ComponentSpillWriter(str(output_dir), rank=0, token_dtype=np.int32)
    writer.open()
    writer.add_component(
        document_id=0,
        component_index=0,
        kind=int(TEXT),
        tokens=np.array([101], dtype=np.int32),
    )
    writer.add_component(
        document_id=0,
        component_index=1,
        kind=int(IMAGE),
        tokens=np.array([200, 201], dtype=np.int32),
    )
    writer.add_component(
        document_id=0,
        component_index=2,
        kind=int(TEXT),
        tokens=np.array([102], dtype=np.int32),
    )
    writer.add_component(
        document_id=0,
        component_index=3,
        kind=int(IMAGE),
        tokens=np.array([210, 211], dtype=np.int32),
    )
    writer.add_component(
        document_id=0,
        component_index=4,
        kind=int(TEXT),
        tokens=np.array([103], dtype=np.int32),
    )
    writer.add_component(
        document_id=0,
        component_index=6,
        kind=int(TEXT),
        tokens=np.array([104], dtype=np.int32),
    )
    writer.add_component(
        document_id=0,
        component_index=8,
        kind=int(TEXT),
        tokens=np.array([105], dtype=np.int32),
    )
    writer.finalize()
    # _SUCCESS is owned by the backend layer; rebuild refuses rank dirs without it.
    (output_dir / "rank_0000" / "_SUCCESS").touch()

    result = rebuild_rank(
        plan,
        rank=0,
        spill_dir=output_dir,
        token_ids=token_ids,
        vocab_size=200000,
    )

    assert result["rank"] == 0
    assert result["sequences"] == 0
    assert result["tokens"] == 0
    assert result["rejected_documents"] == 1
    assert not (output_dir / "rank_0000_chunk_0000.bin").exists()
    assert not (output_dir / "rank_0000_chunk_0000.idx").exists()
