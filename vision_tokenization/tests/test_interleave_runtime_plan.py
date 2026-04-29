from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from vision_tokenization.indexing.manifest import save_interleave_manifest
from vision_tokenization.indexing.planning.tokenization_plan import (
    ComponentIndex,
    DocumentIndex,
    ExecutionPlan,
    ImageBatch,
    ImageBatchTable,
    PlanMetadata,
    TokenizationPlan,
    _plan_image_batches,
    build_tokenization_plan,
)
from vision_tokenization.pipeline.output.backend import SpillBackend
from vision_tokenization.pipeline.runtime.checkpoint import WorkerStats
from vision_tokenization.common.assembly import StructureTokenIds
from vision_tokenization.pipeline.output.rebuild import rebuild_from_plan
from vision_tokenization.pipeline.output.spill import (
    ComponentSpillReader,
    ComponentSpillWriter,
)


def _write_interleave_manifest(tmp_path):
    manifest_path = tmp_path / "interleave_manifest.parquet"
    records = [
        {
            "tar_path": "a.tar",
            "offset_data": 0,
            "file_size": 10,
            "width": 32,
            "height": 32,
            "group_id": 100,
            "image_index": 0,
            "jsonl_path": "docs.jsonl",
            "line_start": 0,
            "line_length": 100,
            "image_ref": "img0.png",
        },
        {
            "tar_path": "a.tar",
            "offset_data": 10,
            "file_size": 10,
            "width": 32,
            "height": 32,
            "group_id": 100,
            "image_index": 1,
            "jsonl_path": "docs.jsonl",
            "line_start": 0,
            "line_length": 100,
            "image_ref": "img1.png",
        },
        {
            "tar_path": "a.tar",
            "offset_data": 20,
            "file_size": 10,
            "width": 32,
            "height": 32,
            "group_id": 101,
            "image_index": 0,
            "jsonl_path": "docs.jsonl",
            "line_start": 200,
            "line_length": 100,
            "image_ref": "img2.png",
        },
    ]
    save_interleave_manifest(records, manifest_path)
    return manifest_path


@pytest.fixture
def token_ids():
    return StructureTokenIds(
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


def test_interleave_plan_contains_only_image_components(tmp_path):
    manifest_path = _write_interleave_manifest(tmp_path)

    plan = build_tokenization_plan(
        manifest_path,
        mode="interleave",
        batch_size=8,
        max_batch_tokens=8192,
        spatial_factor=16,
        resize_min_pixels=256,
        resize_max_pixels=4096,
        window_size=100,
    )

    assert plan.mode == "interleave"
    assert plan.total_documents == 2
    assert plan.total_components == 3
    assert plan.total_text_components == 0
    assert plan.documents.num_images.tolist() == [2, 1]
    assert plan.components.component_index.tolist() == [0, 1, 0]



def test_rebuild_from_plan_interleave_uses_runtime_component_order(tmp_path, token_ids):
    manifest_path = _write_interleave_manifest(tmp_path)
    plan = build_tokenization_plan(
        manifest_path,
        mode="interleave",
        batch_size=8,
        max_batch_tokens=8192,
        spatial_factor=16,
        resize_min_pixels=256,
        resize_max_pixels=4096,
        window_size=100,
    )

    output_dir = tmp_path / "spill_rebuild"
    writer = ComponentSpillWriter(str(output_dir), rank=0, token_dtype=np.int32)
    writer.open()
    writer.add_component(document_id=0, component_index=0, kind=1, tokens=np.array([10], dtype=np.int32))
    writer.add_component(document_id=0, component_index=1, kind=0, tokens=np.array([20, 21], dtype=np.int32))
    writer.add_component(document_id=0, component_index=2, kind=1, tokens=np.array([30], dtype=np.int32))
    writer.add_component(document_id=0, component_index=3, kind=0, tokens=np.array([40, 41], dtype=np.int32))
    writer.add_component(document_id=1, component_index=0, kind=0, tokens=np.array([50, 51], dtype=np.int32))
    writer.finalize()

    result = rebuild_from_plan(
        plan,
        output_dir,
        token_ids=token_ids,
        vocab_size=200000,
        output_name="rebuilt_interleave",
    )

    prefix = Path(result["output_prefix"])
    assert prefix.with_suffix(".bin").exists()
    assert prefix.with_suffix(".idx").exists()


def test_plan_image_batches_empty_returns_empty_batches_and_offsets():
    batches, offsets = _plan_image_batches(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int64),
        batch_size=8,
        max_batch_tokens=8192,
        spatial_factor=16,
        resize_min_pixels=256,
        resize_max_pixels=4096,
        window_size=100,
    )

    assert batches == []
    assert offsets.dtype == np.int64
    assert offsets.shape == (0,)


def test_split_image_batches_uses_batch_level_when_every_doc_has_one_image():
    plan = TokenizationPlan(
        documents=DocumentIndex(
            document_id=np.array([0, 1, 2, 3], dtype=np.int64),
            output_order=np.array([0, 1, 2, 3], dtype=np.int64),
            num_images=np.array([1, 1, 1, 1], dtype=np.int16),
        ),
        components=ComponentIndex(
            document_id=np.array([0, 1, 2, 3], dtype=np.int64),
            component_index=np.zeros(4, dtype=np.int16),
            kind=np.zeros(4, dtype=np.int8),
            source_kind=np.zeros(4, dtype=np.int8),
            source_ref=np.arange(4, dtype=np.int64),
            image_index=np.zeros(4, dtype=np.int16),
        ),
        execution=ExecutionPlan(
            image_batches=ImageBatchTable.from_batches([
                ImageBatch(np.array([0]), 32, 32, 1),
                ImageBatch(np.array([1]), 32, 32, 1),
                ImageBatch(np.array([2]), 32, 32, 1),
                ImageBatch(np.array([3]), 32, 32, 1),
            ]),
            split_batch_offsets=np.array([0], dtype=np.int64),
        ),
    )

    splits = plan.split_image_batches_for_workers(2)
    assert [len(s) for s in splits] == [2, 2]


def test_split_image_batches_uses_segment_level_when_docs_have_multiple_images():
    plan = TokenizationPlan(
        documents=DocumentIndex(
            document_id=np.array([0, 1], dtype=np.int64),
            output_order=np.array([0, 1], dtype=np.int64),
            num_images=np.array([2, 1], dtype=np.int16),
        ),
        components=ComponentIndex(
            document_id=np.array([0, 0, 1], dtype=np.int64),
            component_index=np.array([0, 1, 0], dtype=np.int16),
            kind=np.zeros(3, dtype=np.int8),
            source_kind=np.zeros(3, dtype=np.int8),
            source_ref=np.arange(3, dtype=np.int64),
            image_index=np.array([0, 1, 0], dtype=np.int16),
        ),
        execution=ExecutionPlan(
            image_batches=ImageBatchTable.from_batches([
                ImageBatch(np.array([0]), 32, 32, 1),
                ImageBatch(np.array([1]), 32, 32, 1),
                ImageBatch(np.array([2]), 32, 32, 1),
                ImageBatch(np.array([2]), 32, 32, 1),
            ]),
            split_batch_offsets=np.array([0, 3], dtype=np.int64),
        ),
    )

    splits = plan.split_image_batches_for_workers(2)
    assert [len(s) for s in splits] == [3, 1]


def test_spill_backend_skips_nonstructured_interleave_payload():
    class _FakeWriter:
        def __init__(self):
            self.components = []

        def add_component(self, **kwargs):
            self.components.append(kwargs)

    class _FakeTokenizer:
        bos_id = 1
        eos_id = 2

        @staticmethod
        def text_tokenizer(texts, **kwargs):
            return {"input_ids": [[100 + i] for i, _ in enumerate(texts)]}

    plan = TokenizationPlan(
        documents=DocumentIndex(
            document_id=np.array([0], dtype=np.int64),
            output_order=np.array([0], dtype=np.int64),
            num_images=np.array([1], dtype=np.int16),
        ),
        components=ComponentIndex(
            document_id=np.array([0], dtype=np.int64),
            component_index=np.array([0], dtype=np.int16),
            kind=np.array([0], dtype=np.int8),
            source_kind=np.array([0], dtype=np.int8),
            source_ref=np.array([0], dtype=np.int64),
            image_index=np.array([0], dtype=np.int16),
        ),
        execution=ExecutionPlan(),
        metadata=PlanMetadata(manifest_path="x", manifest_fingerprint="y", mode="interleave", parser="medpix"),
    )

    backend = SpillBackend()
    backend._writer = _FakeWriter()
    stats = WorkerStats()

    backend._write_interleave_batch(
        image_tokens=[torch.tensor([1, 42, 2], dtype=torch.long)],
        texts=["before <|img0|> after"],
        component_indices=np.array([0], dtype=np.int64),
        group_slices=np.array([[0, 1]], dtype=np.int64),
        resize_height=32,
        resize_width=32,
        plan=plan,
        tokenizer=_FakeTokenizer(),
        stats=stats,
    )

    assert backend._writer.components == []
    assert stats.samples_skipped == 1
    assert stats.samples_processed == 0


def test_spill_backend_skips_interleave_doc_when_loader_returns_none():
    class _FakeWriter:
        def __init__(self):
            self.components = []

        def add_component(self, **kwargs):
            self.components.append(kwargs)

    class _FakeTokenizer:
        bos_id = 1
        eos_id = 2

        @staticmethod
        def text_tokenizer(texts, **kwargs):
            return {"input_ids": []}

    plan = TokenizationPlan(
        documents=DocumentIndex(
            document_id=np.array([0], dtype=np.int64),
            output_order=np.array([0], dtype=np.int64),
            num_images=np.array([1], dtype=np.int16),
        ),
        components=ComponentIndex(
            document_id=np.array([0], dtype=np.int64),
            component_index=np.array([0], dtype=np.int16),
            kind=np.array([0], dtype=np.int8),
            source_kind=np.array([0], dtype=np.int8),
            source_ref=np.array([0], dtype=np.int64),
            image_index=np.array([0], dtype=np.int16),
        ),
        execution=ExecutionPlan(),
        metadata=PlanMetadata(
            manifest_path="x",
            manifest_fingerprint="y",
            mode="interleave",
            parser="shizhen",
        ),
    )

    backend = SpillBackend()
    backend._writer = _FakeWriter()
    stats = WorkerStats()

    backend._write_interleave_batch(
        image_tokens=[torch.tensor([1, 42, 2], dtype=torch.long)],
        texts=[None],
        component_indices=np.array([0], dtype=np.int64),
        group_slices=np.array([[0, 1]], dtype=np.int64),
        resize_height=32,
        resize_width=32,
        plan=plan,
        tokenizer=_FakeTokenizer(),
        stats=stats,
    )

    assert backend._writer.components == []
    assert stats.samples_skipped == 1
    assert stats.samples_processed == 0
    assert stats.text_tokens == 0
    assert stats.image_tokens == 0
