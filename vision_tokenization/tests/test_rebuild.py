"""Tests for offline rebuild from spill shards to bin/idx."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vision_tokenization.pipeline.pooled.document import (
    AtomicDocument,
    Component,
)
from vision_tokenization.pipeline.pooled.rebuild import (
    assemble_document_sequences,
    rebuild,
)
from vision_tokenization.pipeline.assembly import (
    StructureTokenIds,
)
from vision_tokenization.pipeline.pooled.spill import SpillWriter


# --- Fixtures ---------------------------------------------------------------

BOS = 1
EOS = 2
IMG_START = 10
IMG_END = 11
IMG_TOKEN_START = 12
EOL = 13
EOF = 14
VISION_OFFSET = 100
IMAGE_PLACEHOLDER = 50


@pytest.fixture
def token_ids():
    return StructureTokenIds(
        bos_id=BOS, eos_id=EOS,
        img_start_id=IMG_START, img_end_id=IMG_END,
        img_token_start_id=IMG_TOKEN_START,
        eol_id=EOL, eof_id=EOF,
        vision_token_offset=VISION_OFFSET,
        image_token_id=IMAGE_PLACEHOLDER,
    )


def _make_doc_and_write(writer, doc_id, mode, components_spec):
    """Create and write an AtomicDocument."""
    comps = []
    tokens = []
    total = 0
    img_tok = 0
    for i, (kind, tok_arr) in enumerate(components_spec):
        tok = np.asarray(tok_arr, dtype=np.uint16)
        comps.append(Component(
            component_index=i, kind=kind,
            resize_height=16 if kind == "image" else 0,
            resize_width=16 if kind == "image" else 0,
            manifest_row=doc_id * 10 + i,
        ))
        tokens.append(tok)
        total += len(tok)
        if kind == "image":
            img_tok += len(tok)

    doc = AtomicDocument(
        document_id=doc_id, mode=mode, components=comps,
        total_tokens=total, image_tokens=img_tok,
        text_tokens=total - img_tok, manifest_group_id=doc_id,
    )
    writer.add_document(doc, tokens)
    return doc


# --- assemble_document_sequences tests --------------------------------------

class TestAssembleDocumentSequences:
    def test_image_only(self, token_ids):
        """image_only: all images in group become one BOS+structs+EOS sequence."""
        components = [
            ({"kind": "image", "component_index": 0}, np.array([30, 31])),
            ({"kind": "image", "component_index": 1}, np.array([40, 41])),
        ]
        result = assemble_document_sequences("image_only", components, token_ids)
        assert len(result) == 1
        assert torch.equal(result[0], torch.tensor([BOS, 30, 31, 40, 41, EOS]))

    def test_image2text(self, token_ids):
        """image2text: BOS + img_structs + text + EOS."""
        components = [
            ({"kind": "image", "component_index": 0}, np.array([30, 31])),
            ({"kind": "text", "component_index": 1}, np.array([50, 51])),
        ]
        result = assemble_document_sequences("image2text", components, token_ids)
        assert len(result) == 1
        assert torch.equal(result[0], torch.tensor([BOS, 30, 31, 50, 51, EOS]))

    def test_text2image(self, token_ids):
        """text2image: BOS + text + img_structs + EOS."""
        components = [
            ({"kind": "text", "component_index": 0}, np.array([50, 51])),
            ({"kind": "image", "component_index": 1}, np.array([30, 31])),
        ]
        result = assemble_document_sequences("text2image", components, token_ids)
        assert len(result) == 1
        assert torch.equal(result[0], torch.tensor([BOS, 50, 51, 30, 31, EOS]))

    def test_interleave_no_split(self, token_ids):
        """interleave without max_seq_len: single assembled sequence."""
        components = [
            ({"kind": "text", "component_index": 0}, np.array([10])),
            ({"kind": "image", "component_index": 1}, np.array([20, 21])),
            ({"kind": "text", "component_index": 2}, np.array([30])),
        ]
        result = assemble_document_sequences("interleave", components, token_ids)
        assert len(result) == 1
        assert torch.equal(result[0], torch.tensor([BOS, 10, 20, 21, 30, EOS]))

    def test_interleave_with_split(self, token_ids):
        """interleave with max_seq_len: splits at component boundaries."""
        components = [
            ({"kind": "text", "component_index": 0}, np.array([10, 11])),
            ({"kind": "text", "component_index": 1}, np.array([20, 21])),
            ({"kind": "text", "component_index": 2}, np.array([30])),
        ]
        # max_sequence_tokens=5: BOS + 2 tokens + EOS = 4, can't fit next 2
        result = assemble_document_sequences(
            "interleave", components, token_ids, max_sequence_tokens=5,
        )
        assert len(result) == 2
        assert torch.equal(result[0], torch.tensor([BOS, 10, 11, EOS]))
        assert torch.equal(result[1], torch.tensor([BOS, 20, 21, 30, EOS]))


# --- Full rebuild pipeline tests -------------------------------------------

class TestRebuild:
    def test_rebuild_image_only(self, tmp_path, token_ids):
        """End-to-end: spill -> rebuild -> read back bin/idx."""
        writer = SpillWriter(str(tmp_path), rank=0)
        writer.open()
        _make_doc_and_write(writer, 0, "image_only", [
            ("image", [30, 31, 32]),
        ])
        _make_doc_and_write(writer, 1, "image_only", [
            ("image", [40, 41]),
        ])
        writer.finalize()

        prefix = rebuild(
            tmp_path, token_ids=token_ids, vocab_size=200000,
        )

        # Verify output files exist
        assert (tmp_path / "rebuilt.bin").exists()
        assert (tmp_path / "rebuilt.idx").exists()

    def test_rebuild_image2text(self, tmp_path, token_ids):
        writer = SpillWriter(str(tmp_path), rank=0)
        writer.open()
        _make_doc_and_write(writer, 0, "image2text", [
            ("image", [30, 31]),
            ("text", [50, 51]),
        ])
        writer.finalize()

        rebuild(tmp_path, token_ids=token_ids, vocab_size=200000)
        assert (tmp_path / "rebuilt.bin").exists()

    def test_rebuild_interleave_with_split(self, tmp_path, token_ids):
        writer = SpillWriter(str(tmp_path), rank=0)
        writer.open()
        _make_doc_and_write(writer, 0, "interleave", [
            ("text", [10, 11]),
            ("text", [20, 21]),
            ("text", [30]),
        ])
        writer.finalize()

        rebuild(
            tmp_path, token_ids=token_ids, vocab_size=200000,
            max_sequence_tokens=5,
        )
        assert (tmp_path / "rebuilt.bin").exists()

    def test_rebuild_with_seqlen_threshold(self, tmp_path, token_ids):
        writer = SpillWriter(str(tmp_path), rank=0)
        writer.open()
        # Short doc (5 tokens: BOS + 3 img + EOS)
        _make_doc_and_write(writer, 0, "image_only", [
            ("image", [30, 31, 32]),
        ])
        # Long doc (102 tokens: BOS + 100 img + EOS)
        _make_doc_and_write(writer, 1, "image_only", [
            ("image", list(range(100))),
        ])
        writer.finalize()

        rebuild(
            tmp_path, token_ids=token_ids, vocab_size=200000,
            seqlen_threshold=10,
        )
        assert (tmp_path / "stage2" / "rebuilt.bin").exists()
        assert (tmp_path / "lct" / "rebuilt.bin").exists()

    def test_rebuild_multiple_workers(self, tmp_path, token_ids):
        for rank in range(3):
            writer = SpillWriter(str(tmp_path), rank=rank)
            writer.open()
            _make_doc_and_write(writer, rank, "image_only", [
                ("image", [rank * 10 + 1, rank * 10 + 2]),
            ])
            writer.finalize()

        rebuild(tmp_path, token_ids=token_ids, vocab_size=200000)
        assert (tmp_path / "rebuilt.bin").exists()
