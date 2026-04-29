"""Tests for extracted sequence assembly helpers.

Verifies that the pure functions in sequence_assembly.py produce identical
output to the inline code they replace in the tokenizer classes.
"""

from __future__ import annotations

import pytest
import torch

from vision_tokenization.common.assembly import (
    StructureTokenIds,
    assemble_image2text,
    assemble_interleaved_sequence,
    assemble_sequence,
    assemble_sft_sequence,
    assemble_text2image,
    encapsulate_image_structure,
    encapsulate_image_structure_batch,
    ensure_bos_eos,
    replace_image_placeholders,
    split_interleaved_sequence,
)


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


def _dim_tokens_fn(h: int, w: int):
    """Fake dim tokens: just encode 'HxW' as [h, w]."""
    return [h, w]


@pytest.fixture
def token_ids():
    return StructureTokenIds(
        bos_id=BOS,
        eos_id=EOS,
        img_start_id=IMG_START,
        img_end_id=IMG_END,
        img_token_start_id=IMG_TOKEN_START,
        eol_id=EOL,
        eof_id=EOF,
        vision_token_offset=VISION_OFFSET,
        image_token_id=IMAGE_PLACEHOLDER,
        dim_tokens_fn=_dim_tokens_fn,
    )


# --- encapsulate_image_structure tests --------------------------------------

class TestEncapsulateImageStructure:
    def test_basic_2x2(self, token_ids):
        indices = torch.tensor([0, 1, 2, 3])
        result = encapsulate_image_structure(indices, 2, 2, token_ids)

        # Expected: img_start + dim(2,2) + img_token_start +
        #   row0: (0+100, 1+100, EOL) + row1: (2+100, 3+100, EOL) +
        #   EOF + img_end
        expected = torch.tensor([
            IMG_START, 2, 2, IMG_TOKEN_START,
            100, 101, EOL,
            102, 103, EOL,
            EOF, IMG_END,
        ])
        assert torch.equal(result, expected)

    def test_no_bos_eos(self, token_ids):
        """Structure tokens should NOT contain BOS or EOS."""
        indices = torch.tensor([5, 6, 7, 8])
        result = encapsulate_image_structure(indices, 2, 2, token_ids)
        assert result[0].item() != BOS
        assert result[-1].item() != EOS

    def test_1x1(self, token_ids):
        indices = torch.tensor([42])
        result = encapsulate_image_structure(indices, 1, 1, token_ids)
        expected = torch.tensor([
            IMG_START, 1, 1, IMG_TOKEN_START,
            142, EOL,
            EOF, IMG_END,
        ])
        assert torch.equal(result, expected)

    def test_dimension_mismatch_raises(self, token_ids):
        indices = torch.tensor([0, 1, 2])
        with pytest.raises(AssertionError):
            encapsulate_image_structure(indices, 2, 2, token_ids)


class TestEncapsulateImageStructureBatch:
    def test_batch_matches_single(self, token_ids):
        """Batched version should produce same results as single."""
        indices_batch = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        batch_result = encapsulate_image_structure_batch(indices_batch, 2, 2, token_ids)

        for i in range(2):
            single_result = encapsulate_image_structure(
                indices_batch[i], 2, 2, token_ids,
            )
            assert torch.equal(batch_result[i], single_result)

    def test_no_bos_eos_in_batch(self, token_ids):
        indices = torch.tensor([[0, 1, 2, 3]])
        result = encapsulate_image_structure_batch(indices, 2, 2, token_ids)
        assert result[0, 0].item() != BOS
        assert result[0, -1].item() != EOS


# --- assemble_sequence tests ------------------------------------------------

class TestAssembleSequence:
    def test_basic(self):
        c1 = torch.tensor([10, 11])
        c2 = torch.tensor([20, 21, 22])
        result = assemble_sequence(bos_id=BOS, eos_id=EOS, component_tokens=[c1, c2])
        expected = torch.tensor([BOS, 10, 11, 20, 21, 22, EOS])
        assert torch.equal(result, expected)

    def test_empty_components(self):
        result = assemble_sequence(bos_id=BOS, eos_id=EOS, component_tokens=[])
        expected = torch.tensor([BOS, EOS])
        assert torch.equal(result, expected)


# --- assemble_interleaved_sequence tests ------------------------------------

class TestAssembleInterleavedSequence:
    def test_text_only(self):
        segments = [{"type": "text", "text": "hello"}]
        text_chunks = [torch.tensor([10, 11])]
        result = assemble_interleaved_sequence(
            bos_id=BOS, eos_id=EOS,
            segments=segments,
            text_token_chunks=text_chunks,
            image_token_chunks=[],
        )
        assert torch.equal(result, torch.tensor([BOS, 10, 11, EOS]))

    def test_image_only(self):
        segments = [{"type": "image"}]
        image_chunks = [torch.tensor([30, 31, 32])]
        result = assemble_interleaved_sequence(
            bos_id=BOS, eos_id=EOS,
            segments=segments,
            text_token_chunks=[],
            image_token_chunks=image_chunks,
        )
        assert torch.equal(result, torch.tensor([BOS, 30, 31, 32, EOS]))

    def test_interleaved_ordering(self):
        segments = [
            {"type": "text", "text": "a"},
            {"type": "image"},
            {"type": "text", "text": "b"},
        ]
        text_chunks = [torch.tensor([10]), torch.tensor([20])]
        image_chunks = [torch.tensor([30, 31])]
        result = assemble_interleaved_sequence(
            bos_id=BOS, eos_id=EOS,
            segments=segments,
            text_token_chunks=text_chunks,
            image_token_chunks=image_chunks,
        )
        assert torch.equal(result, torch.tensor([BOS, 10, 30, 31, 20, EOS]))

    def test_empty_text_skipped(self):
        """Segments with empty text should not consume a text chunk."""
        segments = [
            {"type": "text", "text": ""},
            {"type": "text", "text": "hello"},
        ]
        text_chunks = [torch.tensor([10])]
        result = assemble_interleaved_sequence(
            bos_id=BOS, eos_id=EOS,
            segments=segments,
            text_token_chunks=text_chunks,
            image_token_chunks=[],
        )
        assert torch.equal(result, torch.tensor([BOS, 10, EOS]))

    def test_mismatch_raises(self):
        segments = [{"type": "image"}]
        with pytest.raises(ValueError, match="image"):
            assemble_interleaved_sequence(
                bos_id=BOS, eos_id=EOS,
                segments=segments,
                text_token_chunks=[],
                image_token_chunks=[],
            )


# --- split_interleaved_sequence tests ---------------------------------------

class TestSplitInterleavedSequence:
    def test_no_limit_returns_single(self):
        segments = [{"type": "text", "text": "a"}, {"type": "image"}]
        texts = [torch.tensor([10, 11])]
        images = [torch.tensor([20, 21])]
        result = split_interleaved_sequence(
            bos_id=BOS, eos_id=EOS,
            segments=segments,
            text_token_chunks=texts,
            image_token_chunks=images,
        )
        assert len(result) == 1
        assert torch.equal(result[0], torch.tensor([BOS, 10, 11, 20, 21, EOS]))

    def test_splits_at_boundary(self):
        segments = [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
            {"type": "text", "text": "c"},
        ]
        texts = [torch.tensor([10, 11]), torch.tensor([20, 21]), torch.tensor([30])]

        # max_sequence_tokens=5: BOS(1) + text(2) + EOS(1) = 4, fits one text chunk
        result = split_interleaved_sequence(
            bos_id=BOS, eos_id=EOS,
            segments=segments,
            text_token_chunks=texts,
            image_token_chunks=[],
            max_sequence_tokens=5,
        )
        # First: BOS + [10,11] + EOS = 4 tokens
        # Can't fit [20,21] (would be 6) → flush
        # Second: BOS + [20,21] + EOS = 4
        # Can't fit [30] (would be 5) → actually 5 fits!
        # Let me recalculate: second starts at len=2, add [20,21]=2 → len=4, add [30]=1 → len=5 = limit → fits
        assert len(result) == 2
        assert torch.equal(result[0], torch.tensor([BOS, 10, 11, EOS]))
        assert torch.equal(result[1], torch.tensor([BOS, 20, 21, 30, EOS]))

    def test_single_segment_too_large_raises(self):
        segments = [{"type": "text", "text": "big"}]
        texts = [torch.tensor([10, 11, 12, 13, 14])]
        with pytest.raises(ValueError, match="exceeding"):
            split_interleaved_sequence(
                bos_id=BOS, eos_id=EOS,
                segments=segments,
                text_token_chunks=texts,
                image_token_chunks=[],
                max_sequence_tokens=5,
            )

    def test_empty_entries(self):
        result = split_interleaved_sequence(
            bos_id=BOS, eos_id=EOS,
            segments=[],
            text_token_chunks=[],
            image_token_chunks=[],
            max_sequence_tokens=10,
        )
        assert len(result) == 1
        assert torch.equal(result[0], torch.tensor([BOS, EOS]))


# --- replace_image_placeholders tests ---------------------------------------

class TestReplaceImagePlaceholders:
    def test_single_replacement(self):
        text = torch.tensor([BOS, 10, IMAGE_PLACEHOLDER, 20, EOS])
        img = torch.tensor([30, 31, 32])
        result = replace_image_placeholders(text, [2], [img])
        assert torch.equal(result, torch.tensor([BOS, 10, 30, 31, 32, 20, EOS]))

    def test_multiple_replacements(self):
        text = torch.tensor([BOS, IMAGE_PLACEHOLDER, 10, IMAGE_PLACEHOLDER, EOS])
        img1 = torch.tensor([30, 31])
        img2 = torch.tensor([40])
        result = replace_image_placeholders(text, [1, 3], [img1, img2])
        assert torch.equal(result, torch.tensor([BOS, 30, 31, 10, 40, EOS]))

    def test_no_placeholders(self):
        text = torch.tensor([BOS, 10, 20, EOS])
        result = replace_image_placeholders(text, [], [])
        assert torch.equal(result, text)

    def test_mismatch_raises(self):
        text = torch.tensor([BOS, IMAGE_PLACEHOLDER, EOS])
        with pytest.raises(ValueError):
            replace_image_placeholders(text, [1], [])


# --- ensure_bos_eos tests ---------------------------------------------------

class TestEnsureBosEos:
    def test_preserves_existing_wrapper(self):
        tokens = torch.tensor([BOS, 10, 11, EOS])
        result = ensure_bos_eos(tokens, bos_id=BOS, eos_id=EOS)
        assert torch.equal(result, tokens)

    def test_adds_missing_wrapper(self):
        tokens = torch.tensor([10, 11])
        result = ensure_bos_eos(tokens, bos_id=BOS, eos_id=EOS)
        assert torch.equal(result, torch.tensor([BOS, 10, 11, EOS]))

    def test_empty_sequence_becomes_bos_eos(self):
        tokens = torch.tensor([], dtype=torch.long)
        result = ensure_bos_eos(tokens, bos_id=BOS, eos_id=EOS)
        assert torch.equal(result, torch.tensor([BOS, EOS]))


# --- assemble_sft_sequence tests --------------------------------------------

class TestAssembleSFTSequence:
    def test_uses_existing_bos_eos_from_text_spans(self):
        segments = [
            {"type": "text", "text": "left"},
            {"type": "image"},
            {"type": "text", "text": "right"},
        ]
        text_chunks = [
            torch.tensor([BOS, 10]),
            torch.tensor([11, EOS]),
        ]
        image_chunks = [torch.tensor([30, 31])]
        result = assemble_sft_sequence(
            bos_id=BOS,
            eos_id=EOS,
            segments=segments,
            text_token_chunks=text_chunks,
            image_token_chunks=image_chunks,
        )
        assert torch.equal(result, torch.tensor([BOS, 10, 30, 31, 11, EOS]))

    def test_adds_missing_bos_eos_after_structured_concat(self):
        segments = [
            {"type": "text", "text": "left"},
            {"type": "image"},
            {"type": "text", "text": "right"},
        ]
        text_chunks = [torch.tensor([10]), torch.tensor([11])]
        image_chunks = [torch.tensor([30, 31])]
        result = assemble_sft_sequence(
            bos_id=BOS,
            eos_id=EOS,
            segments=segments,
            text_token_chunks=text_chunks,
            image_token_chunks=image_chunks,
        )
        assert torch.equal(result, torch.tensor([BOS, 10, 30, 31, 11, EOS]))


# --- Mode-specific assembly tests ------------------------------------------

class TestAssembleImage2Text:
    def test_basic(self):
        imgs = [torch.tensor([30, 31]), torch.tensor([40, 41])]
        text = torch.tensor([50, 51])
        result = assemble_image2text(
            bos_id=BOS, eos_id=EOS,
            image_structures=imgs, text_tokens=text,
        )
        assert torch.equal(result, torch.tensor([BOS, 30, 31, 40, 41, 50, 51, EOS]))


class TestAssembleText2Image:
    def test_basic(self):
        text = torch.tensor([50, 51])
        imgs = [torch.tensor([30, 31]), torch.tensor([40, 41])]
        result = assemble_text2image(
            bos_id=BOS, eos_id=EOS,
            text_tokens=text, image_structures=imgs,
        )
        assert torch.equal(result, torch.tensor([BOS, 50, 51, 30, 31, 40, 41, EOS]))
