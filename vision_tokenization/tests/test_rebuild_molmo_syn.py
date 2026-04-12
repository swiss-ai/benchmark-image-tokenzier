from __future__ import annotations

import numpy as np

from vision_tokenization.postprocess.rebuild_molmo_syn import (
    _extract_overall_description,
    _partition_text_components,
)


def test_extract_overall_description_uses_dedicated_trailing_slot():
    texts = {
        1: np.array([101], dtype=np.int32),
        3: np.array([102], dtype=np.int32),
        4: np.array([201], dtype=np.int32),
    }

    desc = _extract_overall_description(texts, sorted_img_indices=[0, 2])

    assert desc.tolist() == [201]


def test_extract_overall_description_returns_none_when_missing():
    texts = {
        1: np.array([101], dtype=np.int32),
        3: np.array([102], dtype=np.int32),
    }

    desc = _extract_overall_description(texts, sorted_img_indices=[0, 2])

    assert desc is None


def test_partition_text_components_handles_sparse_code_slots():
    texts = {
        1: np.array([101], dtype=np.int32),
        4: np.array([103], dtype=np.int32),
        5: np.array([201], dtype=np.int32),
    }

    code_by_image, desc = _partition_text_components(texts, sorted_img_indices=[0, 2, 3])

    assert {ci: toks.tolist() for ci, toks in code_by_image.items()} == {
        0: [101],
        3: [103],
    }
    assert desc.tolist() == [201]


def test_partition_text_components_treats_single_trailing_text_as_ambiguous():
    texts = {
        1: np.array([101], dtype=np.int32),
        3: np.array([201], dtype=np.int32),
    }

    code_by_image, desc = _partition_text_components(texts, sorted_img_indices=[0, 2])

    assert {ci: toks.tolist() for ci, toks in code_by_image.items()} == {
        0: [101],
    }
    assert desc is None
