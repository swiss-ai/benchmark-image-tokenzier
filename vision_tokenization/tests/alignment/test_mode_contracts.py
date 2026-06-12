"""Posttraining mode contracts: entry validation + the publish completeness gate."""

from types import SimpleNamespace

import pytest

from vision_tokenization.pipeline import run_distributed_pipeline
from vision_tokenization.pipeline.runtime.posttraining import (
    EncodeIncompleteError,
    check_encode_complete,
)


def test_posttraining_rejects_multi_image():
    with pytest.raises(ValueError, match="multi_image is meaningless"):
        run_distributed_pipeline({"mode": "posttraining", "multi_image": True})


def test_posttraining_rejects_dry_run():
    with pytest.raises(ValueError, match="dry_run is unsupported"):
        run_distributed_pipeline({"mode": "posttraining", "dry_run": True})


def _media(*ids):
    return [SimpleNamespace(media_id=i) for i in ids]


def test_encode_complete_passes_when_all_media_stored():
    check_encode_complete(_media("a", "b"), {"a": 10, "b": 7})


def test_encode_incomplete_raises_named_contract_error():
    with pytest.raises(EncodeIncompleteError, match=r"1 of 2 scanned media"):
        check_encode_complete(_media("a", "b"), {"a": 10})
