import pytest
from omegaconf import OmegaConf

from vision_tokenization.tokenize import (
    _merge_pipeline_sections,
    _resolve_output_format,
    _validate_pipeline_cfg,
)


def test_interleave_forces_pooled_even_if_direct_requested():
    dataset_cfg = OmegaConf.create(
        {
            "dataset_type": "jsonl_tar_interleave",
            "output_format": "direct",
            "multi_image": True,
        }
    )

    assert _resolve_output_format("interleave", dataset_cfg) == "pooled"


def test_multi_image_forces_pooled_for_non_interleave_modes():
    dataset_cfg = OmegaConf.create(
        {
            "dataset_type": "wds",
            "output_format": "direct",
            "multi_image": True,
        }
    )

    assert _resolve_output_format("image2text", dataset_cfg) == "pooled"


def test_image_list_column_infers_multi_image_and_forces_pooled():
    dataset_cfg = OmegaConf.create(
        {
            "dataset_type": "hf",
            "output_format": "direct",
            "multi_image": None,
            "image_list_column": "images",
        }
    )

    assert _resolve_output_format("image2text", dataset_cfg) == "pooled"


def test_single_image_jobs_keep_direct_by_default():
    dataset_cfg = OmegaConf.create(
        {
            "dataset_type": "hf",
            "multi_image": False,
            "output_format": "direct",
        }
    )

    assert _resolve_output_format("image2text", dataset_cfg) == "direct"


def test_merge_pipeline_sections_prefers_dataset_flat_over_root_section_defaults():
    merged = _merge_pipeline_sections(
        {
            "resume": False,
            "checkpoint_interval_batches": 500,
            "direct": {
                "checkpoint_interval_batches": 5_000,
                "merge_shards": True,
            },
        },
        {
            "dataset_type": "hf",
            "checkpoint_interval_batches": 2_000,
            "batch_plan": "/tmp/dataset-flat.pt",
        },
        output_format="direct",
    )

    assert merged["checkpoint_interval_batches"] == 2_000
    assert merged["batch_plan"] == "/tmp/dataset-flat.pt"
    assert merged["merge_shards"] is True


def test_merge_pipeline_sections_prefers_active_dataset_section_over_flat_aliases():
    merged = _merge_pipeline_sections(
        {
            "pooled": {
                "document_window_docs": 1_000,
                "checkpoint_every_windows": 2,
                "rebuild": True,
            },
            "direct": {
                "checkpoint_interval_batches": 5_000,
            },
        },
        {
            "document_plan_path": "/tmp/dataset-flat.pt",
            "pooled": {
                "document_window_docs": 100,
                "document_plan_path": "/tmp/dataset-section.pt",
            },
        },
        output_format="pooled",
    )

    assert merged["document_window_docs"] == 100
    assert merged["document_plan_path"] == "/tmp/dataset-section.pt"
    assert merged["checkpoint_every_windows"] == 2
    assert merged["rebuild"] is True
    assert "checkpoint_interval_batches" not in merged


def test_merge_pipeline_sections_does_not_map_legacy_pooled_aliases():
    merged = _merge_pipeline_sections(
        {},
        {
            "checkpoint_interval_batches": 123,
            "batch_plan": "/tmp/legacy-pooled.pt",
        },
        output_format="pooled",
    )

    assert "document_window_docs" not in merged
    assert "checkpoint_every_windows" not in merged
    assert merged["batch_plan"] == "/tmp/legacy-pooled.pt"


def test_validate_pipeline_cfg_rejects_legacy_pooled_keys():
    with pytest.raises(ValueError, match="legacy chunk-based keys"):
        _validate_pipeline_cfg(
            {
                "chunk_docs": 100,
                "document_window_docs": 1_000,
                "checkpoint_every_windows": 1,
                "spill_shard_rollover_windows": 8,
            },
            output_format="pooled",
        )


def test_validate_pipeline_cfg_rejects_batch_plan_for_pooled():
    with pytest.raises(ValueError, match="document_plan_path"):
        _validate_pipeline_cfg(
            {
                "batch_plan": "/tmp/old.pt",
                "document_window_docs": 1_000,
                "checkpoint_every_windows": 1,
                "spill_shard_rollover_windows": 8,
            },
            output_format="pooled",
        )
