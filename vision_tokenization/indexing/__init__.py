"""Pre-scan + random-access indexing for vision tokenization datasets."""

from ._scan_wds_worker import DEFAULT_IMAGE_EXTENSIONS, DEFAULT_TEXT_EXTENSIONS
from .clustered_batch_planner import BatchAssignment, BatchPlan, plan_clustered_batches
from .manifest import (
    HF_SCHEMA_MULTI_IMAGE,
    HF_SCHEMA_PHYSICAL,
    HF_SCHEMA_PHYSICAL_MULTI_IMAGE,
    INTERLEAVE_JSONL_TAR_SCHEMA,
    WDS_SCHEMA,
    WDS_SCHEMA_MULTI_IMAGE,
    WDS_SCHEMA_MULTI_IMAGE_WITH_TEXT,
    WDS_SCHEMA_WITH_TEXT,
    load_group_arrays,
    load_hf_manifest,
    load_interleave_manifest,
    load_resolution_arrays,
    load_wds_manifest,
    save_hf_manifest,
    save_interleave_manifest,
    save_wds_manifest,
)
from .reader import TarRandomAccessReader
from .scanner_wds import scan_wds_dataset


def scan_hf_dataset(*args, **kwargs):
    """Lazy wrapper — imports polars only when called."""
    from .scanner_hf import scan_hf_dataset as _scan
    return _scan(*args, **kwargs)


def scan_jsonl_tar_interleave_dataset(*args, **kwargs):
    """Lazy wrapper — imports orjson only when called."""
    from .scanner_interleave import scan_jsonl_tar_interleave_dataset as _scan
    return _scan(*args, **kwargs)

__all__ = [
    # Manifest I/O
    "WDS_SCHEMA",
    "WDS_SCHEMA_WITH_TEXT",
    "WDS_SCHEMA_MULTI_IMAGE",
    "WDS_SCHEMA_MULTI_IMAGE_WITH_TEXT",
    "HF_SCHEMA_MULTI_IMAGE",
    "HF_SCHEMA_PHYSICAL",
    "HF_SCHEMA_PHYSICAL_MULTI_IMAGE",
    "INTERLEAVE_JSONL_TAR_SCHEMA",
    "save_wds_manifest",
    "load_wds_manifest",
    "save_hf_manifest",
    "load_hf_manifest",
    "save_interleave_manifest",
    "load_interleave_manifest",
    "load_resolution_arrays",
    "load_group_arrays",
    # WDS scanning
    "DEFAULT_IMAGE_EXTENSIONS",
    "DEFAULT_TEXT_EXTENSIONS",
    "scan_wds_dataset",
    "scan_jsonl_tar_interleave_dataset",
    # HF scanning
    "scan_hf_dataset",
    # Random-access reader
    "TarRandomAccessReader",
    # Batch planning
    "BatchAssignment",
    "BatchPlan",
    "plan_clustered_batches",
]
