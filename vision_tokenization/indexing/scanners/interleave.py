"""Backward-compatible interleave scanner built on the generic JSONL+tar scanner."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .jsonl_tar import _scan_jsonl_tar_dataset_impl


def scan_jsonl_tar_interleave_dataset(
    input_pattern: str,
    output_manifest: str | Path,
    *,
    document_format: str,
    document_field: str | None = None,
    local_image_prefixes: Sequence[str] | None = None,
    tar_pattern: str,
    tar_scope: str = "parent_dir",
    tar_root: str | Path | None = None,
    num_workers: int = 64,
) -> str:
    """Scan an interleave JSONL+tar dataset into the grouped manifest schema."""
    return _scan_jsonl_tar_dataset_impl(
        input_pattern,
        output_manifest,
        image_field=None,
        document_format=document_format,
        document_field=document_field,
        local_image_prefixes=local_image_prefixes,
        tar_pattern=tar_pattern,
        tar_scope=tar_scope,
        tar_root=tar_root,
        image_path_prefix_strip=None,
        num_workers=num_workers,
        metadata_dataset_type="jsonl_tar",
        metadata_extra={
            "document_format": document_format,
            "document_field": document_field,
        },
    )
