"""Parquet manifest schema and I/O for WDS and HF dataset indexing."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WDS manifest schema
# ---------------------------------------------------------------------------
WDS_SCHEMA = pa.schema(
    [
        pa.field("sample_key", pa.string()),
        pa.field("tar_path", pa.dictionary(pa.int32(), pa.string())),
        pa.field("offset_data", pa.int64()),
        pa.field("file_size", pa.int32()),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
        pa.field("image_ext", pa.dictionary(pa.int32(), pa.string())),
    ]
)

# Extended schema with text sidecar columns (offset_text=-1 means no sidecar)
WDS_SCHEMA_WITH_TEXT = pa.schema(
    list(WDS_SCHEMA)
    + [
        pa.field("offset_text", pa.int64()),
        pa.field("text_file_size", pa.int32()),
        pa.field("text_ext", pa.dictionary(pa.int32(), pa.string())),
    ]
)

# Multi-image columns: group_id groups images from the same sample,
# image_index orders images within the group.
_MULTI_IMAGE_FIELDS = [
    pa.field("group_id", pa.int64()),
    pa.field("image_index", pa.int16()),
]

WDS_SCHEMA_MULTI_IMAGE = pa.schema(list(WDS_SCHEMA) + _MULTI_IMAGE_FIELDS)

WDS_SCHEMA_MULTI_IMAGE_WITH_TEXT = pa.schema(
    list(WDS_SCHEMA_WITH_TEXT) + _MULTI_IMAGE_FIELDS
)

HF_SCHEMA_MULTI_IMAGE = pa.schema(
    [
        pa.field("sample_index", pa.int64()),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
        pa.field("group_id", pa.int64()),
        pa.field("image_index", pa.int16()),
    ]
)

# ---------------------------------------------------------------------------
# HF manifest schema
# ---------------------------------------------------------------------------
HF_SCHEMA = pa.schema(
    [
        pa.field("sample_index", pa.int64()),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
    ]
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CHUNK_SIZE = 1_000_000  # rows per Parquet row-group


def _records_to_table(records: Union[List[Dict], pa.Table], schema: pa.Schema) -> pa.Table:
    """Convert a list of dicts (or pass-through a Table) to a pyarrow Table."""
    if isinstance(records, pa.Table):
        # Select and cast columns to match the target schema (the input
        # table may have different column order or wider integer types).
        return records.select([f.name for f in schema]).cast(schema)
    # Build column arrays individually so we can apply dictionary encoding
    arrays = {}
    for field in schema:
        values = [r[field.name] for r in records]
        if pa.types.is_dictionary(field.type):
            plain = pa.array(values, type=field.type.value_type)
            arrays[field.name] = plain.dictionary_encode()
        else:
            arrays[field.name] = pa.array(values, type=field.type)
    return pa.table(arrays, schema=schema)


# ---------------------------------------------------------------------------
# WDS manifest I/O
# ---------------------------------------------------------------------------
def save_wds_manifest(
    records: Union[List[Dict], pa.Table],
    output_path: Union[str, Path],
    chunk_size: int = _CHUNK_SIZE,
    include_text: bool = False,
    schema: Optional[pa.Schema] = None,
) -> str:
    """Write WDS manifest records to a zstd-compressed Parquet file.

    Args:
        records: List of dicts with keys matching the chosen schema, or a pa.Table.
        output_path: Destination Parquet file path.
        chunk_size: Number of rows per row-group (default 1M).
        include_text: If ``True``, use the extended schema with text sidecar
            columns (``offset_text``, ``text_file_size``, ``text_ext``).
            Ignored when *schema* is provided explicitly.
        schema: Explicit schema override (e.g. multi-image schemas).

    Returns:
        The output path as a string.
    """
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if schema is None:
        schema = WDS_SCHEMA_WITH_TEXT if include_text else WDS_SCHEMA
    table = _records_to_table(records, schema)
    n_rows = len(table)

    writer = pq.ParquetWriter(output_path, schema, compression="zstd")
    try:
        for start in range(0, n_rows, chunk_size):
            writer.write_table(table.slice(start, chunk_size))
    finally:
        writer.close()

    logger.info(f"Saved WDS manifest: {n_rows:,} rows -> {output_path}")
    return output_path


def load_wds_manifest(
    path: Union[str, Path],
    columns: Optional[Sequence[str]] = None,
) -> pa.Table:
    """Read a WDS manifest Parquet file with optional column pruning."""
    return pq.read_table(str(path), columns=columns)


# ---------------------------------------------------------------------------
# HF manifest I/O
# ---------------------------------------------------------------------------
def save_hf_manifest(
    records: Union[List[Dict], pa.Table],
    output_path: Union[str, Path],
    chunk_size: int = _CHUNK_SIZE,
    schema: Optional[pa.Schema] = None,
) -> str:
    """Write HF manifest records to a zstd-compressed Parquet file."""
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if schema is None:
        schema = HF_SCHEMA
    table = _records_to_table(records, schema)
    n_rows = len(table)

    writer = pq.ParquetWriter(output_path, schema, compression="zstd")
    try:
        for start in range(0, n_rows, chunk_size):
            writer.write_table(table.slice(start, chunk_size))
    finally:
        writer.close()

    logger.info(f"Saved HF manifest: {n_rows:,} rows -> {output_path}")
    return output_path


def load_hf_manifest(
    path: Union[str, Path],
    columns: Optional[Sequence[str]] = None,
) -> pa.Table:
    """Read an HF manifest Parquet file with optional column pruning."""
    return pq.read_table(str(path), columns=columns)


# ---------------------------------------------------------------------------
# Convenience: load only resolution arrays
# ---------------------------------------------------------------------------
def load_resolution_arrays(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Read only width/height columns as numpy int32 arrays.

    Works for both WDS and HF manifests (both have width, height columns).
    """
    table = pq.read_table(str(path), columns=["width", "height"])
    widths = table.column("width").to_numpy().astype(np.int32)
    heights = table.column("height").to_numpy().astype(np.int32)
    return widths, heights


def load_group_arrays(
    path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read width, height, group_id, and image_index columns.

    If ``group_id`` is absent (single-image manifest), synthesises
    ``group_id = arange(N)`` and ``image_index = zeros(N)`` for backward
    compatibility.

    Returns:
        ``(widths, heights, group_ids, image_indices)`` as numpy arrays.
    """
    # Probe schema for group_id column
    schema = pq.read_schema(str(path))
    has_groups = "group_id" in schema.names

    if has_groups:
        columns = ["width", "height", "group_id", "image_index"]
    else:
        columns = ["width", "height"]

    table = pq.read_table(str(path), columns=columns)
    widths = table.column("width").to_numpy().astype(np.int32)
    heights = table.column("height").to_numpy().astype(np.int32)

    if has_groups:
        group_ids = table.column("group_id").to_numpy().astype(np.int64)
        image_indices = table.column("image_index").to_numpy().astype(np.int16)
    else:
        n = len(widths)
        group_ids = np.arange(n, dtype=np.int64)
        image_indices = np.zeros(n, dtype=np.int16)

    return widths, heights, group_ids, image_indices
