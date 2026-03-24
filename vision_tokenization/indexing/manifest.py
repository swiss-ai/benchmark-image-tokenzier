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
        pa.field("file_size", pa.int64()),
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
        pa.field("text_file_size", pa.int64()),
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

# Physical-location columns for HF manifests. ``chunk_index`` means:
#   - parquet row-group index
#   - arrow record-batch index
# ``row_in_chunk`` is the row position within that chunk.
_HF_LOCATION_FIELDS = [
    pa.field("shard_path", pa.dictionary(pa.int32(), pa.string())),
    pa.field("chunk_index", pa.int32()),
    pa.field("row_in_chunk", pa.int32()),
]

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

HF_SCHEMA_PHYSICAL = pa.schema(list(HF_SCHEMA) + _HF_LOCATION_FIELDS)

HF_SCHEMA_PHYSICAL_MULTI_IMAGE = pa.schema(
    list(HF_SCHEMA_MULTI_IMAGE) + _HF_LOCATION_FIELDS
)

# ---------------------------------------------------------------------------
# JSONL + tar interleave manifest schema
# ---------------------------------------------------------------------------
INTERLEAVE_JSONL_TAR_SCHEMA = pa.schema(
    [
        pa.field("tar_path", pa.dictionary(pa.int32(), pa.string())),
        pa.field("offset_data", pa.int64()),
        pa.field("file_size", pa.int64()),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
        pa.field("group_id", pa.int64()),
        pa.field("image_index", pa.int16()),
        pa.field("jsonl_path", pa.dictionary(pa.int32(), pa.string())),
        pa.field("line_start", pa.int64()),
        pa.field("line_length", pa.int32()),
        pa.field("image_ref", pa.dictionary(pa.int32(), pa.string())),
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


def records_to_table(records: Union[List[Dict], pa.Table], schema: pa.Schema) -> pa.Table:
    """Public wrapper for converting manifest records into a typed Arrow table."""
    return _records_to_table(records, schema)


def _write_table_chunks(
    writer: pq.ParquetWriter,
    table: pa.Table,
    chunk_size: int = _CHUNK_SIZE,
) -> None:
    """Write a table to Parquet in row-group sized chunks."""
    n_rows = len(table)
    for start in range(0, n_rows, chunk_size):
        writer.write_table(table.slice(start, chunk_size))


def append_parquet_records(
    writer: pq.ParquetWriter,
    records: Union[List[Dict], pa.Table],
    schema: pa.Schema,
    chunk_size: int = _CHUNK_SIZE,
) -> int:
    """Append records to an open Parquet writer and return written rows."""
    table = _records_to_table(records, schema)
    _write_table_chunks(writer, table, chunk_size=chunk_size)
    return len(table)


# ---------------------------------------------------------------------------
# Manifest I/O (shared implementation)
# ---------------------------------------------------------------------------
def _save_manifest(
    records: Union[List[Dict], pa.Table],
    output_path: Union[str, Path],
    schema: pa.Schema,
    label: str,
    chunk_size: int = _CHUNK_SIZE,
) -> str:
    """Write manifest records to a zstd-compressed Parquet file."""
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    table = _records_to_table(records, schema)
    n_rows = len(table)

    writer = pq.ParquetWriter(output_path, schema, compression="zstd")
    try:
        _write_table_chunks(writer, table, chunk_size=chunk_size)
    finally:
        writer.close()

    logger.info(f"Saved {label} manifest: {n_rows:,} rows -> {output_path}")
    return output_path


def load_manifest(
    path: Union[str, Path],
    columns: Optional[Sequence[str]] = None,
) -> pa.Table:
    """Read a manifest Parquet file with optional column pruning."""
    return pq.read_table(str(path), columns=columns)


def save_wds_manifest(
    records: Union[List[Dict], pa.Table],
    output_path: Union[str, Path],
    chunk_size: int = _CHUNK_SIZE,
    include_text: bool = False,
    schema: Optional[pa.Schema] = None,
) -> str:
    """Write WDS manifest records to a zstd-compressed Parquet file."""
    if schema is None:
        schema = WDS_SCHEMA_WITH_TEXT if include_text else WDS_SCHEMA
    return _save_manifest(records, output_path, schema, "WDS", chunk_size)


def save_hf_manifest(
    records: Union[List[Dict], pa.Table],
    output_path: Union[str, Path],
    chunk_size: int = _CHUNK_SIZE,
    schema: Optional[pa.Schema] = None,
) -> str:
    """Write HF manifest records to a zstd-compressed Parquet file."""
    if schema is None:
        schema = HF_SCHEMA
    return _save_manifest(records, output_path, schema, "HF", chunk_size)


def save_interleave_manifest(
    records: Union[List[Dict], pa.Table],
    output_path: Union[str, Path],
    chunk_size: int = _CHUNK_SIZE,
    schema: Optional[pa.Schema] = None,
) -> str:
    """Write interleave manifest records to a zstd-compressed Parquet file."""
    if schema is None:
        schema = INTERLEAVE_JSONL_TAR_SCHEMA
    return _save_manifest(records, output_path, schema, "interleave", chunk_size)


# Typed aliases for readability; all delegate to the same implementation.
load_wds_manifest = load_manifest
load_hf_manifest = load_manifest
load_interleave_manifest = load_manifest


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
