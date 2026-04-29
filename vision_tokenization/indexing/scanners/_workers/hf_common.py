"""Shared HF shard-scan helpers."""

from array import array
from io import BytesIO
from typing import Tuple

import imagesize
import pyarrow as pa
import pyarrow.compute as pc

_HEADER_BYTES = 4096
_HF_WORKER_SCHEMA = pa.schema(
    [
        pa.field("sample_index", pa.int64()),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
        pa.field("chunk_index", pa.int32()),
        pa.field("row_in_chunk", pa.int32()),
    ]
)
_HF_WORKER_SCHEMA_MULTI_IMAGE = pa.schema(
    [
        pa.field("sample_index", pa.int64()),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
        pa.field("group_id", pa.int64()),
        pa.field("image_index", pa.int16()),
        pa.field("chunk_index", pa.int32()),
        pa.field("row_in_chunk", pa.int32()),
    ]
)


def build_hf_output_columns(is_multi: bool) -> dict[str, array]:
    """Create empty manifest-output columns for one shard scan."""
    if is_multi:
        return {
            "sample_index": array("q"),
            "width": array("i"),
            "height": array("i"),
            "group_id": array("q"),
            "image_index": array("h"),
            "chunk_index": array("i"),
            "row_in_chunk": array("i"),
        }
    return {
        "sample_index": array("q"),
        "width": array("i"),
        "height": array("i"),
        "chunk_index": array("i"),
        "row_in_chunk": array("i"),
    }


def build_hf_output_table(columns: dict[str, array], is_multi: bool) -> pa.Table:
    """Convert compact typed column buffers into an Arrow table."""
    schema = _HF_WORKER_SCHEMA_MULTI_IMAGE if is_multi else _HF_WORKER_SCHEMA
    arrays = {
        field.name: pa.array(columns[field.name], type=field.type)
        for field in schema
    }
    return pa.table(arrays, schema=schema)


def get_image_dimensions(img_data) -> Tuple[int, int]:
    """Get dimensions from a raw HF image cell without full decode."""
    try:
        if isinstance(img_data, dict):
            img_bytes = img_data.get("bytes")
            if img_bytes is not None:
                header = img_bytes[:_HEADER_BYTES]
                w, h = imagesize.get(BytesIO(header))
                if w < 0 or h < 0:
                    w, h = imagesize.get(BytesIO(img_bytes))
                return w, h
            img_path = img_data.get("path")
            if img_path is not None:
                return imagesize.get(img_path)
        if isinstance(img_data, bytes):
            header = img_data[:_HEADER_BYTES]
            w, h = imagesize.get(BytesIO(header))
            if w < 0 or h < 0:
                w, h = imagesize.get(BytesIO(img_data))
            return w, h
        if hasattr(img_data, "size"):
            return img_data.size
    except Exception:
        pass
    return -1, -1


def _iter_column_chunks(image_col):
    if isinstance(image_col, pa.ChunkedArray):
        row_base = 0
        for chunk in image_col.chunks:
            yield row_base, chunk
            row_base += len(chunk)
        return
    yield 0, image_col


def _get_dims_from_scalars(header_scalar, bytes_scalar, path_scalar) -> Tuple[int, int]:
    """Read dimensions from a 4KB header, with full-bytes fallback only if needed."""
    try:
        header = header_scalar.as_py() if header_scalar is not None else None
        if header:
            w, h = imagesize.get(BytesIO(header))
            if w >= 0 and h >= 0:
                return w, h

        img_bytes = bytes_scalar.as_py() if bytes_scalar is not None else None
        if img_bytes is not None:
            return imagesize.get(BytesIO(img_bytes))

        img_path = path_scalar.as_py() if path_scalar is not None else None
        if img_path is not None:
            return imagesize.get(img_path)
    except Exception:
        pass
    return -1, -1


def _scan_single_image_chunk(
    out: dict[str, array],
    chunk,
    *,
    chunk_index: int,
    row_base: int,
    source_rows: int,
    failed_dims: int,
) -> Tuple[int, int]:
    if pa.types.is_struct(chunk.type):
        bytes_arr = chunk.field("bytes")
        path_arr = chunk.field("path")
        header_arr = pc.binary_slice(bytes_arr, 0, _HEADER_BYTES)

        for local_idx in range(len(chunk)):
            width, height = _get_dims_from_scalars(
                header_arr[local_idx],
                bytes_arr[local_idx],
                path_arr[local_idx],
            )
            if width < 0 or height < 0:
                failed_dims += 1
            out["sample_index"].append(source_rows)
            out["width"].append(width)
            out["height"].append(height)
            out["chunk_index"].append(chunk_index)
            out["row_in_chunk"].append(row_base + local_idx)
            source_rows += 1
        return source_rows, failed_dims

    for local_idx in range(len(chunk)):
        width, height = get_image_dimensions(chunk[local_idx].as_py())
        if width < 0 or height < 0:
            failed_dims += 1
        out["sample_index"].append(source_rows)
        out["width"].append(width)
        out["height"].append(height)
        out["chunk_index"].append(chunk_index)
        out["row_in_chunk"].append(row_base + local_idx)
        source_rows += 1
    return source_rows, failed_dims


def _scan_multi_image_chunk(
    out: dict[str, array],
    chunk,
    *,
    chunk_index: int,
    row_base: int,
    source_rows: int,
    failed_dims: int,
) -> Tuple[int, int]:
    if pa.types.is_list(chunk.type) or pa.types.is_large_list(chunk.type):
        offsets = chunk.offsets.to_numpy(zero_copy_only=False)
        offset_base = int(offsets[0])
        values = chunk.flatten()

        if pa.types.is_struct(values.type):
            bytes_arr = values.field("bytes")
            path_arr = values.field("path")
            header_arr = pc.binary_slice(bytes_arr, 0, _HEADER_BYTES)

            for local_row_idx in range(len(chunk)):
                local_sample_index = source_rows
                source_rows += 1
                start = int(offsets[local_row_idx]) - offset_base
                end = int(offsets[local_row_idx + 1]) - offset_base
                for flat_idx in range(start, end):
                    width, height = _get_dims_from_scalars(
                        header_arr[flat_idx],
                        bytes_arr[flat_idx],
                        path_arr[flat_idx],
                    )
                    if width < 0 or height < 0:
                        failed_dims += 1
                    out["sample_index"].append(local_sample_index)
                    out["width"].append(width)
                    out["height"].append(height)
                    out["group_id"].append(local_sample_index)
                    out["image_index"].append(flat_idx - start)
                    out["chunk_index"].append(chunk_index)
                    out["row_in_chunk"].append(row_base + local_row_idx)
            return source_rows, failed_dims

    for local_row_idx in range(len(chunk)):
        local_sample_index = source_rows
        source_rows += 1
        image_list = chunk[local_row_idx].as_py()
        for image_index, img_data in enumerate(image_list or []):
            width, height = get_image_dimensions(img_data)
            if width < 0 or height < 0:
                failed_dims += 1
            out["sample_index"].append(local_sample_index)
            out["width"].append(width)
            out["height"].append(height)
            out["group_id"].append(local_sample_index)
            out["image_index"].append(image_index)
            out["chunk_index"].append(chunk_index)
            out["row_in_chunk"].append(row_base + local_row_idx)
    return source_rows, failed_dims


def scan_hf_batch_columns(
    out: dict[str, array],
    image_col,
    chunk_index: int,
    source_rows: int,
    failed_dims: int,
    *,
    is_multi: bool,
) -> Tuple[dict[str, array], int, int]:
    """Append one Arrow/Parquet batch worth of manifest rows."""
    for row_base, chunk in _iter_column_chunks(image_col):
        if is_multi:
            source_rows, failed_dims = _scan_multi_image_chunk(
                out,
                chunk,
                chunk_index=chunk_index,
                row_base=row_base,
                source_rows=source_rows,
                failed_dims=failed_dims,
            )
        else:
            source_rows, failed_dims = _scan_single_image_chunk(
                out,
                chunk,
                chunk_index=chunk_index,
                row_base=row_base,
                source_rows=source_rows,
                failed_dims=failed_dims,
            )

    return out, source_rows, failed_dims
