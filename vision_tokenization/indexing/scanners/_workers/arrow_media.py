"""Shared Arrow media-cell helpers for scanner workers."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from vision_tokenization.indexing.media_identity import (
    sha256_arrow_scalar,
    sha256_buffer,
)
from vision_tokenization.indexing.scanners._workers.hf_common import (
    _binary_header_array,
    _get_dims_from_scalars,
    get_image_dimensions,
)

EMPTY_MEDIA_BYTES = b""
EMPTY_MEDIA_ID = sha256_buffer(EMPTY_MEDIA_BYTES).hex()


@dataclass(frozen=True)
class ArrowImage:
    bytes_scalar: object
    path_scalar: object | None
    header_scalar: object | None


def struct_field_or_none(arr, name: str):
    if not pa.types.is_struct(arr.type):
        return None
    if arr.type.get_field_index(name) < 0:
        return None
    return arr.field(name)


def scalar_to_py(scalar):
    if scalar is None or not scalar.is_valid:
        return None
    return scalar.as_py()


class ImageColumnView:
    """Row-wise image access that keeps encoded bytes in Arrow until needed."""

    def __init__(self, column):
        self._column = column
        self._kind = "python"
        self._bytes = None
        self._paths = None
        self._headers = None
        self._offsets = None

        if pa.types.is_struct(column.type):
            self._kind = "struct"
            self._bytes = struct_field_or_none(column, "bytes")
            self._paths = struct_field_or_none(column, "path")
            self._headers = _binary_header_array(self._bytes)
            return

        if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
            values = column.values
            if pa.types.is_struct(values.type):
                self._kind = "list_struct"
                self._bytes = struct_field_or_none(values, "bytes")
                self._paths = struct_field_or_none(values, "path")
                self._headers = _binary_header_array(self._bytes)
                self._offsets = column.offsets.to_numpy(zero_copy_only=False)

    def row(self, row_idx: int):
        if self._kind == "struct":
            if not self._column[row_idx].is_valid:
                return []
            return ArrowImage(
                self._bytes[row_idx] if self._bytes is not None else None,
                self._paths[row_idx] if self._paths is not None else None,
                self._headers[row_idx] if self._headers is not None else None,
            )

        if self._kind == "list_struct":
            if not self._column[row_idx].is_valid:
                return []
            start = int(self._offsets[row_idx])
            end = int(self._offsets[row_idx + 1])
            return [
                ArrowImage(
                    self._bytes[i] if self._bytes is not None else None,
                    self._paths[i] if self._paths is not None else None,
                    self._headers[i] if self._headers is not None else None,
                )
                for i in range(start, end)
            ]

        return self._column[row_idx].as_py()


def image_bytes(img: dict | ArrowImage) -> bytes:
    if isinstance(img, ArrowImage):
        scalar = img.bytes_scalar
        if scalar is None or not scalar.is_valid:
            return EMPTY_MEDIA_BYTES
        return bytes(memoryview(scalar.as_buffer()))
    return img["bytes"]


def image_raw_length(img: dict | ArrowImage) -> int:
    if isinstance(img, ArrowImage):
        scalar = img.bytes_scalar
        if scalar is None or not scalar.is_valid:
            return 0
        return scalar.as_buffer().size
    raw = img.get("bytes")
    return len(raw) if raw is not None else 0


def image_path(img: dict | ArrowImage) -> str | None:
    if isinstance(img, ArrowImage):
        return scalar_to_py(img.path_scalar)
    return img.get("path")


def image_dimensions(img: dict | ArrowImage) -> tuple[int, int]:
    if isinstance(img, ArrowImage):
        width, height = _get_dims_from_scalars(
            img.header_scalar,
            img.bytes_scalar,
            img.path_scalar,
        )
    else:
        width, height = get_image_dimensions(img)
    return (width, height) if width >= 0 and height >= 0 else (0, 0)


def image_media_id(img: dict | ArrowImage) -> str:
    if isinstance(img, ArrowImage):
        digest = sha256_arrow_scalar(img.bytes_scalar)
        if digest is not None:
            return digest.hex()
        return EMPTY_MEDIA_ID
    return sha256_buffer(img["bytes"]).hex()
