"""Shared media identity helpers for scanner and ingest paths."""

from __future__ import annotations

import hashlib
from typing import BinaryIO

_HASH_CHUNK_BYTES = 8 << 20


def sha256_buffer(data) -> bytes:
    """Return SHA-256 over an existing bytes-like object as 32 raw bytes."""
    return hashlib.sha256(data).digest()


def sha256_arrow_scalar(scalar) -> bytes | None:
    """Return SHA-256 for a PyArrow binary scalar without converting to hex."""
    if scalar is None or not scalar.is_valid:
        return None
    return sha256_buffer(memoryview(scalar.as_buffer()))


def sha256_fileobj(fileobj: BinaryIO, size: int, chunk_size: int = _HASH_CHUNK_BYTES) -> bytes:
    """Hash exactly *size* bytes from the start of a seekable file-like object."""
    h = hashlib.sha256()
    buf = bytearray(chunk_size)
    view = memoryview(buf)
    remaining = int(size)
    fileobj.seek(0)
    while remaining:
        n = fileobj.readinto(view[: min(chunk_size, remaining)])
        if not n:
            raise EOFError("short read while hashing media bytes")
        h.update(view[:n])
        remaining -= n
    return h.digest()
