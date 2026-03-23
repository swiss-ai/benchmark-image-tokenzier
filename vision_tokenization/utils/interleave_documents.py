"""Helpers for parsing interleaved text-image documents.

Supported raw formats:
- PIN-style markdown with inline image syntax
- Structured ``content[]`` arrays with ``{"type": "text"|"image"}``
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Sequence

_DEFAULT_LOCAL_PREFIXES = ("content_image/", "image/")

_IMAGE_PATTERN = re.compile(
    r"""
    (?P<html>
        <img\b[^>]*?\bsrc\s*=\s*
        (?P<html_quote>["'])?
        (?P<html_src>[^"'>\s]+)
        (?P=html_quote)?
        [^>]*>
    )
    |
    (?P<markdown>
        !\[
            (?P<markdown_alt>[^\]]*)
        \]
        \(
            \s*
            <?(?P<markdown_src>[^>\s)]+)>?
            (?:\s+["'][^"']*["'])?
            \s*
        \)
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def _normalize_prefixes(local_prefixes: Sequence[str] | None) -> tuple[str, ...]:
    prefixes = tuple(local_prefixes or _DEFAULT_LOCAL_PREFIXES)
    if not prefixes:
        raise ValueError("local_prefixes must not be empty")
    return prefixes


def _is_remote_ref(ref: str) -> bool:
    lowered = ref.lower()
    return lowered.startswith(("http://", "https://", "//"))


def is_local_interleave_ref(
    ref: str,
    *,
    local_prefixes: Sequence[str] | None = None,
) -> bool:
    """Return ``True`` when *ref* points to a supported local image asset."""
    if not ref or _is_remote_ref(ref):
        return False
    prefixes = _normalize_prefixes(local_prefixes)
    return ref.startswith(prefixes)


def _append_text_segment(segments: list[dict[str, Any]], text: str) -> None:
    if not text:
        return
    if segments and segments[-1]["type"] == "text":
        segments[-1]["text"] += text
    else:
        segments.append({"type": "text", "text": text})


def parse_markdown_interleave(
    markdown: str | None,
    *,
    local_prefixes: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Parse markdown/HTML and return ordered interleave segments.

    Local image refs become ``{"type": "image", "ref": ...}`` segments.
    Remote image syntax is removed from the text stream.
    """
    text = markdown or ""
    prefixes = _normalize_prefixes(local_prefixes)
    segments: list[dict[str, Any]] = []
    last_end = 0

    for match in _IMAGE_PATTERN.finditer(text):
        start, end = match.span()
        _append_text_segment(segments, text[last_end:start])

        ref = match.group("html_src") or match.group("markdown_src") or ""
        if is_local_interleave_ref(ref, local_prefixes=prefixes):
            segments.append({"type": "image", "ref": ref})
        last_end = end

    _append_text_segment(segments, text[last_end:])
    return segments


def parse_content_array_interleave(
    content: Iterable[dict[str, Any]] | None,
    *,
    local_prefixes: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert structured ``content[]`` arrays into ordered segments."""
    prefixes = _normalize_prefixes(local_prefixes)
    segments: list[dict[str, Any]] = []

    for block in content or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            _append_text_segment(segments, str(block.get("text") or ""))
            continue
        if block_type == "image":
            ref = str(block.get("image") or "")
            if is_local_interleave_ref(ref, local_prefixes=prefixes):
                segments.append({"type": "image", "ref": ref})

    return segments


def parse_interleave_segments(
    payload: Any,
    *,
    document_format: str,
    document_field: str | None = None,
    local_prefixes: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Parse a raw sample into ordered interleave segments."""
    prefixes = _normalize_prefixes(local_prefixes)

    if document_format == "pin_markdown":
        if not isinstance(payload, dict):
            raise TypeError("pin_markdown payload must be a dict-like sample")
        field = document_field or "md"
        return parse_markdown_interleave(payload.get(field), local_prefixes=prefixes)

    if document_format == "content_array":
        if not isinstance(payload, dict):
            raise TypeError("content_array payload must be a dict-like sample")
        field = document_field or "content"
        return parse_content_array_interleave(payload.get(field), local_prefixes=prefixes)

    raise ValueError(f"Unsupported document_format: {document_format!r}")


def extract_local_image_refs(segments: Sequence[dict[str, Any]]) -> List[str]:
    """Return local image refs in segment order."""
    return [seg["ref"] for seg in segments if seg.get("type") == "image"]
