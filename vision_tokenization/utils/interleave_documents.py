"""Helpers for parsing interleaved text-image documents.

Delegates to ``vision_tokenization.parsers`` for the actual parsing.
This module is kept for backward compatibility — existing code that imports
``parse_interleave_segments``, ``parse_markdown_interleave``, etc. from here
will continue to work.
"""

from __future__ import annotations

from typing import Any, List, Sequence

# Re-export from parsers for backward compat
from vision_tokenization.parsers.common import (  # noqa: F401
    is_local_ref as is_local_interleave_ref,
    pre_segment_text,
)
from vision_tokenization.parsers.pin200m import parse as parse_markdown_interleave  # noqa: F401
from vision_tokenization.parsers.shizhen import parse as parse_content_array_interleave  # noqa: F401
from vision_tokenization.parsers.medpix import parse as parse_medpix_interleave  # noqa: F401


def extract_local_image_refs(segments: Sequence[dict[str, Any]]) -> List[str]:
    """Return local image refs in segment order."""
    return [seg["ref"] for seg in segments if seg.get("type") == "image"]


def parse_interleave_segments(
    payload: Any,
    *,
    document_format: str,
    document_field: str | None = None,
    local_prefixes: Sequence[str] | None = None,
    num_images: int = 0,
) -> list[dict[str, Any]]:
    """Parse a raw sample into ordered interleave segments.

    Backward-compatible dispatcher that maps ``document_format`` to the
    new ``parsers`` package.
    """
    from vision_tokenization.parsers import parse_segments

    # Map old document_format names to parser names
    _FORMAT_TO_PARSER = {
        "pin_markdown": "pin200m",
        "content_array": "shizhen",
        "medpix": "medpix",
    }
    parser_name = _FORMAT_TO_PARSER.get(document_format, document_format)

    # For dict payloads with document_field, extract the field first
    if isinstance(payload, dict) and document_field:
        payload = payload.get(document_field)

    return parse_segments(
        payload,
        parser=parser_name,
        num_images=num_images,
        local_prefixes=local_prefixes,
    )
