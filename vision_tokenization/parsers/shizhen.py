"""Shizhen (book/web): structured ``content[]`` arrays."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .common import append_text_segment, is_local_ref, normalize_prefixes, pre_segment_text


def parse(
    content: Iterable[dict[str, Any]] | None,
    *,
    local_prefixes: Sequence[str] | None = None,
    **kwargs,
) -> list[dict[str, Any]]:
    """Convert structured content arrays into ordered segments."""
    prefixes = normalize_prefixes(local_prefixes)
    segments: list[dict[str, Any]] = []

    for block in content or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            append_text_segment(segments, str(block.get("text") or ""))
        elif block_type == "image":
            ref = str(block.get("image") or "")
            if is_local_ref(ref, local_prefixes=prefixes):
                segments.append({"type": "image", "ref": ref})

    return pre_segment_text(segments)
