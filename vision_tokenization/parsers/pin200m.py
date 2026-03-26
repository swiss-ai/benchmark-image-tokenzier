"""PIN-200M: markdown/HTML with inline ``<img>`` / ``![]()`` image refs."""

from __future__ import annotations

import re
from typing import Any, Sequence

from .common import append_text_segment, is_local_ref, normalize_prefixes, pre_segment_text

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


def parse(
    text: str | None,
    *,
    local_prefixes: Sequence[str] | None = None,
    **kwargs,
) -> list[dict[str, Any]]:
    """Parse markdown/HTML into ordered interleave segments."""
    raw = text or ""
    prefixes = normalize_prefixes(local_prefixes)
    segments: list[dict[str, Any]] = []
    last_end = 0

    for match in _IMAGE_PATTERN.finditer(raw):
        start, end = match.span()
        append_text_segment(segments, raw[last_end:start])
        ref = match.group("html_src") or match.group("markdown_src") or ""
        if is_local_ref(ref, local_prefixes=prefixes):
            segments.append({"type": "image", "ref": ref})
        last_end = end

    append_text_segment(segments, raw[last_end:])
    return pre_segment_text(segments)
