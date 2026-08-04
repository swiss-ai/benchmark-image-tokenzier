"""Multilingual recap: single image with captions in multiple languages.

Text format (from .txt sidecar):
    lang=en
    English caption paragraph...
    lang=de
    German caption paragraph...
    lang=ja
    Japanese caption paragraph...

Returns one image + N text segments, one per language.
Each text segment is tagged with its language code.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

_LANG_PATTERN = re.compile(r"^lang=([A-Za-z0-9_-]+)\s*$", re.MULTILINE)


def parse(
    text: str | None,
    *,
    languages: Sequence[str] | None = None,
    **kwargs,
) -> list[dict[str, Any]]:
    """Parse multilingual recap text into interleave segments.

    Args:
        text: Raw text with ``lang=xx`` headers separating languages.
        languages: If set, only include these language codes.
            None means include all languages.

    Returns:
        Segments: [{"type": "image"}, {"type": "text", "text": "...", "lang": "en"}, ...]
    """
    raw = text or ""
    segments: list[dict[str, Any]] = []

    # Split by lang=xx headers
    splits = _LANG_PATTERN.split(raw)
    # splits = [preamble, lang1, text1, lang2, text2, ...]

    # First element is preamble (before any lang= tag), skip if empty
    lang_texts = []
    for i in range(1, len(splits) - 1, 2):
        lang_code = splits[i].strip()
        lang_text = splits[i + 1].strip()
        if lang_text and (languages is None or lang_code in languages):
            lang_texts.append((lang_code, lang_text))

    if not lang_texts:
        return []

    # Image first, then text segments per language
    segments.append({"type": "image"})
    for lang_code, lang_text in lang_texts:
        segments.append({
            "type": "text",
            "text": lang_text,
            "lang": lang_code,
        })

    return segments
