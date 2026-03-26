"""Shared helpers for interleave segment parsers."""

from __future__ import annotations

import re
from typing import Any, Sequence

_DEFAULT_LOCAL_PREFIXES = ("content_image/", "image/")


def normalize_prefixes(local_prefixes: Sequence[str] | None) -> tuple[str, ...]:
    prefixes = tuple(local_prefixes or _DEFAULT_LOCAL_PREFIXES)
    if not prefixes:
        raise ValueError("local_prefixes must not be empty")
    return prefixes


def is_remote_ref(ref: str) -> bool:
    return ref.lower().startswith(("http://", "https://", "//"))


def is_local_ref(
    ref: str,
    *,
    local_prefixes: Sequence[str] | None = None,
) -> bool:
    """Return True when *ref* points to a supported local image asset."""
    if not ref or is_remote_ref(ref):
        return False
    return ref.startswith(normalize_prefixes(local_prefixes))


def append_text_segment(segments: list[dict[str, Any]], text: str) -> None:
    """Append text, merging into the previous text segment if adjacent."""
    if not text:
        return
    if segments and segments[-1]["type"] == "text":
        segments[-1]["text"] += text
    else:
        segments.append({"type": "text", "text": text})


def pre_segment_text(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split long text segments at paragraph boundaries.

    Gives offline rebuild finer-grained split points so the
    boundary-preserving policy can find good cut locations.
    Image segments pass through unchanged.
    """
    result: list[dict[str, Any]] = []
    for seg in segments:
        if seg["type"] != "text":
            result.append(seg)
            continue
        for para in re.split(r"\n\n+", seg["text"]):
            stripped = para.strip()
            if stripped:
                result.append({"type": "text", "text": stripped})
    return result
