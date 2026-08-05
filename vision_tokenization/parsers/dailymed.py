"""Parser for DailyMed SPL interleaved dataset.

Each parquet row has ``segments_json`` — a JSON-encoded list of
segments already in canonical reading-order interleave shape::

    {"type": "text",  "text": "<body>"}
    {"type": "image", "image_index": <i>, "filename": "...", "alt": "<VLM-generated description>"}

This parser decodes the JSON and emits canonical segments. Each image
segment's ``alt`` text (a VLM-generated description) is appended as a
text segment immediately after the corresponding image — this keeps
each image paired with its description, since dailymed's pure-text
segments are often empty and the alt is the dominant text signal.

The image segments are emitted in their original document order so
``image_index`` matches the manifest. Extra text segments do not affect
that ordering.
"""

from __future__ import annotations

import json
from typing import Any

# Pre-built frozen segment dict to avoid per-call allocation.
_IMAGE_SEG = {"type": "image"}


def parse(
    payload: Any,
    *,
    num_images: int = 0,
    **kwargs,
) -> list[dict[str, Any]]:
    """Decode ``segments_json`` into canonical interleave segments.

    Args:
        payload: The raw ``segments_json`` column value (string of JSON).
        num_images: Provided by the framework — unused here because the
            count is implicit in the JSON itself; passed for interface
            consistency.

    Returns:
        Ordered segment list interleaving text and image, with each
        image followed by its ``alt`` text segment when non-empty.
    """
    if not payload:
        return []

    try:
        raw = json.loads(payload)
    except (TypeError, ValueError):
        return []

    out: list[dict[str, Any]] = []
    for s in raw:
        t = s.get("type")
        if t == "text":
            text = s.get("text") or ""
            if text:
                out.append({"type": "text", "text": text})
        elif t == "image":
            out.append(_IMAGE_SEG)
            alt = s.get("alt") or ""
            if alt:
                out.append({"type": "text", "text": alt})
    return out
