"""MedPix: plain text with ``<|imgN|>`` placeholders."""

from __future__ import annotations

import re
from typing import Any

from .common import pre_segment_text

_MEDPIX_IMAGE_PATTERN = re.compile(r"<\|img\d+\|>")


def parse(
    text: str | None,
    *,
    num_images: int = 0,
    **kwargs,
) -> list[dict[str, Any]]:
    """Parse medpix text into ordered interleave segments.

    Falls back to image2text layout (images then text) when no
    ``<|imgN|>`` markers are found.
    """
    raw = (text or "").strip()
    if not raw:
        return [{"type": "image"} for _ in range(num_images)]

    if not _MEDPIX_IMAGE_PATTERN.search(raw):
        # No markers — fall back to image2text
        segments: list[dict[str, Any]] = [{"type": "image"} for _ in range(num_images)]
        segments.append({"type": "text", "text": raw})
        return pre_segment_text(segments)

    parts = _MEDPIX_IMAGE_PATTERN.split(raw)
    segments = []
    img_count = 0

    for i, part in enumerate(parts):
        stripped = part.strip()
        if stripped:
            segments.append({"type": "text", "text": stripped})
        if i < len(parts) - 1:
            segments.append({"type": "image"})
            img_count += 1

    while img_count < num_images:
        segments.append({"type": "image"})
        img_count += 1

    return pre_segment_text(segments)
