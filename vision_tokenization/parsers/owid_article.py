"""Parser for OWID ``articles.parquet`` — DOM-ordered interleave.

Each parquet row has:
  - ``images_bytes``: list[bytes] of chart/hero PNGs (1:1 with image_index)
  - ``segments_json``: JSON list of ordered DOM segments::

        {"type": "text",  "value": "<paragraph>"}
        {"type": "image", "image_index": N, "filename": "...", "alt": "..."}

  - ``title``: article H1 (prepended as ``# <title>``)
  - ``subtitle``: one-sentence summary (prepended after title, if present)

Produces interleave segments in DOM order::

    [text(# title), text(subtitle), text(para), image, text(alt?), text(para), image, ..., text(para)]

Heading levels (``## Section``, ``### Subsection``) are already baked
into ``segments_json`` as text markers, so the parser just prepends the
H1 title and the subtitle lead-in without otherwise touching structure.

Alt handling: when ``alt`` is empty the image is not followed by an alt
paragraph (≈44% of article images have no alt). The image count emitted
by the parser is clamped to ``num_images`` (the manifest's
ground-truth count from the HF scanner); segments referencing an
``image_index >= num_images`` are skipped so image-count parity with
the scanner is preserved.
"""
from __future__ import annotations

import json
from typing import Any


def parse(
    row: dict,
    *,
    num_images: int = 0,
    **kwargs,
) -> list[dict[str, Any]]:
    segments_raw = row.get("segments_json") or "[]"
    segments = json.loads(segments_raw) if isinstance(segments_raw, str) else segments_raw
    title = (row.get("title") or "").strip()
    subtitle = (row.get("subtitle") or "").strip()

    out: list[dict[str, Any]] = []
    if title:
        out.append({"type": "text", "text": f"# {title}"})
    if subtitle:
        out.append({"type": "text", "text": subtitle})

    for seg in segments:
        stype = seg.get("type")
        if stype == "text":
            text = (seg.get("value") or "").strip()
            if text:
                out.append({"type": "text", "text": text})
        elif stype == "image":
            idx = seg.get("image_index")
            if idx is None or idx >= num_images:
                continue
            out.append({"type": "image"})
            alt = (seg.get("alt") or "").strip()
            if alt:
                out.append({"type": "text", "text": alt})

    return out
