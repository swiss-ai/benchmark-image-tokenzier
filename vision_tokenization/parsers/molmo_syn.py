"""Parser for Molmo2-SynMultiImageQA dataset.

Each parquet row has:
  - images: list of image dicts (bytes + path)
  - code: list of code strings (1:1 with images)
  - metadata.overall_description: summary text

Produces interleave segments alternating image/code pairs,
with overall_description appended at the end.
"""

from __future__ import annotations

from typing import Any

# Pre-built frozen segment dicts to avoid per-call allocation.
_IMAGE_SEG = {"type": "image"}


def parse(
    row: dict,
    *,
    num_images: int = 0,
    **kwargs,
) -> list[dict[str, Any]]:
    """Parse a Molmo-SynMultiImageQA row into interleave segments.

    Args:
        row: Dict with ``code`` (list[str]) and ``metadata`` (dict with
            ``overall_description``).
        num_images: Number of images in this sample (from manifest).

    Returns:
        Ordered segment list::

            [image, text(code_0), image, text(code_1), ..., text(desc)]
    """
    codes = row.get("code") or ()
    metadata = row.get("metadata")
    overall_desc = metadata.get("overall_description", "") if isinstance(metadata, dict) else ""

    # Total images in the document — from metadata (ground truth), not batch group size.
    if isinstance(metadata, dict) and metadata.get("num_images"):
        n = metadata["num_images"]
    elif codes:
        n = len(codes)
    else:
        n = num_images
    # Pre-size: n images + up to n code texts + 1 description
    segments: list[dict[str, Any]] = []

    for i in range(n):
        segments.append(_IMAGE_SEG)
        if i < len(codes):
            code = codes[i]
            if code:
                segments.append({"type": "text", "text": code})

    if overall_desc:
        segments.append({"type": "text", "text": overall_desc})

    return segments
