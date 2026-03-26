"""AtomicDocument — shared internal model for multi-image and interleave.

Normalizes all multi-image modes into one representation:
- image_only: ordered image components
- image2text: ordered image components + one text tail
- text2image: one text head + ordered image components
- sft: text with image placeholders (conversation)
- interleave: arbitrary ordered text/image sequence

Image components store structure tokens WITHOUT outer BOS/EOS.
Text components store plain text tokens WITHOUT BOS/EOS.
Wrap with BOS/EOS only in offline rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Component:
    """One component of an atomic document (image or text)."""

    component_index: int
    kind: str  # "image" or "text"

    # Offset and length into the rank-local tokens.bin
    token_offset: int = 0
    token_length: int = 0

    # Image-only metadata
    resize_height: int = 0
    resize_width: int = 0

    # Source metadata for audit / debugging
    manifest_row: int = -1


@dataclass
class AtomicDocument:
    """One logical sample: all images + text from one group/document."""

    document_id: int
    mode: str  # "image_only", "image2text", "text2image", "sft", "interleave"
    components: List[Component] = field(default_factory=list)

    # Aggregate stats (filled after tokenization)
    total_tokens: int = 0
    image_tokens: int = 0
    text_tokens: int = 0

    # Source metadata
    manifest_group_id: int = -1

    @property
    def num_components(self) -> int:
        return len(self.components)

    @property
    def image_components(self) -> List[Component]:
        return [c for c in self.components if c.kind == "image"]

    @property
    def text_components(self) -> List[Component]:
        return [c for c in self.components if c.kind == "text"]
