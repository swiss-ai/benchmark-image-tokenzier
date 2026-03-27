"""Interleave segment parsers.

Each parser module exposes a ``parse(payload, **kwargs)`` function that
returns the canonical segment IR::

    [{"type": "text", "text": "..."}, {"type": "image"}, ...]

Config: ``parser: medpix`` dispatches to ``parsers/medpix.py``.
"""

from __future__ import annotations

import importlib
from typing import Any

# Registry: parser name -> module name under vision_tokenization.parsers
_REGISTRY = {
    "pin200m": "pin200m",
    "pin_markdown": "pin200m",     # alias for backward compat
    "shizhen": "shizhen",
    "content_array": "shizhen",    # alias for backward compat
    "medpix": "medpix",
    "molmo_syn": "molmo_syn",
}


def parse_segments(
    payload: Any,
    *,
    parser: str,
    **kwargs,
) -> list[dict[str, Any]]:
    """Parse raw text/payload into ordered interleave segments.

    Args:
        payload: Raw text (str), structured content (list/dict), etc.
        parser: Parser name — ``medpix``, ``pin200m``, or ``shizhen``.
        **kwargs: Passed to the parser (e.g., ``num_images``,
            ``local_prefixes``).
    """
    module_name = _REGISTRY.get(parser)
    if module_name is None:
        raise ValueError(
            f"Unknown parser: {parser!r}. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )

    mod = importlib.import_module(f".{module_name}", __package__)
    return mod.parse(payload, **kwargs)
