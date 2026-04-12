"""Dataset parser entrypoints.

Two parser families live under ``vision_tokenization.parsers``:

- ``parse_segments(...)`` for interleave-style parsers that return
  ``text/image`` segment IR.
- ``parse_sft_messages(...)`` for SFT parsers that return canonical message
  lists ready for conversation normalization.
"""

from __future__ import annotations

import importlib
from typing import Any

from .sft import parse_messages as parse_sft_messages

# Registry: parser name -> module name under vision_tokenization.parsers
_REGISTRY = {
    "pin200m": "pin200m",
    "pin_markdown": "pin200m",     # alias for backward compat
    "shizhen": "shizhen",
    "content_array": "shizhen",    # alias for backward compat
    "medpix": "medpix",
    "molmo_syn": "molmo_syn",
    "multilingual_recap": "multilingual_recap",
    "recap_multilingual": "multilingual_recap",  # alias
}

# Interleave parsers that take a full row dict (with multiple named fields)
# instead of a single-column document payload as their positional argument.
# Any parser NOT listed here is assumed to take the value of a single
# ``parser_columns`` entry as its positional payload.
_ROW_SHAPED_INTERLEAVE_PARSERS: frozenset[str] = frozenset({"molmo_syn"})


def is_row_shaped_interleave_parser(parser: str) -> bool:
    """Return True if the named interleave parser expects a full row dict."""
    return parser in _ROW_SHAPED_INTERLEAVE_PARSERS


def parse_segments(
    payload: Any,
    *,
    parser: str,
    **kwargs,
) -> list[dict[str, Any]]:
    """Parse raw text/payload into ordered interleave segments.

    Args:
        payload: For document-shaped parsers (pin200m, shizhen, medpix,
            multilingual_recap) — the raw column value (string or list).
            For row-shaped parsers (molmo_syn) — the full row dict.
        parser: Parser name.
        **kwargs: Passed to the parser (e.g., ``num_images``, ``local_prefixes``).
    """
    module_name = _REGISTRY.get(parser)
    if module_name is None:
        raise ValueError(
            f"Unknown parser: {parser!r}. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )

    mod = importlib.import_module(f".{module_name}", __package__)
    return mod.parse(payload, **kwargs)
