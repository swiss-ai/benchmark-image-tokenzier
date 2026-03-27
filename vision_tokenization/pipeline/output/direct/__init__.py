"""Direct-write output helpers.

These classes write final Megatron-compatible micro-shards immediately during
tokenization, without going through spill/rebuild.
"""

from .handler import TokenizationHandler
from .writer import MicroShardWriter, SplitMicroShardWriter

__all__ = [
    "MicroShardWriter",
    "SplitMicroShardWriter",
    "TokenizationHandler",
]
