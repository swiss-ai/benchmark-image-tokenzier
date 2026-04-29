"""Compatibility re-export for Megatron indexed dataset builders.

Historically these builders lived under ``vision_tokenization.utils``.
They now live in ``vision_tokenization.formats.megatron``.
"""

from vision_tokenization.formats.megatron import (
    IndexedDatasetBuilder,
    VisionTokenIndexedDatasetBuilder,
)

__all__ = [
    "IndexedDatasetBuilder",
    "VisionTokenIndexedDatasetBuilder",
]
