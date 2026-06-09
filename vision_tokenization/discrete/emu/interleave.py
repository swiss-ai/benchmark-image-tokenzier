#!/usr/bin/env python3
"""EMU tokenizer for interleaved document sequences."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

import numpy as np
import torch

from ._mixins import ThreadPoolExecutorOwner
from .image_only import EMUImageOnlyTokenizer

# Canonical assembly/splitting logic lives in common.assembly.
# Re-export for backward compatibility (existing tests import from here).
from vision_tokenization.common.assembly import (  # noqa: F401
    assemble_interleaved_sequence,
    split_interleaved_sequence,
)
from vision_tokenization.discrete.sft_segments import build_segment_component_maps

logger = logging.getLogger(__name__)


class EMUInterleaveTokenizer(ThreadPoolExecutorOwner, EMUImageOnlyTokenizer):
    """Tokenizer for plain interleaved document sequences."""

    def __init__(self, *args, max_sequence_tokens: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_sequence_tokens = (
            int(max_sequence_tokens) if max_sequence_tokens is not None else None
        )

    def tokenize_batch(self, images, resize_size, text=None, group_slices=None):
        """Tokenize grouped interleaved documents.

        Args:
            images: Flat list of images across groups, in manifest order.
            resize_size: Batch-wide resize target used by the vision tokenizer.
            text: Required list of structured document segments, one per group.
            group_slices: Required ``(num_groups, 2)`` array mapping groups to
                positions in *images*.
        """
        if text is None or len(text) == 0:
            raise ValueError("Structured interleave text is required")
        if group_slices is None:
            raise ValueError("interleave mode requires group_slices")
        if len(text) != len(group_slices):
            raise ValueError(
                f"Number of documents ({len(text)}) must match number of groups ({len(group_slices)})"
            )
        if self.executor is None:
            raise RuntimeError("Tokenizer executor has been closed")

        flat_texts, doc_text_map, _doc_image_positions = build_segment_component_maps(text)

        image_future = (
            self.executor.submit(self.tokenize_images, images, resize_size)
            if len(images) > 0 else None  # len(): images may be a preprocessed Tensor
        )
        text_future = self.executor.submit(self._tokenize_flat_texts_cpu, flat_texts)

        text_chunks = text_future.result()
        image_tokens_batch = image_future.result() if image_future is not None else None

        results = []
        for g_idx, (gs, ge) in enumerate(group_slices):
            gs, ge = int(gs), int(ge)
            segments = text[g_idx]
            text_token_chunks = [text_chunks[flat_idx] for _runtime_ci, flat_idx in doc_text_map[g_idx]]

            image_count_expected = sum(1 for seg in segments if seg.get("type") == "image")
            image_count_actual = ge - gs
            if image_count_expected != image_count_actual:
                logger.warning(
                    "Interleave group has %d parsed image segments but %d manifest images — skipping",
                    image_count_expected,
                    image_count_actual,
                )
                results.append(None)
                continue

            if image_count_actual > 0:
                assert image_tokens_batch is not None
                image_token_chunks = [
                    image_tokens_batch[i, 1:-1]
                    for i in range(gs, ge)
                ]
            else:
                image_token_chunks = []

            try:
                split_sequences = split_interleaved_sequence(
                    bos_id=self.bos_id,
                    eos_id=self.eos_id,
                    segments=segments,
                    text_token_chunks=text_token_chunks,
                    image_token_chunks=image_token_chunks,
                    max_sequence_tokens=self.max_sequence_tokens,
                )
            except ValueError as exc:
                logger.warning(
                    "Interleave group cannot fit within max_sequence_tokens=%s without "
                    "breaking a segment boundary — skipping: %s",
                    self.max_sequence_tokens,
                    exc,
                )
                results.append(None)
                continue

            results.extend(split_sequences)

        return results
