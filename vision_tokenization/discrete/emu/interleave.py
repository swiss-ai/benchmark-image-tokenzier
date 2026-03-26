#!/usr/bin/env python3
"""EMU tokenizer for interleaved document sequences."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Sequence

import numpy as np
import torch

from .image_only import EMUImageOnlyTokenizer

# Canonical assembly/splitting logic lives in sequence_assembly.
# Re-export for backward compatibility (existing tests import from here).
from vision_tokenization.pipeline.assembly import (  # noqa: F401
    assemble_interleaved_sequence,
    split_interleaved_sequence,
)

logger = logging.getLogger(__name__)


class EMUInterleaveTokenizer(EMUImageOnlyTokenizer):
    """Tokenizer for plain interleaved document sequences."""

    def __init__(self, *args, max_sequence_tokens: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="TokenizerPool")
        self.max_sequence_tokens = (
            int(max_sequence_tokens) if max_sequence_tokens is not None else None
        )

    def close(self) -> None:
        executor = getattr(self, "executor", None)
        if executor is None:
            return
        executor.shutdown(wait=True)
        self.executor = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):  # pragma: no cover - best-effort cleanup only
        try:
            self.close()
        except Exception:
            pass

    def tokenize(self, image=None, text=None) -> torch.Tensor:
        segments = text or []
        images = list(image or [])

        text_chunks = []
        for seg in segments:
            if seg.get("type") == "text" and seg.get("text"):
                ids = self.text_tokenizer(
                    seg["text"],
                    truncation=False,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"].squeeze(0)
                text_chunks.append(ids.cpu())

        image_chunks = [self.tokenize_image(img)[1:-1].cpu() for img in images]

        return assemble_interleaved_sequence(
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            segments=segments,
            text_token_chunks=text_chunks,
            image_token_chunks=image_chunks,
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

        text_segments_flat: list[str] = []
        doc_text_positions: list[list[int]] = []
        for segments in text:
            doc_positions: list[int] = []
            for seg in segments:
                if seg.get("type") == "text" and seg.get("text"):
                    doc_positions.append(len(text_segments_flat))
                    text_segments_flat.append(seg["text"])
            doc_text_positions.append(doc_positions)

        def tokenize_texts_cpu():
            if not text_segments_flat:
                return []
            with torch.cuda.device(-1):
                encoded = self.text_tokenizer(
                    text_segments_flat,
                    truncation=False,
                    add_special_tokens=False,
                    return_tensors=None,
                    padding=False,
                )
                return [torch.tensor(ids, dtype=torch.long) for ids in encoded["input_ids"]]

        image_future = None
        if self.executor is None:
            raise RuntimeError("Tokenizer executor has been closed")
        if images:
            image_future = self.executor.submit(self.tokenize_images, images, resize_size)
        text_future = self.executor.submit(tokenize_texts_cpu)

        text_chunks = text_future.result()
        image_tokens_batch = image_future.result().cpu() if image_future is not None else None

        results = []
        for g_idx, (gs, ge) in enumerate(group_slices):
            gs, ge = int(gs), int(ge)
            segments = text[g_idx]
            text_indices = doc_text_positions[g_idx]
            text_token_chunks = [text_chunks[idx] for idx in text_indices]

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
