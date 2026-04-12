#!/usr/bin/env python3
"""EMU tokenizer for SFT (Supervised Fine-Tuning) data.

Renders chat templates to text, splits into ordered ``text`` / ``image``
segments, and assembles the final sequence structurally.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from vision_tokenization.common.assembly import assemble_sft_sequence
from vision_tokenization.discrete.conversation import ConversationPolicy
from vision_tokenization.discrete.sft_segments import (
    ChatTemplateSFTDocumentRenderer,
    RenderedSFTDocument,
    build_segment_component_maps,
)

from ._mixins import ThreadPoolExecutorOwner
from .image_only import EMUImageOnlyTokenizer

logger = logging.getLogger(__name__)


class EMUSftTokenizer(ThreadPoolExecutorOwner, EMUImageOnlyTokenizer):
    """Tokenizer for SFT (Supervised Fine-Tuning) data.

    Supports one or more images per conversation in batched execution.
    """

    def __init__(self, *args, conversation_policy: Optional[ConversationPolicy] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_policy = conversation_policy or ConversationPolicy()
        # image_token_id is consumed by StructureTokenIds in the rebuild path.
        self.image_token_id = self.text_tokenizer.convert_tokens_to_ids("<|image|>")
        self._sft_renderer = ChatTemplateSFTDocumentRenderer(
            text_tokenizer=self.text_tokenizer,
            conversation_policy=self.conversation_policy,
        )

    def render_sft_document(
        self,
        raw_text: List[Dict[str, Any]],
        *,
        expected_num_images: Optional[int] = None,
    ) -> RenderedSFTDocument:
        return self._sft_renderer.render_document(
            raw_text, expected_num_images=expected_num_images,
        )

    def render_sft_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        expected_num_images: Optional[int] = None,
    ) -> RenderedSFTDocument:
        return self._sft_renderer.render_messages(
            messages, expected_num_images=expected_num_images,
        )

    def tokenize_batch(
        self,
        images: List[Any],
        resize_size: Tuple[int, int],
        text: Optional[List[Any]] = None,
        group_slices: Optional[np.ndarray] = None,
    ) -> List[Optional[torch.Tensor]]:
        """Batched SFT tokenization. GPU image encode runs ∥ CPU chat render.

        Args:
            images: Flat list of PIL Images for the batch.
            resize_size: Batch-wide resize target.
            text: One conversation per group (or per image when ``group_slices``
                is ``None``).
            group_slices: Optional ``(num_groups, 2)`` int64 array mapping each
                group to a contiguous slice of *images*. ``None`` means each
                image is its own 1-image group.

        Returns:
            One entry per group: a token tensor, or ``None`` if the group was
            skipped (render or assembly failure).
        """
        if text is None or len(text) == 0:
            raise ValueError("Text (conversations) is required for SFT tokenization")
        group_slices = self._normalize_group_slices(group_slices, len(images), len(text))

        image_future = (
            self.executor.submit(self.tokenize_images, images, resize_size)
            if images else None
        )
        text_future = self.executor.submit(
            self._render_and_tokenize_groups, text, group_slices,
        )

        segment_groups, text_chunks_by_group = text_future.result()
        image_tokens_batch = image_future.result() if image_future is not None else None

        results: List[Optional[torch.Tensor]] = []
        for g_idx, (gs, ge) in enumerate(group_slices):
            segments = segment_groups[g_idx]
            text_token_chunks = text_chunks_by_group[g_idx]
            if segments is None or text_token_chunks is None:
                results.append(None)
                continue

            gs_i, ge_i = int(gs), int(ge)
            image_token_chunks = [
                image_tokens_batch[i, 1:-1] for i in range(gs_i, ge_i)
            ]
            try:
                results.append(self._assemble_sft_group(
                    segments=segments,
                    text_token_chunks=text_token_chunks,
                    image_token_chunks=image_token_chunks,
                ))
            except ValueError as exc:
                logger.warning(
                    "Failed to assemble SFT group %d — skipping: %s", g_idx, exc,
                )
                results.append(None)
        return results

    @staticmethod
    def _normalize_group_slices(
        group_slices: Optional[np.ndarray],
        n_images: int,
        n_texts: int,
    ) -> np.ndarray:
        """Validate or synthesize a ``(num_groups, 2)`` int64 group_slices array.

        When ``group_slices is None``, each image becomes its own 1-image group
        and ``n_images == n_texts`` is required.

        When provided, the array must:
        - have shape ``(num_groups, 2)`` and dtype int64-castable,
        - contain monotonic non-overlapping slices,
        - satisfy ``0 <= start <= end <= n_images`` for every entry,
        - have exactly ``n_texts`` rows.

        Note on coverage: gaps between groups are *allowed* — i.e. an image
        whose index is not covered by any group's ``[start, end)`` is silently
        dropped. This preserves the current permissive contract; tightening it
        (requiring every image to belong to exactly one group) would be a
        separate semantic change.
        """
        if group_slices is None:
            if n_images != n_texts:
                raise ValueError(
                    f"Number of images ({n_images}) must match "
                    f"number of conversations ({n_texts})"
                )
            return np.array(
                [[i, i + 1] for i in range(n_images)], dtype=np.int64,
            ).reshape(-1, 2)

        gs_arr = np.asarray(group_slices, dtype=np.int64)
        if gs_arr.ndim != 2 or gs_arr.shape[1] != 2:
            raise ValueError(
                f"group_slices must have shape (num_groups, 2), got {gs_arr.shape}"
            )
        if len(gs_arr) != n_texts:
            raise ValueError(
                f"group_slices has {len(gs_arr)} groups but received {n_texts} conversations"
            )
        if len(gs_arr) > 0:
            starts = gs_arr[:, 0]
            ends = gs_arr[:, 1]
            if (starts < 0).any() or (ends > n_images).any() or (starts > ends).any():
                raise ValueError(
                    f"group_slices entries must satisfy 0 <= start <= end <= {n_images}"
                )
            if len(gs_arr) > 1 and (starts[1:] < ends[:-1]).any():
                raise ValueError("group_slices must be monotonic and non-overlapping")
        return gs_arr

    def _render_and_tokenize_groups(
        self,
        text: List[Any],
        group_slices: np.ndarray,
    ) -> Tuple[
        List[Optional[List[Dict[str, Any]]]],
        List[Optional[List[torch.Tensor]]],
    ]:
        """Render each group's chat then batch-tokenize all text spans at once."""
        segment_groups = self._render_groups_to_segments(text, group_slices)
        text_chunks_by_group = self._batch_tokenize_text_spans(segment_groups)
        return segment_groups, text_chunks_by_group

    def _render_groups_to_segments(
        self,
        text: List[Any],
        group_slices: np.ndarray,
    ) -> List[Optional[List[Dict[str, Any]]]]:
        """Render each group's conversation into structured segments.

        ``tokenize_batch`` never sees fragmented docs — all of a doc's images
        are passed together — so the group's image count is the conversation's
        expected placeholder count. Multi-image SFT with batch fragmentation
        goes through ``SpillBackend`` instead.
        """
        segment_groups: List[Optional[List[Dict[str, Any]]]] = []
        for g_idx, (gs, ge) in enumerate(group_slices):
            try:
                rendered_doc = self.render_sft_document(
                    text[g_idx],
                    expected_num_images=int(ge) - int(gs),
                )
                segment_groups.append(rendered_doc.segments)
            except ValueError as exc:
                logger.warning(
                    "Failed to render SFT conversation for group %d — skipping: %s",
                    g_idx,
                    exc,
                )
                segment_groups.append(None)
        return segment_groups

    def _batch_tokenize_text_spans(
        self,
        segment_groups: List[Optional[List[Dict[str, Any]]]],
    ) -> List[Optional[List[torch.Tensor]]]:
        """Flatten text spans across all groups, batch-tokenize, then regroup."""
        flat_texts, doc_text_map, _ = build_segment_component_maps(segment_groups)
        flat_token_chunks = self._tokenize_flat_texts_cpu(flat_texts)

        text_chunks_by_group: List[Optional[List[torch.Tensor]]] = []
        for segments, text_entries in zip(segment_groups, doc_text_map):
            if segments is None:
                text_chunks_by_group.append(None)
                continue
            text_chunks_by_group.append(
                [flat_token_chunks[idx] for _, idx in text_entries]
            )
        return text_chunks_by_group

    def _assemble_sft_group(
        self,
        *,
        segments: List[Dict[str, Any]],
        text_token_chunks: List[torch.Tensor],
        image_token_chunks: List[torch.Tensor],
    ) -> torch.Tensor:
        """Assemble one SFT group's structured segments into a final token sequence.

        Precondition: ``segments`` and ``text_token_chunks`` are both non-None.
        The caller (``tokenize_batch``) handles the skip-on-None case so this
        helper has a clean group-local contract.
        """
        return assemble_sft_sequence(
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            segments=segments,
            text_token_chunks=text_token_chunks,
            image_token_chunks=image_token_chunks,
        )
