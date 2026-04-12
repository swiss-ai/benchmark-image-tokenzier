"""Shared helpers for rendering SFT conversations into text/image segments."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any, Optional, Sequence

from .conversation import ConversationPolicy, apply_conversation_policy


@lru_cache(maxsize=8)
def _compile_marker_pattern(markers: tuple[str, ...]) -> re.Pattern:
    return re.compile(
        "|".join(re.escape(m) for m in sorted(markers, key=len, reverse=True))
    )


@dataclass(frozen=True)
class RenderedSFTDocument:
    """Rendered chat template plus ordered structural segments."""

    rendered_text: str
    segments: list[dict[str, Any]]


class ChatTemplateSFTDocumentRenderer:
    """Tokenizer-backed renderer for chat-template SFT documents."""

    def __init__(
        self,
        *,
        text_tokenizer: Any,
        conversation_policy: ConversationPolicy,
    ) -> None:
        self.text_tokenizer = text_tokenizer
        self.conversation_policy = conversation_policy
        self.image_marker_candidates = build_image_marker_candidates(
            text_tokenizer=text_tokenizer,
            conversation_policy=conversation_policy,
        )

    def render_document(
        self,
        raw_messages: list[dict[str, Any]],
        *,
        expected_num_images: Optional[int] = None,
    ) -> RenderedSFTDocument:
        """Normalize a raw SFT sample, then render."""
        messages = apply_conversation_policy(raw_messages, self.conversation_policy)
        return self.render_messages(messages, expected_num_images=expected_num_images)

    def render_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        expected_num_images: Optional[int] = None,
    ) -> RenderedSFTDocument:
        """Render already-normalized chat messages."""
        return render_sft_segments(
            messages,
            text_tokenizer=self.text_tokenizer,
            image_marker_candidates=self.image_marker_candidates,
            expected_num_images=expected_num_images,
        )


def build_image_marker_candidates(
    *,
    text_tokenizer: Any,
    conversation_policy: ConversationPolicy,
) -> tuple[str, ...]:
    """Derive the rendered image marker strings to accept for one tokenizer.

    The rendered chat may contain either the dataset-level placeholder such as
    ``<image>`` or a model-specific special token like ``<|image|>``; both are
    accepted.
    """
    image_token_id = text_tokenizer.convert_tokens_to_ids("<|image|>")
    return tuple(_unique_non_empty([
        text_tokenizer.convert_ids_to_tokens(image_token_id),
        text_tokenizer.decode([image_token_id], skip_special_tokens=False),
        conversation_policy.image_placeholder,
        "<|image|>",
    ]))


def render_sft_segments(
    messages: list[dict[str, Any]],
    *,
    text_tokenizer: Any,
    image_marker_candidates: Sequence[str],
    expected_num_images: Optional[int] = None,
) -> RenderedSFTDocument:
    """Render a normalized conversation and split it into SFT segments.

    ``apply_chat_template(..., tokenize=False)`` keeps the model-specific chat
    prompt formatting, while the returned segment list keeps image placement as
    explicit structure instead of relying on token-id scanning later.
    """
    rendered = text_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str):
        raise ValueError(
            f"apply_chat_template(tokenize=False) must return str, got {type(rendered).__name__}"
        )

    selected_markers = _select_image_markers(
        rendered,
        image_marker_candidates,
        expected_num_images=expected_num_images,
    )
    segments = split_rendered_sft(rendered, selected_markers)
    num_slots = count_image_segments(segments)
    if expected_num_images is not None and num_slots != expected_num_images:
        raise ValueError(
            f"Rendered chat has {num_slots} image slots but {expected_num_images} images"
        )
    return RenderedSFTDocument(rendered_text=rendered, segments=segments)


def split_rendered_sft(
    rendered_text: str,
    image_markers: Sequence[str],
) -> list[dict[str, Any]]:
    """Split a rendered chat string into ordered text/image segments."""
    markers = _unique_non_empty(image_markers)
    if not markers:
        return [{"type": "text", "text": rendered_text}] if rendered_text else []

    pattern = _compile_marker_pattern(tuple(markers))
    segments: list[dict[str, Any]] = []
    last_pos = 0
    for match in pattern.finditer(rendered_text):
        if match.start() > last_pos:
            segments.append({"type": "text", "text": rendered_text[last_pos:match.start()]})
        segments.append({"type": "image"})
        last_pos = match.end()

    if last_pos < len(rendered_text):
        segments.append({"type": "text", "text": rendered_text[last_pos:]})
    return segments


def count_image_segments(segments: Sequence[dict[str, Any]]) -> int:
    """Return the number of structural image slots in a segment list."""
    return sum(1 for seg in segments if seg.get("type") == "image")


def build_segment_component_maps(
    segment_groups: Sequence[Optional[Sequence[dict[str, Any]]]],
) -> tuple[list[str], list[list[tuple[int, int]]], list[list[int]]]:
    """Flatten text spans and build runtime component maps per document.

    Returns:
        ``flat_texts``:
            All non-empty text spans across every document, batch-flattened.
        ``doc_text_map``:
            For each document, ``(runtime_component_index, flat_text_index)``
            pairs for every text span.
        ``doc_image_positions``:
            For each document, runtime component indices for image slots,
            ordered by image occurrence.
    """
    flat_texts: list[str] = []
    doc_text_map: list[list[tuple[int, int]]] = []
    doc_image_positions: list[list[int]] = []

    for segments in segment_groups:
        text_entries: list[tuple[int, int]] = []
        image_positions: list[int] = []
        runtime_ci = 0

        if segments is None:
            doc_text_map.append(text_entries)
            doc_image_positions.append(image_positions)
            continue

        for seg in segments:
            seg_type = seg.get("type")
            if seg_type == "text":
                seg_text = seg.get("text")
                if seg_text:
                    text_entries.append((runtime_ci, len(flat_texts)))
                    flat_texts.append(seg_text)
                    runtime_ci += 1
            elif seg_type == "image":
                image_positions.append(runtime_ci)
                runtime_ci += 1
            else:
                raise ValueError(f"Unsupported SFT segment type: {seg_type!r}")

        doc_text_map.append(text_entries)
        doc_image_positions.append(image_positions)

    return flat_texts, doc_text_map, doc_image_positions


def _select_image_markers(
    rendered_text: str,
    image_marker_candidates: Sequence[str],
    *,
    expected_num_images: Optional[int],
) -> tuple[str, ...]:
    markers = tuple(marker for marker in _unique_non_empty(image_marker_candidates) if marker in rendered_text)
    if expected_num_images is None:
        return markers

    if expected_num_images == 0:
        if markers:
            total = sum(rendered_text.count(marker) for marker in markers)
            if total != 0:
                raise ValueError(
                    "Rendered chat unexpectedly contains image markers for a text-only sample"
                )
        return ()

    if not markers:
        raise ValueError(
            f"Rendered chat contains no recognized image marker; expected {expected_num_images} images"
        )

    for marker in markers:
        if rendered_text.count(marker) == expected_num_images:
            return (marker,)

    total = sum(rendered_text.count(marker) for marker in markers)
    if total == expected_num_images:
        return markers

    counts = {marker: rendered_text.count(marker) for marker in markers}
    raise ValueError(
        f"Rendered chat marker counts {counts} do not match expected_num_images={expected_num_images}"
    )


def _unique_non_empty(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
