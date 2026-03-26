"""Pure-CPU sequence assembly and splitting helpers.

These functions take pre-tokenized components (image structure tokens, text
tokens) and assemble them into final training sequences.  They have no
dependency on a loaded tokenizer instance — all special token IDs are passed
explicitly.

The online tokenization stage stores components WITHOUT outer BOS/EOS.
These helpers wrap with BOS/EOS during offline rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union

import torch


# ---------------------------------------------------------------------------
# Token ID configuration — replaces tokenizer instance dependency
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StructureTokenIds:
    """All special token IDs needed for assembly, independent of tokenizer."""

    bos_id: int
    eos_id: int
    img_start_id: int
    img_end_id: int
    img_token_start_id: int
    eol_id: int
    eof_id: int
    vision_token_offset: int
    image_token_id: int  # <|image|> placeholder for SFT mode

    # Callable that returns dimension token IDs for a given (height, width)
    # in token space.  During online stage, backed by tokenizer cache.
    # During offline rebuild, not needed (components are pre-tokenized).
    dim_tokens_fn: Optional[Callable[[int, int], List[int]]] = None


# ---------------------------------------------------------------------------
# Image structure encapsulation (WITHOUT BOS/EOS)
# ---------------------------------------------------------------------------

def encapsulate_image_structure(
    image_indices: torch.Tensor,
    height: int,
    width: int,
    token_ids: StructureTokenIds,
) -> torch.Tensor:
    """Build image structure tokens from raw vision indices.

    Returns the inner structure WITHOUT outer BOS/EOS::

        img_start + dim_tokens + img_token_start +
        [vision_tokens + EOL per row] + EOF + img_end

    Args:
        image_indices: Flat tensor of vision codebook indices [H*W].
        height: Image height in token-space (after spatial downsampling).
        width: Image width in token-space.
        token_ids: Special token IDs and helpers.

    Returns:
        1-D int64 tensor of image structure tokens.
    """
    num_tokens_needed = height * width
    assert image_indices.numel() == num_tokens_needed, (
        f"Dimension mismatch: {height}x{width} needs {num_tokens_needed} "
        f"indices, got {image_indices.numel()}"
    )

    if token_ids.dim_tokens_fn is None:
        raise ValueError("dim_tokens_fn is required for image encapsulation")
    dim_tokens = token_ids.dim_tokens_fn(height, width)

    total_size = (
        1  # img_start
        + len(dim_tokens)
        + 1  # img_token_start
        + num_tokens_needed
        + height  # EOL per row
        + 1  # EOF
        + 1  # img_end
    )

    output = torch.empty(total_size, dtype=torch.long)
    idx = 0

    # Prefix: img_start + dim_tokens + img_token_start
    output[idx] = token_ids.img_start_id
    idx += 1
    output[idx : idx + len(dim_tokens)] = torch.tensor(dim_tokens, dtype=torch.long)
    idx += len(dim_tokens)
    output[idx] = token_ids.img_token_start_id
    idx += 1

    # Vision tokens with EOL markers
    image_indices = image_indices.view(height, width)
    vision_tokens = image_indices + token_ids.vision_token_offset
    vision_part = torch.empty((height, width + 1), dtype=torch.long)
    vision_part[:, :width] = vision_tokens
    vision_part[:, -1] = token_ids.eol_id
    output[idx : idx + height * (width + 1)] = vision_part.flatten()
    idx += height * (width + 1)

    # Suffix: EOF + img_end
    output[idx] = token_ids.eof_id
    output[idx + 1] = token_ids.img_end_id

    return output


def encapsulate_image_structure_batch(
    image_indices: torch.Tensor,
    height: int,
    width: int,
    token_ids: StructureTokenIds,
) -> torch.Tensor:
    """Batched version of :func:`encapsulate_image_structure`.

    Args:
        image_indices: [B, H*W] tensor of vision codebook indices.
        height: Image height in token-space.
        width: Image width in token-space.
        token_ids: Special token IDs and helpers.

    Returns:
        [B, total_structure_len] int64 tensor WITHOUT BOS/EOS.
    """
    batch_size, num_tokens_input = image_indices.shape
    num_tokens_needed = height * width
    assert num_tokens_input == num_tokens_needed

    if token_ids.dim_tokens_fn is None:
        raise ValueError("dim_tokens_fn is required for image encapsulation")
    dim_tokens = token_ids.dim_tokens_fn(height, width)

    struct_len = (
        1 + len(dim_tokens) + 1  # img_start + dims + img_token_start
        + num_tokens_needed + height  # vision + EOLs
        + 1 + 1  # EOF + img_end
    )

    output = torch.empty(
        (batch_size, struct_len), dtype=torch.long, device=image_indices.device,
    )

    # Vision part: [B, H, W] + offset, then interleave EOL
    image_2d = image_indices.view(batch_size, height, width)
    vision_with_offset = image_2d + token_ids.vision_token_offset
    vision_part = torch.empty(
        (batch_size, height, width + 1), dtype=torch.long, device=image_indices.device,
    )
    vision_part[:, :, :width] = vision_with_offset
    vision_part[:, :, width] = token_ids.eol_id
    vision_flat = vision_part.flatten(start_dim=1)

    # Prefix
    prefix = [token_ids.img_start_id, *dim_tokens, token_ids.img_token_start_id]
    prefix_t = torch.tensor(prefix, dtype=torch.long, device=image_indices.device)
    plen = len(prefix)
    output[:, :plen] = prefix_t.unsqueeze(0)

    # Vision
    vstart = plen
    vend = vstart + vision_flat.shape[1]
    output[:, vstart:vend] = vision_flat

    # Suffix
    output[:, vend] = token_ids.eof_id
    output[:, vend + 1] = token_ids.img_end_id

    return output


# ---------------------------------------------------------------------------
# Interleave assembly (unchanged semantics, now works with Component model)
# ---------------------------------------------------------------------------

def assemble_sequence(
    *,
    bos_id: int,
    eos_id: int,
    component_tokens: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Assemble ordered component tokens into a final sequence.

    Wraps with BOS/EOS::

        [BOS] + component_0 + component_1 + ... + [EOS]

    Each component should already be in its final token form (image structure
    tokens or plain text tokens), WITHOUT BOS/EOS.
    """
    parts = [torch.tensor([bos_id], dtype=torch.long)]
    parts.extend(component_tokens)
    parts.append(torch.tensor([eos_id], dtype=torch.long))
    return torch.cat(parts)


def assemble_interleaved_sequence(
    *,
    bos_id: int,
    eos_id: int,
    segments: Sequence[dict[str, Any]],
    text_token_chunks: Sequence[torch.Tensor],
    image_token_chunks: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Assemble one interleaved document sequence from tokenized chunks.

    Preserves the original segment ordering.  Components should be WITHOUT
    BOS/EOS — this function adds the outer BOS/EOS wrapper.
    """
    if sum(1 for seg in segments if seg.get("type") == "text" and seg.get("text")) != len(text_token_chunks):
        raise ValueError("Number of text token chunks does not match non-empty text segments")
    if sum(1 for seg in segments if seg.get("type") == "image") != len(image_token_chunks):
        raise ValueError("Number of image token chunks does not match image segments")

    parts: list[torch.Tensor] = [torch.tensor([bos_id], dtype=torch.long)]
    text_idx = 0
    image_idx = 0

    for seg in segments:
        seg_type = seg.get("type")
        if seg_type == "text":
            if seg.get("text"):
                parts.append(text_token_chunks[text_idx])
                text_idx += 1
        elif seg_type == "image":
            parts.append(image_token_chunks[image_idx])
            image_idx += 1
        else:
            raise ValueError(f"Unsupported interleave segment type: {seg_type!r}")

    parts.append(torch.tensor([eos_id], dtype=torch.long))
    return torch.cat(parts)


def split_interleaved_sequence(
    *,
    bos_id: int,
    eos_id: int,
    segments: Sequence[dict[str, Any]],
    text_token_chunks: Sequence[torch.Tensor],
    image_token_chunks: Sequence[torch.Tensor],
    max_sequence_tokens: Optional[int] = None,
) -> list[torch.Tensor]:
    """Assemble one or more sequences, splitting only at segment boundaries.

    Greedy boundary-preserving policy:
    - Append next component if it fits
    - Otherwise flush current sequence and start a new one
    - A single component that exceeds the limit raises ``ValueError``

    When ``max_sequence_tokens`` is ``None``, returns a single assembled sequence.
    """
    if max_sequence_tokens is None:
        return [
            assemble_interleaved_sequence(
                bos_id=bos_id,
                eos_id=eos_id,
                segments=segments,
                text_token_chunks=text_token_chunks,
                image_token_chunks=image_token_chunks,
            )
        ]

    max_sequence_tokens = int(max_sequence_tokens)
    if max_sequence_tokens < 2:
        raise ValueError("max_sequence_tokens must be >= 2")

    entries: list[tuple[str, dict[str, Any], torch.Tensor]] = []
    text_idx = 0
    image_idx = 0
    for seg in segments:
        seg_type = seg.get("type")
        if seg_type == "text":
            if seg.get("text"):
                entries.append(("text", seg, text_token_chunks[text_idx]))
                text_idx += 1
        elif seg_type == "image":
            entries.append(("image", seg, image_token_chunks[image_idx]))
            image_idx += 1
        else:
            raise ValueError(f"Unsupported interleave segment type: {seg_type!r}")

    if not entries:
        return [torch.tensor([bos_id, eos_id], dtype=torch.long)]

    sequences: list[torch.Tensor] = []
    current_entries: list[tuple[str, dict[str, Any], torch.Tensor]] = []
    current_len = 2  # BOS + EOS

    def _flush_current() -> None:
        if not current_entries:
            return
        chunk_segments = [seg for _kind, seg, _tokens in current_entries]
        chunk_texts = [tokens for kind, _seg, tokens in current_entries if kind == "text"]
        chunk_images = [tokens for kind, _seg, tokens in current_entries if kind == "image"]
        sequences.append(
            assemble_interleaved_sequence(
                bos_id=bos_id,
                eos_id=eos_id,
                segments=chunk_segments,
                text_token_chunks=chunk_texts,
                image_token_chunks=chunk_images,
            )
        )

    for kind, seg, tokens in entries:
        seg_len = int(tokens.numel())
        if seg_len + 2 > max_sequence_tokens:
            raise ValueError(
                f"Single {kind} segment requires {seg_len + 2} tokens, exceeding "
                f"max_sequence_tokens={max_sequence_tokens}"
            )
        if current_entries and current_len + seg_len > max_sequence_tokens:
            _flush_current()
            current_entries = []
            current_len = 2
        current_entries.append((kind, seg, tokens))
        current_len += seg_len

    _flush_current()
    return sequences


# ---------------------------------------------------------------------------
# SFT image placeholder replacement
# ---------------------------------------------------------------------------

def replace_image_placeholders(
    text_tokens: torch.Tensor,
    image_positions: List[int],
    image_token_chunks: List[torch.Tensor],
) -> torch.Tensor:
    """Replace ``<|image|>`` placeholders in text tokens with image structure tokens.

    Args:
        text_tokens: 1-D token tensor with placeholder tokens.
        image_positions: Sorted positions of placeholder tokens.
        image_token_chunks: Image structure tokens (without BOS/EOS) to insert.

    Returns:
        Assembled token tensor with placeholders replaced.
    """
    if len(image_positions) != len(image_token_chunks):
        raise ValueError(
            f"Number of image positions ({len(image_positions)}) must match "
            f"number of image tokens ({len(image_token_chunks)})"
        )
    if not image_positions:
        return text_tokens

    parts = []
    last_pos = 0
    for i, pos in enumerate(image_positions):
        if pos > last_pos:
            parts.append(text_tokens[last_pos:pos])
        parts.append(image_token_chunks[i])
        last_pos = pos + 1

    if last_pos < len(text_tokens):
        parts.append(text_tokens[last_pos:])

    return torch.cat(parts, dim=0) if parts else torch.tensor(
        [], dtype=text_tokens.dtype, device=text_tokens.device,
    )


# ---------------------------------------------------------------------------
# Mode-specific assembly helpers for image2text / text2image
# ---------------------------------------------------------------------------

def assemble_image2text(
    *,
    bos_id: int,
    eos_id: int,
    image_structures: List[torch.Tensor],
    text_tokens: torch.Tensor,
) -> torch.Tensor:
    """Assemble: [BOS] + image_structs... + text + [EOS]."""
    parts = [torch.tensor([bos_id], dtype=torch.long)]
    parts.extend(image_structures)
    parts.append(text_tokens)
    parts.append(torch.tensor([eos_id], dtype=torch.long))
    return torch.cat(parts)


def assemble_text2image(
    *,
    bos_id: int,
    eos_id: int,
    text_tokens: torch.Tensor,
    image_structures: List[torch.Tensor],
) -> torch.Tensor:
    """Assemble: [BOS] + text + image_structs... + [EOS]."""
    parts = [torch.tensor([bos_id], dtype=torch.long)]
    parts.append(text_tokens)
    parts.extend(image_structures)
    parts.append(torch.tensor([eos_id], dtype=torch.long))
    return torch.cat(parts)
