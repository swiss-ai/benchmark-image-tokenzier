"""Helpers for image-map SFT rows.

Energon-style multi-image SFT stores image references inside JSON-encoded
messages and stores local pixels in an ``images`` map keyed by those references.
The helpers here keep message-order image semantics in one place for scanning,
loading, and SFT parsing.
"""

from __future__ import annotations

from typing import Any, Iterable

from vision_tokenization.utils.json import json_loads

NORMALIZED_MESSAGES_KEY = "__image_map_normalized_messages__"


def _decode_messages(value: Any) -> list[dict[str, Any]]:
    """Decode messages from JSON text or an existing list."""
    if isinstance(value, (bytearray, memoryview)):
        value = bytes(value)
    if isinstance(value, (str, bytes)):
        value = json_loads(value)
    if not isinstance(value, list) or not value:
        raise ValueError("Image-map SFT row requires a non-empty messages list")
    return value


def _content_parts(content: Any) -> Iterable[Any]:
    if isinstance(content, list):
        return content
    return [content]


def _image_ref_from_part(part: dict[str, Any]) -> str:
    """Extract an image ref from an image content part."""
    ref = part.get("image") or part.get("path") or part.get("url")
    if ref is None:
        raise ValueError("Image content part is missing image/path/url")
    ref = str(ref)
    if not ref:
        raise ValueError("Image content part has an empty image ref")
    return ref


def extract_image_refs(messages_value: Any) -> list[str]:
    """Return image refs in the exact order they appear in messages."""
    refs: list[str] = []
    for idx, message in enumerate(_decode_messages(messages_value)):
        if not isinstance(message, dict):
            raise ValueError(f"Image-map SFT message at index {idx} is not a dict")
        for part in _content_parts(message.get("content")):
            if isinstance(part, dict) and part.get("type") == "image":
                refs.append(_image_ref_from_part(part))
    return refs


def _chat_template_content(parts: list[dict[str, Any]]) -> str | dict[str, list[dict[str, Any]]]:
    if not parts:
        return ""
    if all(part.get("type") == "text" for part in parts):
        return "".join(str(part.get("text") or "") for part in parts)
    return {"parts": parts}


def _convert_part(part: Any, refs: list[str]) -> dict[str, str] | None:
    """Convert one content part to normalized form; appends to refs in place."""
    if isinstance(part, dict):
        part_type = part.get("type")
        if part_type == "image":
            refs.append(_image_ref_from_part(part))
            return {"type": "image"}
        if part_type == "text":
            return {"type": "text", "text": str(part.get("text") or "")}
        return None
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if part is not None:
        return {"type": "text", "text": str(part)}
    return None


def normalize_messages_and_refs(messages_value: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize image-map messages and extract image refs in one pass."""
    normalized: list[dict[str, Any]] = []
    refs: list[str] = []
    for idx, message in enumerate(_decode_messages(messages_value)):
        if not isinstance(message, dict):
            raise ValueError(f"Image-map SFT message at index {idx} is not a dict")

        parts: list[dict[str, str]] = []
        for part in _content_parts(message.get("content")):
            converted = _convert_part(part, refs)
            if converted is not None:
                parts.append(converted)

        normalized.append({
            "role": str(message.get("role") or ""),
            "content": _chat_template_content(parts),
        })
    return normalized, refs


def normalize_messages(messages_value: Any) -> list[dict[str, Any]]:
    """Normalize JSON messages to canonical structured SFT messages."""
    normalized, _refs = normalize_messages_and_refs(messages_value)
    return normalized


def _bytes_from_image_value(value: Any) -> bytes | None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, dict):
        value_bytes = value.get("bytes")
        if isinstance(value_bytes, (bytes, bytearray, memoryview)):
            return bytes(value_bytes)
    return None


def _normalize_image_cell(value: Any) -> Any | None:
    value_bytes = _bytes_from_image_value(value)
    if value_bytes is not None:
        if isinstance(value, dict):
            cell = dict(value)
            cell["bytes"] = value_bytes
            return cell
        return value_bytes
    if isinstance(value, dict) and value.get("path") is not None:
        return dict(value)
    return None


def image_map_lookup(raw_images: Any) -> dict[str, Any]:
    """Return image cells keyed by the refs used in messages.

    Normalizes each cell so callers receive uniformly-shaped bytes / struct
    values. Use this on the decode path. Scanners that only need image
    dimensions should use :func:`image_map_as_dict` instead — it skips the
    byte-copy that normalization performs.
    """
    if raw_images is None:
        return {}

    if isinstance(raw_images, dict):
        result: dict[str, Any] = {}
        for key, value in raw_images.items():
            cell = _normalize_image_cell(value)
            if cell is not None:
                result[str(key)] = cell
        return result

    if isinstance(raw_images, list):
        result: dict[str, Any] = {}
        for item in raw_images:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                cell = _normalize_image_cell(item[1])
                if cell is not None:
                    result[str(item[0])] = cell
            else:
                raise ValueError("Image-map entry must be a key/value pair")
        return result

    raise TypeError(f"Unsupported image-map container: {type(raw_images).__name__}")


def image_map_as_dict(raw_images: Any) -> dict[str, Any]:
    """Lightweight ref → raw-value lookup; preserves the original cell shape.

    ``image_map_lookup`` copies bytes through ``_normalize_image_cell``; this
    helper skips that work. Use it when the caller only reads image headers
    (e.g. dimensions) and doesn't keep the cells around.
    """
    if raw_images is None:
        return {}
    if isinstance(raw_images, list):
        return {str(k): v for k, v in raw_images}
    if isinstance(raw_images, dict):
        return {str(k): v for k, v in raw_images.items()}
    raise TypeError(f"Unsupported image-map container: {type(raw_images).__name__}")
