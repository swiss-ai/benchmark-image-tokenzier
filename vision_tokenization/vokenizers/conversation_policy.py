"""Config-driven conversation normalization for SFT data.

Auto-detects dataset format (role/content, from/value, user/assistant pairs)
and normalizes to standard role/content messages. Optionally prepends an image
placeholder and/or an empty system message.

Usage with Hydra::

    # In your yaml config:
    conversation_policy:
      add_image_placeholder: true
      add_system_message: true

    # In code:
    policy = ConversationPolicy(**cfg.conversation_policy)
    messages = apply_conversation_policy(raw_messages, policy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationPolicy:
    """All conversation normalization config in one place.

    Attributes:
        role_map: Maps non-standard role names to standard ones.
        add_system_message: If True, prepend empty system message when absent.
        add_image_placeholder: If True, prepend image_placeholder to first user message.
        image_placeholder: Token string for the image placeholder.
    """

    role_map: dict[str, str] = field(
        default_factory=lambda: {"human": "user", "gpt": "assistant", "system": "system"}
    )
    add_system_message: bool = False
    add_image_placeholder: bool = False
    image_placeholder: str = "<image>"


def apply_conversation_policy(
    raw_messages: list[dict[str, Any]],
    policy: ConversationPolicy | None = None,
) -> list[dict[str, Any]]:
    """Normalize raw dataset messages and apply policy.

    Auto-detects the conversation format from the first message:
        - ``{"role": ..., "content": ...}`` → use directly
        - ``{"from": ..., "value": ...}`` → map roles via role_map
        - ``{"user": ..., "assistant": ...}`` → expand pairs

    Then optionally adds system message and image placeholder.
    """
    if policy is None:
        policy = ConversationPolicy()

    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError(f"Expected non-empty list of messages, got {type(raw_messages)}")

    messages = _normalize(raw_messages, policy.role_map)

    if policy.add_system_message and messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})

    if policy.add_image_placeholder:
        _prepend_image(messages, policy.image_placeholder)

    return messages


# ── Internals ───────────────────────────────────────────────────────────────


def _normalize(
    raw_messages: list[dict[str, Any]],
    role_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Auto-detect format and normalize to role/content."""
    first = raw_messages[0]
    if not isinstance(first, dict):
        raise ValueError(f"Expected dict at index 0, got {type(first)}")

    # Detect format from first message
    if "user" in first or "assistant" in first:
        return _from_pairs(raw_messages)
    if _non_null(first, "role", "content"):
        return _from_fields(raw_messages, "role", "content", role_map)
    if _non_null(first, "from", "value"):
        return _from_fields(raw_messages, "from", "value", role_map)

    raise ValueError(
        f"Cannot detect conversation format. First message keys: {list(first.keys())}. "
        f"Expected role/content, from/value, or user/assistant pairs."
    )


def _non_null(msg: dict, key_a: str, key_b: str) -> bool:
    return msg.get(key_a) is not None and msg.get(key_b) is not None


def _from_fields(
    messages: list[dict[str, Any]],
    role_key: str,
    content_key: str,
    role_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Normalize messages that use role_key/content_key fields."""
    result = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"Expected dict at index {i}, got {type(msg)}")
        role = msg.get(role_key)
        content = msg.get(content_key)
        if role is None or content is None:
            raise ValueError(
                f"Message at index {i}: missing '{role_key}' or '{content_key}'. "
                f"Got keys: {list(msg.keys())}"
            )
        role = role_map.get(str(role), str(role))
        result.append({"role": role, "content": content if isinstance(content, (str, list, dict)) else str(content)})
    return result


def _from_pairs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize {user: ..., assistant: ...} pair records."""
    result = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"Expected dict at index {i}, got {type(msg)}")
        for role in ("user", "assistant"):
            val = msg.get(role)
            if val is not None:
                result.append({"role": role, "content": val if isinstance(val, (str, list, dict)) else str(val)})
    return result


def _prepend_image(messages: list[dict[str, Any]], placeholder: str) -> None:
    """Prepend image placeholder to the first user message, in place."""
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            content = msg["content"]
            if isinstance(content, str):
                if not content.startswith(placeholder):
                    msg["content"] = f"{placeholder}\n{content}" if content else placeholder
            elif isinstance(content, list):
                # Structured content (e.g. [{"type": "image"}, {"type": "text", ...}])
                # Check if placeholder text is already present
                has_image = any(
                    isinstance(c, dict) and c.get("type") == "image" for c in content
                )
                if not has_image:
                    content.insert(0, {"type": "text", "text": placeholder})
            # dict or other types: leave as-is (placeholder not applicable)
            return
