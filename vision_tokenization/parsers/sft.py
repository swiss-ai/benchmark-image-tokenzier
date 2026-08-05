"""Schema-driven parsers for raw SFT rows.

Returns canonical ``[{"role": str, "content": str | list}]`` messages suitable
for ``apply_conversation_policy``.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from vision_tokenization.utils.image_map_sft import (
    NORMALIZED_MESSAGES_KEY,
    normalize_messages,
)
from vision_tokenization.utils.json import json_loads

IMAGE_MARKERS = ("<image>", "<|image|>")


def parse_messages(
    row: dict[str, Any],
    *,
    parser: str,
    num_images: Optional[int] = None,
    parser_args: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Convert one raw row into canonical SFT messages."""
    fn = _PARSERS.get(parser)
    if fn is None:
        raise ValueError(
            f"Unknown SFT parser: {parser!r}. Available: {sorted(_PARSERS)}"
        )
    return fn(row, num_images=num_images, parser_args=parser_args or {})

def _extract_messages_value(
    row: dict[str, Any],
    parser_args: dict[str, Any],
    *,
    default_keys: tuple[str, ...],
    parser_name: str,
    allow_single_column: bool = False,
) -> Any:
    """Look up the messages column from a row, honoring parser_args overrides."""
    column = parser_args.get("conversation_column")
    if column is not None:
        value = row.get(str(column))
    else:
        value = _first_present(row, *default_keys)
        if value is None and allow_single_column and len(row) == 1:
            value = next(iter(row.values()))
    if value is None:
        if column is not None:
            raise ValueError(
                f"{parser_name} parser requires column {column!r}"
            )
        raise ValueError(
            f"{parser_name} parser requires one of: {', '.join(default_keys)}, "
            "or parser_args.conversation_column"
        )
    return value


def _parse_conversation(
    row: dict[str, Any],
    *,
    num_images: Optional[int] = None,
    parser_args: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    parser_args = parser_args or {}
    messages = _extract_messages_value(
        row,
        parser_args,
        default_keys=("messages", "conversations", "conversation"),
        parser_name="conversation",
        allow_single_column=True,
    )
    if isinstance(messages, str):
        try:
            messages = json_loads(messages)
        except json.JSONDecodeError as exc:
            raise ValueError("conversation field must be JSON-decodable when stored as str") from exc
    if not isinstance(messages, list) or not messages:
        raise ValueError("conversation parser expects a non-empty message list")
    return messages


def _parse_image_map_conversation(
    row: dict[str, Any],
    *,
    num_images: Optional[int] = None,
    parser_args: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    parser_args = parser_args or {}
    normalized_messages = row.get(NORMALIZED_MESSAGES_KEY)
    if normalized_messages is not None:
        return normalized_messages

    messages = _extract_messages_value(
        row,
        parser_args,
        default_keys=("messages",),
        parser_name="image_map_conversation",
    )
    return normalize_messages(messages)


def _parse_qa(
    row: dict[str, Any],
    *,
    num_images: Optional[int],
    parser_args: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    parser_args = parser_args or {}
    prompt_col = parser_args.get("prompt_column")
    answer_col = parser_args.get("answer_column")
    prompt = row.get(prompt_col) if prompt_col else _first_present(row, "question", "query", "input", "prompt")
    answer = row.get(answer_col) if answer_col else _first_present(row, "answer", "output", "response")
    if prompt is None or answer is None:
        raise ValueError("qa parser requires a prompt/question field and an answer field")

    prompt = str(prompt)
    answer = str(answer)
    prompt = _ensure_image_placeholders(prompt, num_images or 1)
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]


def _parse_molmo_multi_image_qa(
    row: dict[str, Any],
    *,
    num_images: Optional[int],
    parser_args: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    qa_pairs = row.get("qa_pairs")
    if not isinstance(qa_pairs, dict):
        raise ValueError("molmo_multi_image_qa parser requires a qa_pairs dict")

    questions = qa_pairs.get("question")
    answers = qa_pairs.get("answer")
    if not isinstance(questions, list) or not isinstance(answers, list):
        raise ValueError("qa_pairs.question and qa_pairs.answer must both be lists")
    if not questions or not answers:
        raise ValueError("qa_pairs.question and qa_pairs.answer cannot be empty")
    if len(questions) != len(answers):
        raise ValueError(
            f"qa_pairs length mismatch: {len(questions)} questions vs {len(answers)} answers"
        )

    if not num_images or num_images <= 0:
        raise ValueError(
            "molmo_multi_image_qa requires num_images>0 from the loader manifest"
        )

    messages: list[dict[str, Any]] = []
    for idx, (question, answer) in enumerate(zip(questions, answers)):
        prompt = str(question)
        if idx == 0:
            prompt = _ensure_image_placeholders(prompt, num_images)
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": str(answer)})

    return messages


def _parse_user_assistant_pairs(
    row: dict[str, Any],
    *,
    num_images: Optional[int] = None,
    parser_args: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Input shape: list[{user, assistant}] turn pairs (one per turn).

    Flattens into the canonical [{role, content}, ...] message list. Adds
    `<image>` placeholders on the first user turn if absent and num_images > 0.

    Used by any dataset that stores conversations as packed turn pairs rather
    than alternating role messages — FineVision (`texts`), some image-QA dumps,
    etc.
    """
    parser_args = parser_args or {}
    turns = _extract_messages_value(
        row,
        parser_args,
        default_keys=("texts",),
        parser_name="user_assistant_pairs",
    )
    if not isinstance(turns, list) or not turns:
        raise ValueError("user_assistant_pairs parser expects a non-empty list of {user, assistant} pairs")
    messages: list[dict[str, Any]] = []
    for idx, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"user_assistant_pairs turn {idx} is not a dict: {type(turn).__name__}")
        user = turn.get("user")
        assistant = turn.get("assistant")
        if user is None or assistant is None:
            raise ValueError(f"user_assistant_pairs turn {idx} missing user or assistant key")
        if idx == 0:
            user = _ensure_image_placeholders(str(user), num_images or 0)
        messages.append({"role": "user", "content": str(user)})
        messages.append({"role": "assistant", "content": str(assistant)})
    return messages


_PARSERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "conversation": _parse_conversation,
    "image_map_conversation": _parse_image_map_conversation,
    "user_assistant_pairs": _parse_user_assistant_pairs,
    "qa": _parse_qa,
    "molmo_multi_image_qa": _parse_molmo_multi_image_qa,
}


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _count_image_markers(text: str) -> int:
    return sum(text.count(marker) for marker in IMAGE_MARKERS)


def _ensure_image_placeholders(text: str, num_images: int) -> str:
    if num_images <= 0:
        return text

    marker_count = _count_image_markers(text)
    if marker_count == num_images:
        return text
    if marker_count != 0:
        raise ValueError(
            f"Prompt contains {marker_count} image marker(s), expected {num_images}"
        )

    prefix = "\n".join("<image>" for _ in range(num_images))
    return f"{prefix}\n{text}" if text else prefix
