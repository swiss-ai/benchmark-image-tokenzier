"""Schema-driven parsers for raw SFT rows.

Returns canonical ``[{"role": str, "content": str | list}]`` messages suitable
for ``apply_conversation_policy``.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

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

def _parse_conversation(
    row: dict[str, Any],
    *,
    num_images: Optional[int] = None,
    parser_args: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    parser_args = parser_args or {}
    conversation_column = parser_args.get("conversation_column")
    if conversation_column is not None:
        messages = row.get(str(conversation_column))
    else:
        messages = _first_present(row, "messages", "conversations", "conversation")
        if messages is None and len(row) == 1:
            messages = next(iter(row.values()))
    if messages is None:
        raise ValueError(
            "conversation parser requires one of: messages, conversations, conversation, "
            "or parser_args.conversation_column"
        )
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError as exc:
            raise ValueError("conversation field must be JSON-decodable when stored as str") from exc
    if not isinstance(messages, list) or not messages:
        raise ValueError("conversation parser expects a non-empty message list")
    return messages


def _parse_qa(
    row: dict[str, Any],
    *,
    num_images: Optional[int],
    parser_args: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    prompt = _first_present(row, "question", "query", "input", "prompt")
    answer = _first_present(row, "answer", "output", "response")
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


_PARSERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "conversation": _parse_conversation,
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

