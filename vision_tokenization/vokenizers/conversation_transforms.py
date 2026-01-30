#!/usr/bin/env python3
"""
Conversation transformation system for SFT tokenization.

Provides base classes and registry for converting dataset-specific conversation
formats to the standard chat template format expected by the tokenizer.

If you are tokenizing a new dataset, there is a chance you might need to add a new conversation transform.

Usage:
    # Register a custom transform
    @ConversationTransformRegistry.register("my_format")
    class MyFormatTransform(BaseConversationTransform):
        name = "my_format"

        def transform(self, text):
            # Convert dataset format to messages
            return messages

    # In config JSON:
    {
        "conversation_transform": "my_format"
    }
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

logger = logging.getLogger(__name__)


class BaseConversationTransform(ABC):
    """
    Base class for conversation format transforms.

    Subclasses must implement:
        - name: class attribute with transform name for registry
        - transform: method to convert dataset format to chat messages as expected by tokenizer you aim to use
    """

    name: str = ""

    @abstractmethod
    def transform(self, text: Any) -> List[Dict[str, Any]]:
        """
        Transform dataset conversation format to standard chat messages.

        Args:
            text: Dataset-specific conversation format (structure varies by dataset)

        Returns:
            List of message dicts with 'role' and 'content' keys,
            formatted for apply_chat_template()

        Raises:
            ValueError: If conversion fails
        """
        pass


class ConversationTransformRegistry:
    """
    Registry for conversation transform classes.

    Use decorator to register transforms:
        @ConversationTransformRegistry.register("my_transform")
        class MyTransform(BaseConversationTransform):
            ...
    """

    _transforms: Dict[str, Type[BaseConversationTransform]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a conversation transform.

        Args:
            name: Unique name for the transform
        """

        def decorator(transform_cls: Type[BaseConversationTransform]) -> Type[BaseConversationTransform]:
            if name in cls._transforms:
                logger.warning(f"Overwriting existing conversation transform: {name}")
            cls._transforms[name] = transform_cls
            return transform_cls

        return decorator

    @classmethod
    def get_transform(cls, name: str) -> Type[BaseConversationTransform]:
        """
        Get a conversation transform class by name.

        Args:
            name: Transform name

        Returns:
            Transform class (not instantiated)

        Raises:
            ValueError: If transform not found
        """
        if name not in cls._transforms:
            available = list(cls._transforms.keys())
            raise ValueError(f"Unknown conversation transform: '{name}'. " f"Available: {available}")
        return cls._transforms[name]

    @classmethod
    def list_transforms(cls) -> List[str]:
        """List all registered conversation transform names."""
        return list(cls._transforms.keys())


# =============================================================================
# Built-in Transforms
# =============================================================================


@ConversationTransformRegistry.register("finevision_to_llama")
class FineVisionToLLaMATransform(BaseConversationTransform):
    """
    Transform HF Finevision conversation format to LLaMA multimodal chat template format.
    Finevision: https://huggingface.co/datasets/HuggingFaceM4/FineVision
    LLaMa Multimodal: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision

    Input format (Finevision):
        [
            {"user": "question1", "assistant": "answer1"},
            {"user": "question2", "assistant": "answer2"},
            ...
        ]

    Output format (LLaMA multimodal):
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "question1"}
                ]
            },
            {"role": "assistant", "content": "answer1"},
            {"role": "user", "content": "question2"},
            {"role": "assistant", "content": "answer2"},
            ...
        ]

    The first user message includes an image placeholder that will be replaced
    with actual vision tokens during tokenization.
    """

    name = "finevision_to_llama"

    def transform(self, text: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Convert Finevision format to LLaMa multimodal messages.

        Args:
            text: List of {"user": str, "assistant": str} conversation dicts

        Returns:
            List of message dicts formatted for LLaMA chat template

        Raises:
            ValueError: If input format is invalid
        """
        if not isinstance(text, list):
            raise ValueError(f"Expected list of conversation dicts, got {type(text)}")

        if not text:
            raise ValueError("Conversation list cannot be empty")

        messages = []

        for i, conv in enumerate(text):
            if not isinstance(conv, dict):
                raise ValueError(f"Expected dict at index {i}, got {type(conv)}")

            if "user" not in conv or "assistant" not in conv:
                raise ValueError(
                    f"Conversation dict at index {i} must contain 'user' and 'assistant' keys, "
                    f"got keys: {list(conv.keys())}"
                )

            # Add user message
            if i == 0:
                # First message gets the image placeholder
                messages.append(
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": conv["user"]}]}
                )
            else:
                messages.append({"role": "user", "content": conv["user"]})

            # Add assistant message
            messages.append({"role": "assistant", "content": conv["assistant"]})

        return messages


@ConversationTransformRegistry.register("llava_to_apertus")
class LLaVaInstructToApertusTransform(BaseConversationTransform):
    """
    Transform LLaVa-Onevision-Instruct conversation format to Apertus chat template format.
    Handles all conversation formats found in the LLaVA-OneVision-1.5-Instruct-Data dataset.
    Llava-OneVision-Instruct: https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data
    Apertus: https://huggingface.co/collections/swiss-ai/apertus-llm

    Supported input formats:
        Format A: [{"role": "user", "content": "..."},  {"role": "assistant", "content": "..."}]
        Format B: [{"from": "human", "value": "..."},   {"from": "gpt", "value": "..."}]
        Format C: [{"content": null, "from": "human", "role": null, "value": "..."}]  (null role/content, data in from/value)
        Format D: [{"content": "...", "from": null, "role": "user", "value": null, ...}]  (extra null keys, data in role/content)

    Output format (Apertus):
        [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": {
                    "parts": [
                        {"type": "image"},
                        {"type": "text", "text": "question1"}
                    ]
                }
            },
            {"role": "assistant", "content": "answer1"},
        ]

    The first user message includes an image part so the chat template adds the image placeholder
    we later replace with actual vision tokens during tokenization.
    Also configures an empty system prompt (otherwise cutoff date and so on included).
    Deliberation and Tool Capabilities are set false by default.
    """

    name = "llava_to_apertus"

    # Mapping from "from" field values to standard role names
    _ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}

    def _normalize_message(self, msg: Dict[str, Any], index: int) -> tuple:
        """
        Extract (role, content) from any LLaVA OneVision conversation format.

        Priority: role/content (if both non-None) > from/value (if both non-None).

        Args:
            msg: Message dict in any supported format
            index: Message index (for error reporting)

        Returns:
            Tuple of (role, content) as strings

        Raises:
            ValueError: If neither key pair yields valid data
        """
        role = msg.get("role")
        content = msg.get("content")
        from_field = msg.get("from")
        value = msg.get("value")

        # Prefer role/content when both are non-None (Format A, D)
        if role is not None and content is not None:
            return str(role), str(content)

        # Fall back to from/value (Format B, C)
        if from_field is not None and value is not None:
            mapped_role = self._ROLE_MAP.get(from_field)
            if mapped_role is None:
                raise ValueError(
                    f"Message at index {index}: unknown 'from' value '{from_field}'. "
                    f"Expected one of: {list(self._ROLE_MAP.keys())}"
                )
            return mapped_role, str(value)

        raise ValueError(
            f"Message at index {index}: cannot extract role/content. "
            f"Expected non-null 'role'+'content' or 'from'+'value', "
            f"got: role={role}, content={content}, from={from_field}, value={value}"
        )

    def transform(self, text: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform conversation from any LLaVA OneVision format to Apertus chat template format.
        The first user message always gets the image placeholder.
        """
        if not isinstance(text, list):
            raise ValueError(f"Expected list of conversation dicts, got {type(text)}")

        if not text:
            raise ValueError("Conversation list cannot be empty")

        messages = []

        # Normalize all messages first
        normalized = []
        for i, conv in enumerate(text):
            if not isinstance(conv, dict):
                raise ValueError(f"Expected dict at index {i}, got {type(conv)}")
            role, content = self._normalize_message(conv, i)
            normalized.append((role, content))

        # Handle system message: use from data if present, otherwise add empty one
        start_idx = 0
        if normalized[0][0] == "system":
            messages.append({"role": "system", "content": normalized[0][1]})
            start_idx = 1
        else:
            messages.append({"role": "system", "content": ""})

        # Process remaining messages; first user message gets the image placeholder
        first_user_seen = False
        for role, content in normalized[start_idx:]:
            if role == "user" and not first_user_seen:
                messages.append(
                    {
                        "role": "user",
                        "content": {"parts": [{"type": "image"}, {"type": "text", "text": content}]},
                    }
                )
                first_user_seen = True
            else:
                messages.append({"role": role, "content": content})

        return messages
