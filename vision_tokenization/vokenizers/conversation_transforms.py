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
    The LLaVa Instruct data is quite simple in structure so we only include necessary transforms.
    Llava-OneVision-Instruct: https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data
    Apertus: https://huggingface.co/collections/swiss-ai/apertus-llm

    Input format (LLaVa Onevision Instruct):
        [
            {"role": "user", "content": "question1"}, # here we add the image additionally
            {"role": "assistant", "content": "answer1"},
            ...
        ]

    Output format (Apertus):
        [
            {
                "role": "user",
                "content": {
                    "parts": [
                        {
                            "type": "image"
                        },
                        {
                            "type": "text",
                            "text": "question1"
                        }
                    ]
                }
            },
            {"role": "assistant", "content": "answer1"},
        ]

    The first user message we include message part, so the chat template adds the image placeholder we later replace with
    actual vision tokens during tokenization.
    Also Configure for empty system prompt! (Otherwise cutoff date and so on included.)
    Deliberation and Tool Capabilities are set false by default.
    """

    name = "llava_to_apertus"

    def transform(self, text: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Expect a list of dicts.
        Here we set the 1st msg to always contain the input image.
        """
        if not isinstance(text, list):
            raise ValueError(f"Expected list of conversation dicts, got {type(text)}")

        if not text:
            raise ValueError("Conversation list cannot be empty")

        messages = []

        # Add empty system prompt
        messages.append({"role": "system", "content": ""})

        for i, conv in enumerate(text):
            if not isinstance(conv, dict):
                raise ValueError(f"Expected dict at index {i}, got {type(conv)}")

            if "role" not in conv or "content" not in conv:
                raise ValueError(
                    f"Conversation dict at index {i} must contain 'role' and 'content' keys, "
                    f"got keys: {list(conv.keys())}"
                )

            # First message gets the image placeholder and question for user
            if i == 0:
                assert conv["role"] == "user"
                messages.append(
                    {
                        "role": "user",
                        "content": {"parts": [{"type": "image"}, {"type": "text", "text": conv["content"]}]},
                    }
                )
            else:
                # Normal messages
                messages.append({"role": conv["role"], "content": conv["content"]})

        return messages
