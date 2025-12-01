#!/usr/bin/env python3
"""
Transform system for vision tokenization pipeline.

Provides base classes and registry for image and text transforms that can be
chained together and applied during data processing.

Usage:
    # Register a custom transform
    @TransformRegistry.register_image("convert_rgb")
    class ConvertRGB(ImageTransform):
        name = "convert_rgb"

        def __call__(self, image: Image.Image) -> Image.Image:
            return image.convert("RGB")

    # In config JSON:
    {
        "image_transforms": "convert_rgb,resize_max",
        "text_transforms": "strip_whitespace",
        "transform_params": {
            "resize_max": {"max_size": 1024}
        }
    }
"""

import copy
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from PIL import Image

logger = logging.getLogger(__name__)


class TransformError(Exception):
    """Raised when a transform fails. Sample will be skipped."""

    pass


class ImageTransform(ABC):
    """
    Base class for image transforms.

    Subclasses must implement:
        - name: class attribute with transform name for registry
        - __call__: transform logic operating on PIL Image
    """

    name: str = ""

    @abstractmethod
    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply transform to PIL Image.

        Args:
            image: Input PIL Image

        Returns:
            Transformed PIL Image

        Raises:
            TransformError: If transform fails
        """
        pass


class TextTransform(ABC):
    """
    Base class for text transforms.

    Subclasses must implement:
        - name: class attribute with transform name for registry
        - __call__: transform logic operating on string
    """

    name: str = ""

    @abstractmethod
    def __call__(self, text: str) -> str:
        """
        Apply transform to text string.

        Args:
            text: Input text string

        Returns:
            Transformed text string

        Raises:
            TransformError: If transform fails
        """
        pass


class TransformRegistry:
    """
    Registry for transform classes.

    Use decorators to register transforms:
        @TransformRegistry.register_image("my_transform")
        class MyTransform(ImageTransform):
            ...
    """

    _image_transforms: Dict[str, Type[ImageTransform]] = {}
    _text_transforms: Dict[str, Type[TextTransform]] = {}

    @classmethod
    def register_image(cls, name: str):
        """
        Decorator to register an image transform.

        Args:
            name: Unique name for the transform
        """

        def decorator(transform_cls: Type[ImageTransform]) -> Type[ImageTransform]:
            if name in cls._image_transforms:
                logger.warning(f"Overwriting existing image transform: {name}")
            cls._image_transforms[name] = transform_cls
            return transform_cls

        return decorator

    @classmethod
    def register_text(cls, name: str):
        """
        Decorator to register a text transform.

        Args:
            name: Unique name for the transform
        """

        def decorator(transform_cls: Type[TextTransform]) -> Type[TextTransform]:
            if name in cls._text_transforms:
                logger.warning(f"Overwriting existing text transform: {name}")
            cls._text_transforms[name] = transform_cls
            return transform_cls

        return decorator

    @classmethod
    def get_image_transform(cls, name: str) -> Type[ImageTransform]:
        """
        Get an image transform class by name.

        Args:
            name: Transform name

        Returns:
            Transform class

        Raises:
            ValueError: If transform not found
        """
        if name not in cls._image_transforms:
            available = list(cls._image_transforms.keys())
            raise ValueError(f"Unknown image transform: '{name}'. " f"Available: {available}")
        return cls._image_transforms[name]

    @classmethod
    def get_text_transform(cls, name: str) -> Type[TextTransform]:
        """
        Get a text transform class by name.

        Args:
            name: Transform name

        Returns:
            Transform class

        Raises:
            ValueError: If transform not found
        """
        if name not in cls._text_transforms:
            available = list(cls._text_transforms.keys())
            raise ValueError(f"Unknown text transform: '{name}'. " f"Available: {available}")
        return cls._text_transforms[name]

    @classmethod
    def list_image_transforms(cls) -> List[str]:
        """List all registered image transform names."""
        return list(cls._image_transforms.keys())

    @classmethod
    def list_text_transforms(cls) -> List[str]:
        """List all registered text transform names."""
        return list(cls._text_transforms.keys())


class TransformPipeline:
    """
    Chained transform pipeline for image and text.

    Applies a sequence of transforms to image and text data.
    """

    def __init__(
        self,
        image_transforms: Optional[List[str]] = None,
        text_transforms: Optional[List[str]] = None,
        transform_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize transform pipeline.

        Args:
            image_transforms: List of image transform names to apply in order
            text_transforms: List of text transform names to apply in order
            transform_params: Dict mapping transform names to their parameters
        """
        self.transform_params = transform_params or {}

        # Build image transform chain
        self.image_chain: List[ImageTransform] = []
        if image_transforms:
            for name in image_transforms:
                params = self.transform_params.get(name, {})
                transform_cls = TransformRegistry.get_image_transform(name)
                self.image_chain.append(transform_cls(**params))
                logger.debug(f"Added image transform: {name}")

        # Build text transform chain
        self.text_chain: List[TextTransform] = []
        if text_transforms:
            for name in text_transforms:
                params = self.transform_params.get(name, {})
                transform_cls = TransformRegistry.get_text_transform(name)
                self.text_chain.append(transform_cls(**params))
                logger.debug(f"Added text transform: {name}")

        logger.info(
            f"TransformPipeline initialized with "
            f"{len(self.image_chain)} image transforms, "
            f"{len(self.text_chain)} text transforms"
        )

    def apply(self, image: Optional[Image.Image], text: Optional[str]) -> Tuple[Optional[Image.Image], Optional[str]]:
        """
        Apply transform chains to image and text.

        Args:
            image: PIL Image or None
            text: Text string or None

        Returns:
            Tuple of (transformed_image, transformed_text)

        Raises:
            TransformError: If any transform fails
        """
        # Apply image transforms
        if image is not None:
            for transform in self.image_chain:
                try:
                    image = transform(image)
                except TransformError:
                    raise
                except Exception as e:
                    raise TransformError(f"Image transform '{transform.name}' failed: {e}")

        # Apply text transforms
        if text is not None:
            for transform in self.text_chain:
                try:
                    text = transform(text)
                except TransformError:
                    raise
                except Exception as e:
                    raise TransformError(f"Text transform '{transform.name}' failed: {e}")

        return image, text

    @property
    def has_transforms(self) -> bool:
        """Check if pipeline has any transforms."""
        return bool(self.image_chain or self.text_chain)


def parse_transforms(transform_string: Optional[str]) -> List[str]:
    """
    Parse comma-separated transform names from config string.

    Args:
        transform_string: Comma-separated transform names or None

    Returns:
        List of transform names (empty if input is None or empty)
    """
    if not transform_string:
        return []
    return [t.strip() for t in transform_string.split(",") if t.strip()]


def create_transform_pipeline(
    image_transforms: Optional[str] = None,
    text_transforms: Optional[str] = None,
    transform_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[TransformPipeline]:
    """
    Create a TransformPipeline from configuration.

    Args:
        image_transforms: Comma-separated string of image transform names
        text_transforms: Comma-separated string of text transform names
        transform_params: Dict of {transform_name: {param: value}}

    Returns:
        TransformPipeline if any transforms specified, None otherwise
    """
    image_list = parse_transforms(image_transforms)
    text_list = parse_transforms(text_transforms)

    if not image_list and not text_list:
        return None

    return TransformPipeline(image_transforms=image_list, text_transforms=text_list, transform_params=transform_params)


# =============================================================================
# Built-in Transforms
# =============================================================================


def _transform_nested_value(obj: Dict[str, Any], key_path: str, fn: Callable[[str], str]) -> None:
    """Get a nested value, transform it suding the given function fn, and set it back."""
    keys = key_path.split(".")
    target = obj
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = fn(target[keys[-1]])


@TransformRegistry.register_text("remove_string")
class RemoveStringTransform(TextTransform):
    """
    Remove all occurrences of one or more strings from text.
    Supports processing single strings, lists of strings, or lists of objects.

    Config examples:
        # Single string to remove
        "transform_params": {
            "remove_string": {"strings": "<unwanted_tag>"}
        }

        # Multiple strings to remove
        "transform_params": {
            "remove_string": {"strings": ["<tag1>", "<tag2>", "unwanted"]}
        }

        # For processing list of objects with simple field
        "transform_params": {
            "remove_string": {
                "strings": ["<unwanted>"],
                "text_key": "content"
            }
        }

        # For processing list of objects with nested field
        "transform_params": {
            "remove_string": {
                "strings": ["<unwanted>"],
                "text_key": "msg.content"
            }
        }
    """

    name = "remove_string"

    def __init__(self, strings: Union[str, List[str]], text_key: Optional[str] = None):
        """
        Args:
            strings: String or list of strings to remove from text
            text_key: Key path for extracting text from objects (supports dot notation, e.g., "msg.content")
                     Only used when __call__ receives a list of objects
        """
        if isinstance(strings, str):
            self.strings = [strings]
        else:
            self.strings = list(strings)

        if not self.strings or not any(s for s in self.strings):
            raise ValueError("remove_string transform requires at least one non-empty string")

        self.text_key = text_key

    def _remove_strings(self, text: str) -> str:
        """Apply string removal to a single text string."""
        for s in self.strings:
            text = text.replace(s, "")
        return text

    def __call__(
        self, text: Union[str, List[str], List[Dict[str, Any]]]
    ) -> Union[str, List[str], List[Dict[str, Any]]]:
        """
        Apply string removal to text.

        Args:
            text: Single string, list of strings, or list of objects

        Returns:
            Same type as input with strings removed
        """
        # Single string
        if isinstance(text, str):
            return self._remove_strings(text)

        # List input
        if isinstance(text, list):
            if not text:
                return text

            # List of strings
            if isinstance(text[0], str):
                return [self._remove_strings(t) for t in text]

            # List of objects
            if isinstance(text[0], dict):
                if not self.text_key:
                    raise ValueError("text_key must be configured to process list of objects")

                result = []
                for obj in text:
                    if not isinstance(obj, dict):
                        raise ValueError(f"Expected dict, got {type(obj)}")
                    # Create a deep copy to avoid mutating input (handles nested dicts)
                    obj_copy = copy.deepcopy(obj)
                    _transform_nested_value(obj_copy, self.text_key, self._remove_strings)
                    result.append(obj_copy)
                return result

        raise ValueError(f"Unsupported text type: {type(text)}")
