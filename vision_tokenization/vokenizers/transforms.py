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

        # Multiple text keys (tries each, skips if key is missing or value is null)
        "transform_params": {
            "remove_string": {
                "strings": ["<unwanted>"],
                "text_key": ["content", "value"]
            }
        }
    """

    name = "remove_string"

    def __init__(self, strings: Union[str, List[str]], text_key: Optional[Union[str, List[str]]] = None):
        """
        Args:
            strings: String or list of strings to remove from text
            text_key: Key path(s) for extracting text from objects (supports dot notation, e.g., "msg.content").
                     Can be a single key or a list of keys. When multiple keys are provided, each is tried
                     in order; missing keys or null values are skipped without error.
                     Only used when __call__ receives a list of objects.
        """
        if isinstance(strings, str):
            self.strings = [strings]
        else:
            self.strings = list(strings)

        if not self.strings or not any(s for s in self.strings):
            raise ValueError("remove_string transform requires at least one non-empty string")

        # Normalize text_key to a list
        if text_key is None:
            self.text_keys = []
        elif isinstance(text_key, str):
            self.text_keys = [text_key]
        elif isinstance(text_key, list):
            self.text_keys = list(text_key)
        else:
            raise ValueError(f"text_key must be a string, list of strings, or None, got {type(text_key)}")

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
                if not self.text_keys:
                    raise ValueError("text_key must be configured to process list of objects")

                result = []
                for obj in text:
                    if not isinstance(obj, dict):
                        raise ValueError(f"Expected dict, got {type(obj)}")
                    # Create a deep copy to avoid mutating input (handles nested dicts)
                    obj_copy = copy.deepcopy(obj)
                    for key in self.text_keys:
                        try:
                            _transform_nested_value(obj_copy, key, self._remove_strings)
                        except (KeyError, TypeError, AttributeError):
                            # Key not present, intermediate path invalid, or value is None — skip
                            continue
                    result.append(obj_copy)
                return result

        raise ValueError(f"Unsupported text type: {type(text)}")


@TransformRegistry.register_image("random_augment")
class RandomAugment(ImageTransform):
    """
    Apply albumentations augmentations to images with configurable probability.

    Supports both global probability (apply entire pipeline or skip) and
    per-augmentation probability (configured within each augmentation).

    Config example (inline):
        "transform_params": {
            "random_augment": {
                "probability": 0.7,
                "transforms": [
                    {
                        "__class_fullname__": "HorizontalFlip",
                        "p": 0.5
                    },
                    {
                        "__class_fullname__": "Rotate",
                        "limit": 45,
                        "p": 0.3
                    },
                    {
                        "__class_fullname__": "ColorJitter",
                        "brightness": 0.2,
                        "contrast": 0.2,
                        "p": 0.5
                    }
                ]
            }
        }

    External config file (recommended for complex augmentations):
        "transform_params": {
            "random_augment": {
                "config_path": "examples/augmentation_config.json"
            }
        }

        # In examples/augmentation_config.json:
        {
            "random_augment": {
                "probability": 1.0,
                "transforms": [...]
            }
        }

        See vision_tokenization/examples/augmentation_config.json for a complete
        example generated from test_augmentations.ipynb notebook.
    """

    name = "random_augment"

    def __init__(
        self, probability: float = 1.0, transforms: List[Dict[str, Any]] = None, config_path: Optional[str] = None
    ):
        """
        Initialize random augmentation transform.

        Args:
            probability: Global probability of applying the entire pipeline (0.0-1.0)
                        1.0 means always apply, 0.0 means never apply
            transforms: List of albumentations transform configurations
                       Each transform should be a dict with "__class_fullname__" key
                       and any transform-specific parameters
            config_path: Path to external augmentation config JSON file (optional)
                        If provided, loads probability and transforms from file.
                        Takes precedence over inline probability/transforms parameters.

        Raises:
            ImportError: If albumentations is not installed
            ValueError: If probability is not in [0, 1] or transforms is empty
            FileNotFoundError: If config_path provided but file doesn't exist
        """
        import json
        from pathlib import Path

        # Load from external config file if path provided
        if config_path is not None:
            path = Path(config_path)
            if not path.is_absolute():
                path = Path.cwd() / path

            if not path.exists():
                raise FileNotFoundError(
                    f"Augmentation config file not found: {path}\n"
                    f"  Config path provided: {config_path}\n"
                    f"  Resolved to: {path}"
                )

            try:
                with open(path, "r") as f:
                    aug_config = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Malformed JSON in augmentation config: {path}\n"
                    f"  Error at line {e.lineno}, column {e.colno}: {e.msg}"
                )

            # Validate structure
            if "random_augment" not in aug_config:
                raise ValueError(
                    f"Invalid augmentation config structure: {path}\n"
                    f"  Expected top-level key 'random_augment'\n"
                    f"  Found keys: {list(aug_config.keys())}"
                )

            random_augment_config = aug_config["random_augment"]
            required_keys = ["probability", "transforms"]
            missing_keys = [k for k in required_keys if k not in random_augment_config]

            if missing_keys:
                raise ValueError(
                    f"Invalid random_augment structure in: {path}\n"
                    f"  Missing required keys: {missing_keys}\n"
                    f"  Found keys: {list(random_augment_config.keys())}"
                )

            # Extract from config file (overrides inline params)
            probability = random_augment_config["probability"]
            transforms = random_augment_config["transforms"]

            logger.info(f"✓ Loaded augmentation config from: {path}")

        # Validate inputs
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability}")

        if transforms is None or len(transforms) == 0:
            raise ValueError("transforms must be a non-empty list")

        self.probability = probability

        # Import albumentations (lazy import to avoid requiring it if not used)
        try:
            import albumentations as A
        except ImportError:
            raise ImportError(
                "albumentations is required for random_augment transform. " "Install with: pip install albumentations"
            )

        # Build albumentations pipeline from transform configs
        # Use the configured probability in Compose to control pipeline application
        pipeline_dict = {
            "transform": {
                "__class_fullname__": "Compose",
                "p": float(self.probability),  # Use configured probability
                "transforms": transforms,
            }
        }

        try:
            self.pipeline = A.from_dict(pipeline_dict)
        except Exception as e:
            raise ValueError(
                f"Failed to create albumentations pipeline from config: {e}\n" f"Transforms config: {transforms}"
            )

        # Build detailed pipeline information for logging
        transform_details = []
        for t in transforms:
            name = t.get("__class_fullname__", "Unknown")
            prob = t.get("p", 1.0)
            # Collect other params (excluding __class_fullname__ and p)
            other_params = {k: v for k, v in t.items() if k not in ["__class_fullname__", "p"]}
            if other_params:
                params_str = ", ".join(f"{k}={v}" for k, v in other_params.items())
                transform_details.append(f"{name}(p={prob}, {params_str})")
            else:
                transform_details.append(f"{name}(p={prob})")

        logger.info(
            f"RandomAugment initialized:\n"
            f"  Global probability: {probability}\n"
            f"  Number of augmentations: {len(transforms)}\n"
            f"  Pipeline: {' -> '.join(transform_details)}"
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply augmentations to image with configured probability.

        Args:
            image: Input PIL Image

        Returns:
            Augmented PIL Image (or original if probability check fails)

        Raises:
            TransformError: If augmentation fails
        """
        import numpy as np

        try:
            # Convert PIL Image to numpy array for albumentations
            img_np = np.array(image)

            # Apply augmentation pipeline (probability handled by albumentations Compose)
            augmented = self.pipeline(image=img_np)

            # Convert back to PIL Image
            result = Image.fromarray(augmented["image"])

            # Ensure mode matches original (albumentations might change it)
            if result.mode != image.mode and image.mode in ["RGB", "L", "RGBA"]:
                result = result.convert(image.mode)

            return result

        except Exception as e:
            raise TransformError(f"RandomAugment failed: {e}")


@TransformRegistry.register_image("rescale")
class RescaleTransform(ImageTransform):
    """
    Rescale image dimensions by a multiplier factor.

    Config example:
        "transform_params": {
            "rescale": {"multiplier": 0.5}  # Scale to 50% of original size
        }
    """

    name = "rescale"

    def __init__(self, multiplier: float = 1.0):
        """
        Args:
            multiplier: Scale factor for width and height. Default 1.0 (no change).
                       Values < 1.0 shrink, values > 1.0 enlarge.
        """
        if multiplier <= 0:
            raise ValueError(f"multiplier must be positive, got {multiplier}")
        self.multiplier = multiplier

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.multiplier == 1.0:
            return image

        new_width = int(image.width * self.multiplier)
        new_height = int(image.height * self.multiplier)

        if new_width < 1 or new_height < 1:
            raise TransformError(f"Rescale resulted in invalid dimensions: {new_width}x{new_height}")

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
