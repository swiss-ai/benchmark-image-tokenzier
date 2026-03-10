"""
Base class for benchmark metrics.

All metrics inherit from BaseMetric and implement the __call__ method
to compute metrics from input data.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Set


class BaseMetric(ABC):
    """
    Abstract base class for benchmark metrics.

    Metrics compute quality scores from input data (images, text, etc.)
    and return flat key-value pairs that are merged into the result's
    metrics dictionary.

    Class Attributes:
        name: Unique identifier for this metric
        REQUIRED_KEYS: Set of keys that must be present in input data
        OPTIONAL_KEYS: Dict of optional keys with their default values

    Example:
        @register_metric("completion_quality")
        class CompletionQualityMetric(BaseMetric):
            name = "completion_quality"
            REQUIRED_KEYS = {"completed_image", "reference_image", "boundary_y"}
            OPTIONAL_KEYS = {"device": "cpu"}

            def __call__(self, data: dict) -> dict:
                # Compute metrics and return flat dict
                return {"psnr_full": 25.3, "ssim_full": 0.92, ...}
    """

    name: ClassVar[str] = ""
    REQUIRED_KEYS: ClassVar[Set[str]] = set()
    OPTIONAL_KEYS: ClassVar[Dict[str, Any]] = {}

    def __init__(self, device: str = "cpu", **kwargs):
        """
        Initialize the metric.

        Args:
            device: Device for computation (e.g., "cpu", "cuda:0")
            **kwargs: Additional arguments for subclass _setup()
        """
        self.device = device
        self._setup(**kwargs)

    def _setup(self, **kwargs):
        """
        Optional setup hook for subclasses to load models, etc.

        Override this method to perform initialization that requires
        the constructor arguments (e.g., loading LPIPS or CLIP models).
        """
        pass

    def validate_input(self, data: dict) -> dict:
        """
        Validate input data and fill in optional keys with defaults.

        Args:
            data: Input data dictionary

        Returns:
            Data dictionary with optional keys filled in

        Raises:
            ValueError: If required keys are missing
        """
        missing = self.REQUIRED_KEYS - set(data.keys())
        if missing:
            raise ValueError(f"Metric '{self.name}' missing required keys: {missing}")

        # Fill in optional keys with defaults
        result = dict(data)
        for key, default in self.OPTIONAL_KEYS.items():
            if key not in result:
                result[key] = default

        return result

    @abstractmethod
    def __call__(self, data: dict) -> dict:
        """
        Compute metrics from input data.

        Args:
            data: Dictionary containing input data. Required keys are
                  defined by REQUIRED_KEYS class attribute.

        Returns:
            Flat dictionary of metric key-value pairs.
            Keys should be descriptive (e.g., "psnr_full", "clip_score").
            Values should be numeric (float or int) or bool.

        Example:
            >>> metric = CompletionQualityMetric(device="cuda")
            >>> result = metric({
            ...     "completed_image": completed_pil,
            ...     "reference_image": reference_pil,
            ...     "boundary_y": 128,
            ...     "expected_generated_height": 256,
            ...     "actual_generated_height": 256,
            ... })
            >>> result
            {"psnr_full": 25.3, "ssim_full": 0.92, "lpips_full": 0.08, ...}
        """
        raise NotImplementedError
