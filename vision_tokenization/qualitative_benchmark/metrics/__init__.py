"""
Metrics module for qualitative benchmarks.

Provides a registry system for metrics that can be dynamically loaded and used
across different benchmark types.
"""

from typing import Dict, List, Type

from vision_tokenization.qualitative_benchmark.metrics.base import BaseMetric

# Global metric registry
METRIC_REGISTRY: Dict[str, Type[BaseMetric]] = {}


def register_metric(name: str):
    """
    Decorator to register a metric class.

    Usage:
        @register_metric("completion_quality")
        class CompletionQualityMetric(BaseMetric):
            ...
    """

    def decorator(cls: Type[BaseMetric]) -> Type[BaseMetric]:
        if name in METRIC_REGISTRY:
            raise ValueError(f"Metric '{name}' is already registered")
        METRIC_REGISTRY[name] = cls
        return cls

    return decorator


def get_metric(name: str, **kwargs) -> BaseMetric:
    """
    Factory function to create a metric instance by name.

    Args:
        name: Metric identifier (e.g., "completion_quality", "clip_score")
        **kwargs: Arguments passed to the metric constructor

    Returns:
        Initialized metric instance

    Raises:
        ValueError: If metric name is not found in registry
    """
    if name not in METRIC_REGISTRY:
        available = ", ".join(METRIC_REGISTRY.keys()) or "none"
        raise ValueError(f"Unknown metric '{name}'. Available metrics: {available}")

    metric_cls = METRIC_REGISTRY[name]
    return metric_cls(**kwargs)


def list_metrics() -> List[str]:
    """
    List all registered metric names.

    Returns:
        List of metric identifiers
    """
    return list(METRIC_REGISTRY.keys())


# Import metrics to trigger registration
# These imports must come after the registry is defined
from vision_tokenization.qualitative_benchmark.metrics.completion_quality import CompletionQualityMetric
from vision_tokenization.qualitative_benchmark.metrics.clip_score import CLIPScoreMetric
from vision_tokenization.qualitative_benchmark.metrics.polos_score import POLOSMetric

__all__ = [
    "BaseMetric",
    "METRIC_REGISTRY",
    "register_metric",
    "get_metric",
    "list_metrics",
    "CompletionQualityMetric",
    "CLIPScoreMetric",
    "POLOSMetric",
]
