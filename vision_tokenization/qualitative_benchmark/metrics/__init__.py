"""
Metrics module for qualitative benchmarks.

Provides a registry system for metrics that can be dynamically loaded and used
across different benchmark types.
"""

import logging
from importlib import import_module
from typing import Dict, List, Type

from vision_tokenization.qualitative_benchmark.metrics.base import BaseMetric

logger = logging.getLogger(__name__)

# Global metric registry
METRIC_REGISTRY: Dict[str, Type[BaseMetric]] = {}
_BUILTIN_METRIC_MODULES = (
    ("vision_tokenization.qualitative_benchmark.metrics.clip_score", "CLIPScoreMetric"),
    ("vision_tokenization.qualitative_benchmark.metrics.completion_quality", "CompletionQualityMetric"),
)
_BUILTINS_REGISTERED = False


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


def ensure_builtin_metrics_registered() -> None:
    """Import bundled metric modules lazily so optional deps stay optional."""
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    for module_name, class_name in _BUILTIN_METRIC_MODULES:
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as exc:
            logger.debug("Skipping optional metric module %s: %s", module_name, exc)
            continue
        globals()[class_name] = getattr(module, class_name)

    _BUILTINS_REGISTERED = True


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
    ensure_builtin_metrics_registered()
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
    ensure_builtin_metrics_registered()
    return list(METRIC_REGISTRY.keys())

__all__ = [
    "BaseMetric",
    "METRIC_REGISTRY",
    "register_metric",
    "ensure_builtin_metrics_registered",
    "get_metric",
    "list_metrics",
]
