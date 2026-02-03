"""
Benchmarks module for qualitative evaluation.

Provides a registry system for benchmarks that can be dynamically loaded
and executed via the CLI.
"""

from typing import Dict, List, Type

from vision_tokenization.qualitative_benchmark.benchmarks.base import BaseBenchmark

# Global benchmark registry
BENCHMARK_REGISTRY: Dict[str, Type[BaseBenchmark]] = {}


def register_benchmark(name: str):
    """
    Decorator to register a benchmark class.

    Usage:
        @register_benchmark("vlm")
        class VLMBenchmark(BaseBenchmark):
            ...
    """

    def decorator(cls: Type[BaseBenchmark]) -> Type[BaseBenchmark]:
        if name in BENCHMARK_REGISTRY:
            raise ValueError(f"Benchmark '{name}' is already registered")
        BENCHMARK_REGISTRY[name] = cls
        return cls

    return decorator


def get_benchmark(name: str, **kwargs) -> BaseBenchmark:
    """
    Factory function to create a benchmark instance by name.

    Args:
        name: Benchmark identifier (e.g., "vlm", "image_completion", "captioning")
        **kwargs: Arguments passed to the benchmark constructor

    Returns:
        Initialized benchmark instance

    Raises:
        ValueError: If benchmark name is not found in registry
    """
    if name not in BENCHMARK_REGISTRY:
        available = ", ".join(BENCHMARK_REGISTRY.keys()) or "none"
        raise ValueError(f"Unknown benchmark '{name}'. Available benchmarks: {available}")

    benchmark_cls = BENCHMARK_REGISTRY[name]
    return benchmark_cls(**kwargs)


def list_benchmarks() -> List[str]:
    """
    List all registered benchmark names.

    Returns:
        List of benchmark identifiers
    """
    return list(BENCHMARK_REGISTRY.keys())


# Import benchmarks to trigger registration
# These imports must come after the registry is defined
from vision_tokenization.qualitative_benchmark.benchmarks.vlm_benchmark import VLMBenchmark
from vision_tokenization.qualitative_benchmark.benchmarks.image_completion import ImageCompletionBenchmark
from vision_tokenization.qualitative_benchmark.benchmarks.captioning import CaptioningBenchmark

__all__ = [
    "BaseBenchmark",
    "BENCHMARK_REGISTRY",
    "register_benchmark",
    "get_benchmark",
    "list_benchmarks",
    "VLMBenchmark",
    "ImageCompletionBenchmark",
    "CaptioningBenchmark",
]
