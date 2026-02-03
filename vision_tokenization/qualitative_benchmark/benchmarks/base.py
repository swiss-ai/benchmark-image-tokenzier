"""
Base class for benchmarks.

All benchmarks inherit from BaseBenchmark and implement the run() method.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from vision_tokenization.qualitative_benchmark.metrics import get_metric
from vision_tokenization.qualitative_benchmark.metrics.base import BaseMetric


class BaseBenchmark(ABC):
    """
    Abstract base class for benchmarks.

    Benchmarks orchestrate running inference tasks and computing metrics.
    They handle result storage, metric computation, and provide common utilities.

    Class Attributes:
        name: Unique identifier for this benchmark type
        METRICS: List of metric names to compute (from metrics registry)

    Subclasses must implement:
        run(**kwargs) -> dict: Execute the benchmark and return results
    """

    name: ClassVar[str] = ""
    METRICS: ClassVar[List[str]] = []

    def __init__(
        self,
        vlm,
        results_dir: str = "results",
        debug: bool = False,
        metrics: Optional[List[str]] = None,
    ):
        """
        Initialize the benchmark.

        Args:
            vlm: VLM instance for running inference
            results_dir: Directory where results will be stored
            debug: Enable debug mode for detailed output
            metrics: Optional list of metric names to use (overrides METRICS)
        """
        self.vlm = vlm
        self.results_dir = Path(results_dir)
        self.debug = debug

        # Initialize metrics
        metric_names = metrics if metrics is not None else self.METRICS
        self._init_metrics(metric_names)

    def _init_metrics(self, metric_names: List[str]):
        """
        Initialize metric instances from registry.

        Args:
            metric_names: List of metric identifiers to initialize
        """
        self.metrics: Dict[str, BaseMetric] = {}

        # Determine device from VLM if possible
        device = "cpu"
        if hasattr(self.vlm, "vision_tokenizer") and hasattr(self.vlm.vision_tokenizer, "tokenizer"):
            tokenizer = self.vlm.vision_tokenizer.tokenizer
            if hasattr(tokenizer, "device"):
                device = str(tokenizer.device)

        for name in metric_names:
            try:
                self.metrics[name] = get_metric(name, device=device)
            except ValueError as e:
                print(f"Warning: Could not initialize metric '{name}': {e}")

    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        """Load JSON configuration file."""
        with open(path, "r") as f:
            return json.load(f)

    def _serialize_inference_args(self) -> dict:
        """
        Serialize VLM inference args to a dictionary.

        Returns:
            Dictionary with inference configuration
        """
        inf_args = self.vlm.inf_args
        return {
            "apply_chat_template": inf_args.apply_chat_template,
            "prompt_builder": inf_args.prompt_builder,
            "temperature": inf_args.temperature,
            "top_p": inf_args.top_p,
            "max_new_tokens": inf_args.max_new_tokens,
            "max_emu_aspect_ratio": inf_args.max_emu_aspect_ratio,
            "min_emu_aspect_ratio": inf_args.min_emu_aspect_ratio,
            "stop_token_ids": inf_args.stop_token_ids,
        }

    def _compute_metrics(self, data: dict) -> dict:
        """
        Run all registered metrics and merge results.

        Args:
            data: Dictionary with input data for metrics

        Returns:
            Flat dictionary of merged metric key-value pairs
        """
        metrics_result = {}
        for metric_name, metric in self.metrics.items():
            try:
                result = metric(data)
                metrics_result.update(result)
            except Exception as e:
                print(f"Warning: Metric '{metric_name}' failed: {e}")
        return metrics_result

    def _create_base_results(self) -> dict:
        """
        Create common result structure.

        Returns:
            Dictionary with standard result metadata
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "model_path": self.vlm.model_path,
            "tokenizer_path": self.vlm.tokenizer_path,
            "vision_tokenizer": self.vlm.vision_tokenizer.name,
            "inferencer": type(self.vlm.inferencer).__name__,
            "inference_args": self._serialize_inference_args(),
            "total_runs": 0,
            "runs": [],
        }

    def _save_results(self, results: dict, filename: str) -> Path:
        """
        Save results to JSON file.

        Args:
            results: Results dictionary to save
            filename: Output filename (relative to results_dir)

        Returns:
            Path to saved file
        """
        output_path = self.results_dir / filename
        self.results_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        return output_path

    @abstractmethod
    def run(self, **kwargs) -> dict:
        """
        Execute the benchmark.

        Subclasses must implement this method to:
        1. Load images/prompts from configuration
        2. Run inference using self.vlm
        3. Compute metrics using self._compute_metrics()
        4. Save and return results

        Returns:
            Dictionary containing benchmark results
        """
        raise NotImplementedError
