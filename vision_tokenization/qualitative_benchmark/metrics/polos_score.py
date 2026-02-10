"""
POLOS metric for evaluating image caption quality.

POLOS (CVPR 2024) is a reference-free caption quality metric that correlates
well with human judgments. It runs in an isolated subprocess because it requires
Python <=3.10 and incompatible dependency versions.

Setup: python -m vision_tokenization.qualitative_benchmark.metrics.setup_polos
"""

from typing import ClassVar, Set

from vision_tokenization.qualitative_benchmark.metrics import register_metric
from vision_tokenization.qualitative_benchmark.metrics.subprocess_base import (
    SubprocessMetric,
    encode_image_b64,
)

# Shared constant: venv directory name (used by both metric and setup script)
POLOS_VENV_DIR = ".polos_env"


@register_metric("polos_score")
class POLOSMetric(SubprocessMetric):
    """
    POLOS caption quality metric (reference-free).

    Sends image and caption to an isolated subprocess running POLOS,
    returns a quality score. If the POLOS venv is not set up, returns None
    and the benchmark continues with other metrics.

    Required input keys:
        image: PIL Image
        caption: Text caption to evaluate

    Output keys:
        polos_score: Quality score (float) or None if unavailable
    """

    name: ClassVar[str] = "polos_score"
    REQUIRED_KEYS: ClassVar[Set[str]] = {"image", "caption"}
    VENV_DIR: ClassVar[str] = POLOS_VENV_DIR
    WORKER_SCRIPT: ClassVar[str] = "_polos_worker.py"

    def __call__(self, data: dict) -> dict:
        data = self.validate_input(data)

        caption = data["caption"]
        if not caption or not caption.strip():
            return {"polos_score": None}

        image_b64 = encode_image_b64(data["image"])
        payload = {"image_b64": image_b64, "caption": caption}

        result = self._call_subprocess(payload)
        if result is None:
            return {"polos_score": None}

        return {"polos_score": result.get("polos_score")}
