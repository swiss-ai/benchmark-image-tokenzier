"""
Base class for metrics that run in an isolated subprocess with their own venv.

Useful for metrics with dependency conflicts (e.g., POLOS requires Python <=3.10).
"""

import base64
import io
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

from PIL import Image

from vision_tokenization.qualitative_benchmark.metrics.base import BaseMetric

logger = logging.getLogger(__name__)

# Directory containing this file (metrics/)
METRICS_DIR = Path(__file__).resolve().parent


def encode_image_b64(image: Image.Image) -> str:
    """Encode a PIL Image as a base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class SubprocessMetric(BaseMetric):
    """
    Base class for metrics that execute in an isolated Python venv.

    Subclasses must define:
        VENV_DIR: Directory name of the venv (relative to metrics/)
        WORKER_SCRIPT: Filename of the worker script (relative to metrics/)

    The worker script communicates via JSON on stdin/stdout.
    If the venv is not set up, the metric gracefully returns None values.
    """

    VENV_DIR: ClassVar[str] = ""
    WORKER_SCRIPT: ClassVar[str] = ""
    SUBPROCESS_TIMEOUT: ClassVar[int] = 120

    @property
    def _venv_python(self) -> Path:
        """Path to the Python executable in the isolated venv."""
        return METRICS_DIR / self.VENV_DIR / "bin" / "python"

    @property
    def _worker_path(self) -> Path:
        """Path to the worker script."""
        return METRICS_DIR / self.WORKER_SCRIPT

    @property
    def _venv_ready(self) -> bool:
        """Check if the isolated venv exists and has a Python binary."""
        return self._venv_python.is_file()

    def _call_subprocess(self, payload: dict) -> Optional[dict]:
        """
        Call the worker script in the isolated venv.

        Args:
            payload: JSON-serializable dict to send via stdin.

        Returns:
            Parsed JSON dict from stdout, or None on failure.
        """
        if not self._venv_ready:
            logger.warning(
                f"Metric '{self.name}': venv not found at {METRICS_DIR / self.VENV_DIR}. "
                f"Run the setup script to install it. Returning None."
            )
            return None

        if not self._worker_path.is_file():
            logger.warning(f"Metric '{self.name}': worker script not found at {self._worker_path}. " f"Returning None.")
            return None

        try:
            result = subprocess.run(
                [str(self._venv_python), str(self._worker_path)],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=self.SUBPROCESS_TIMEOUT,
            )

            if result.returncode != 0:
                logger.warning(
                    f"Metric '{self.name}' subprocess failed (rc={result.returncode}):\n"
                    f"stderr: {result.stderr.strip()}"
                )
                return None

            return json.loads(result.stdout.strip())

        except subprocess.TimeoutExpired:
            logger.warning(f"Metric '{self.name}' subprocess timed out after {self.SUBPROCESS_TIMEOUT}s.")
            return None
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Metric '{self.name}' subprocess error: {e}")
            return None
