#!/usr/bin/env python
"""
Interactive setup script for the POLOS metric environment.

Creates an isolated Python 3.10 venv using `uv`, installs POLOS,
downloads the model, and validates the installation.

Usage:
    python -m vision_tokenization.qualitative_benchmark.metrics.setup_polos
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from vision_tokenization.qualitative_benchmark.metrics.polos_score import POLOS_VENV_DIR

METRICS_DIR = Path(__file__).resolve().parent
VENV_PATH = METRICS_DIR / POLOS_VENV_DIR
VENV_PYTHON = VENV_PATH / "bin" / "python"
WORKER_SCRIPT = METRICS_DIR / "_polos_worker.py"


def run(cmd, **kwargs):
    """Run a command and return the CompletedProcess, printing on failure."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode})")
        if result.stderr:
            print(f"  stderr: {result.stderr.strip()}")
        if result.stdout:
            print(f"  stdout: {result.stdout.strip()}")
    return result


def check_uv():
    """Check for uv on PATH, offer to install if missing."""
    uv = shutil.which("uv")
    if uv:
        print(f"[1/6] uv found: {uv}")
        return True

    print("[1/6] uv not found on PATH.")
    print()
    print("  uv is a fast Python package manager needed to create the POLOS venv.")
    print("  Install it from: https://docs.astral.sh/uv/getting-started/installation/")
    print()
    answer = input("  Would you like to install uv now? [y/N] ").strip().lower()
    if answer != "y":
        print("  Skipping uv installation. Install uv manually and re-run this script.")
        return False

    print("  Installing uv...")
    result = subprocess.run(
        ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("  Failed to install uv. Please install manually:")
        print("    curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False

    # Re-check
    uv = shutil.which("uv")
    if not uv:
        print("  uv installed but not on PATH. You may need to restart your shell.")
        print("  Then re-run this setup script.")
        return False

    print(f"  uv installed: {uv}")
    return True


def install_python():
    """Ensure Python 3.10 is available via uv."""
    print("[2/6] Ensuring Python 3.10 is available...")
    result = run(["uv", "python", "install", "3.10"])
    if result.returncode != 0:
        print("  Failed to install Python 3.10 via uv.")
        return False
    print("  Python 3.10 ready.")
    return True


def create_venv():
    """Create the isolated venv."""
    print(f"[3/6] Creating venv at {VENV_PATH}...")
    if VENV_PATH.exists():
        print(f"  Venv already exists. To recreate, delete {VENV_PATH} and re-run.")
        if VENV_PYTHON.is_file():
            print("  Existing venv looks valid, continuing.")
            return True
        print("  Existing venv is broken (no python binary). Removing...")
        shutil.rmtree(VENV_PATH)

    result = run(["uv", "venv", "--python", "3.10", str(VENV_PATH)])
    if result.returncode != 0:
        print("  Failed to create venv.")
        return False
    print("  Venv created.")
    return True


def install_polos():
    """Install POLOS into the venv."""
    print("[4/6] Installing POLOS...")
    result = run(["uv", "pip", "install", "polos", "--python", str(VENV_PYTHON)])
    if result.returncode != 0:
        print("  Failed to install POLOS.")
        return False
    print("  POLOS installed.")
    return True


def download_model():
    """Download the POLOS model weights."""
    print("[5/6] Downloading POLOS model...")
    script = "from polos.models import download_model; download_model('polos'); print('Model downloaded.')"
    result = run([str(VENV_PYTHON), "-c", script])
    if result.returncode != 0:
        print("  Failed to download POLOS model.")
        return False
    print("  Model ready.")
    return True


def validate():
    """Run the worker with dummy input to verify everything works."""
    print("[6/6] Validating installation...")

    # Create a tiny 8x8 red test image
    import base64
    import io

    from PIL import Image

    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    payload = json.dumps({"image_b64": img_b64, "caption": "a red square"})

    result = subprocess.run(
        [str(VENV_PYTHON), str(WORKER_SCRIPT)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        print(f"  Validation failed (rc={result.returncode})")
        if result.stderr:
            print(f"  stderr: {result.stderr.strip()}")
        return False

    try:
        output = json.loads(result.stdout.strip())
        score = output.get("polos_score")
        if score is None:
            print("  Validation failed: no polos_score in output.")
            return False
        print(f"  Validation passed! Test score: {score:.4f}")
        return True
    except json.JSONDecodeError:
        print(f"  Validation failed: could not parse output: {result.stdout}")
        return False


def main():
    print("=" * 60)
    print("POLOS Metric Setup")
    print("=" * 60)
    print()

    steps = [check_uv, install_python, create_venv, install_polos, download_model, validate]

    for step in steps:
        if not step():
            print()
            print("Setup FAILED. Fix the issue above and re-run.")
            sys.exit(1)
        print()

    print("=" * 60)
    print("POLOS setup complete!")
    print(f"Venv location: {VENV_PATH}")
    print("The polos_score metric will now be available in captioning benchmarks.")
    print("=" * 60)


if __name__ == "__main__":
    main()
