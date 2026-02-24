#!/usr/bin/env python
"""
POLOS worker script — runs inside the isolated .polos_env venv.

Reads a JSON payload from stdin with keys:
    image_b64: base64-encoded PNG image
    caption: text caption to evaluate

Writes a JSON result to stdout:
    polos_score: float quality score

All diagnostic output goes to stderr to avoid corrupting JSON on stdout.
"""

import base64
import io
import json
import sys


def main():
    try:
        payload = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        print(f"Failed to parse input JSON: {e}", file=sys.stderr)
        sys.exit(1)

    image_b64 = payload.get("image_b64")
    caption = payload.get("caption")

    if not image_b64 or not caption:
        print("Missing 'image_b64' or 'caption' in payload", file=sys.stderr)
        sys.exit(1)

    # Decode image
    from PIL import Image

    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Redirect stdout to stderr before importing POLOS to silence any library
    # output that would corrupt the JSON result written to stdout later.
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr

    # Load POLOS model
    from polos.models import download_model, load_checkpoint

    print("Loading POLOS model...", file=sys.stderr)
    model_path = download_model("polos")
    model = load_checkpoint(model_path)
    print("POLOS model loaded.", file=sys.stderr)

    # Build data dict (reference-free: empty refs list)
    data = {
        "img": [image],
        "mt": [caption],
        "refs": [[]],
    }

    # Run prediction
    _, scores = model.predict(data)
    score = float(scores[0])

    # Restore stdout and write JSON result
    sys.stdout = _real_stdout
    json.dump({"polos_score": score}, sys.stdout)


if __name__ == "__main__":
    main()
