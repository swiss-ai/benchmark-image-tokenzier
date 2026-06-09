"""The wrapper's two-phase preprocess must be bit-identical to the legacy
per-image path: same PIL ops, same cast/normalize arithmetic, same layout.

CPU-only — guards the contract that lets prefetch workers run the CPU phase
while inference consumers keep calling preprocess_batch.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "Tokenizer" / "submodules" / "Emu3.5" / "src"))
pytest.importorskip("vision_tokenizer")

from Tokenizer.Emu3_5_IBQ import Emu3_5_IBQ  # noqa: E402


def _legacy_preprocess(images, resize_size, device, dtype):
    """The pre-refactor per-image implementation, kept here as the reference."""
    height, width = resize_size
    out = []
    for image in images:
        if image.mode != "RGB":
            image = image.convert("RGB")
        iw, ih = image.size
        if iw != width or ih != height:
            image = image.resize((width, height), Image.BICUBIC)
        t = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True))
        t = t.permute(2, 0, 1).to(device, dtype=dtype)
        t.div_(127.5).sub_(1.0)
        out.append(t)
    return torch.stack(out, dim=0)


def _dummy_wrapper():
    """Minimal stand-in carrying just the attrs the preprocess methods read,
    with the real methods bound so the composition path works too."""
    d = types.SimpleNamespace()
    d.device = torch.device("cpu")
    d.dtype = torch.float32
    d.preprocess_cpu = Emu3_5_IBQ.preprocess_cpu.__get__(d)
    d.to_device = Emu3_5_IBQ.to_device.__get__(d)
    return d


def _sample_images(rng):
    sizes = [(64, 48), (32, 32), (100, 80)]
    modes = ["RGB", "L", "RGBA"]
    images = []
    for (w, h), mode in zip(sizes, modes):
        channels = {"RGB": 3, "L": 1, "RGBA": 4}[mode]
        arr = rng.integers(0, 256, size=(h, w, channels), dtype=np.uint8).squeeze()
        images.append(Image.fromarray(arr, mode=mode))
    return images


def test_two_phase_matches_legacy_bitwise():
    rng = np.random.default_rng(42)
    images = _sample_images(rng)
    dummy = _dummy_wrapper()
    size = (32, 32)

    expected = _legacy_preprocess(images, size, dummy.device, dummy.dtype)

    pixels = Emu3_5_IBQ.preprocess_cpu(dummy, images, size)
    assert pixels.dtype == torch.uint8
    assert pixels.shape == (3, 32, 32, 3)  # [B, H, W, C]
    actual = Emu3_5_IBQ.to_device(dummy, pixels)

    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_preprocess_batch_is_the_composition():
    rng = np.random.default_rng(7)
    images = _sample_images(rng)
    dummy = _dummy_wrapper()
    size = (48, 64)

    composed = Emu3_5_IBQ.preprocess_batch(dummy, images, size)
    expected = _legacy_preprocess(images, size, dummy.device, dummy.dtype)
    assert torch.equal(composed, expected)
