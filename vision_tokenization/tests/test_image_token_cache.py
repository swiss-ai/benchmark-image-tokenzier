import os
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from vision_tokenization.qualitative_benchmark.image_token_cache import ImageTokenCache
from vision_tokenization.qualitative_benchmark.vlm import InferenceArgs, VLM


class DummyTextTokenizer:
    eos_token_id = 1
    unk_token_id = 0
    init_kwargs = {}


class DummyInferencer:
    def __init__(self):
        self._txt_tokenizer = DummyTextTokenizer()

    @property
    def txt_tokenizer(self):
        return self._txt_tokenizer

    def run_inference(self, prompt, **kwargs):
        return {"generated_ids": [1, 2, 3], "generated_text": prompt}


class DummyVisionTokenizer:
    def __init__(self):
        self.encode_calls = 0
        self.tokenizer = type("CoreTokenizer", (), {"device": "cpu"})()

    @property
    def name(self):
        return "DummyVisionTokenizer"

    def encode_for_vlm(self, image):
        self.encode_calls += 1
        indices = torch.tensor([[11, 12], [13, 14]], dtype=torch.long)
        metadata = {"height": 2, "width": 2, "num_tokens": 4}
        return indices, metadata

    def format_tokens_for_chat(self, indices, metadata, special_tokens):
        flat = indices.flatten().tolist()
        return f"encoded:{metadata['height']}x{metadata['width']}:{','.join(str(v) for v in flat)}"

    def create_partial_prompt(self, visual_indices, height, width, given_rows):
        return f"partial:{height}:{width}:{given_rows}:{','.join(str(v) for v in visual_indices)}"

    @property
    def vision_mapping(self):
        return {}


class DummyPromptFormatter:
    def __init__(self, tokenizer_path=None):
        self.tokenizer_path = tokenizer_path

    def prepare_non_chat_prompt(self, txt_string: str, img_token_string: str = None, img_right: bool = False) -> str:
        prompt = txt_string
        if img_token_string is not None:
            return prompt + img_token_string if img_right else img_token_string + prompt
        return prompt

    def prepare_chat_prompt(self, *args, **kwargs):
        raise AssertionError("chat template path should not be used in this unit test")

    def prepare_custom_prompt(self, *args, **kwargs):
        raise AssertionError("custom prompt path should not be used in this unit test")


def _make_vlm(tmp_path: Path):
    cache = ImageTokenCache(
        cache_dir=tmp_path / "cache",
        tokenizer_type="emu3.5",
        tokenizer_kwargs={"min_pixels": 256 * 256, "max_pixels": 512 * 512, "tokenizer_path": "dummy"},
    )
    vision_tokenizer = DummyVisionTokenizer()
    with patch("vision_tokenization.qualitative_benchmark.vlm.PromptFormatter", DummyPromptFormatter):
        vlm = VLM(
            vision_tokenizer=vision_tokenizer,
            inferencer=DummyInferencer(),
            inf_args=InferenceArgs(
                apply_chat_template=False,
                temperature=0.0,
                top_p=1.0,
                stop_token_ids=[],
                max_new_tokens=16,
                max_emu_aspect_ratio=512 * 512,
                min_emu_aspect_ratio=256 * 256,
            ),
            tokenizer_path="dummy",
            model_path="dummy-model",
            image_token_cache=cache,
        )

    return vlm, vision_tokenizer, cache


def test_vlm_reuses_cached_image_tokens(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(image_path)
    vlm, vision_tokenizer, cache = _make_vlm(tmp_path)

    first_prompt = vlm.preprocess(str(image_path), "Describe")
    second_prompt = vlm.preprocess(str(image_path), "Describe")

    assert first_prompt == second_prompt
    assert vision_tokenizer.encode_calls == 1
    assert vlm.image_token_cache_stats == {"hits": 1, "misses": 1, "writes": 1}
    assert cache.inspect_many([image_path])["hits"] == 1


def test_vlm_reencodes_when_source_image_changes(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(image_path)
    vlm, vision_tokenizer, _ = _make_vlm(tmp_path)

    vlm.preprocess(str(image_path), "Describe")
    Image.new("RGB", (8, 8), color=(30, 20, 10)).save(image_path)
    stat = image_path.stat()
    os.utime(image_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    vlm.preprocess(str(image_path), "Describe")

    assert vision_tokenizer.encode_calls == 2
