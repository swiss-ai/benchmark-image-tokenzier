import json
import sys
import types

import torch

from vision_tokenization.qualitative_benchmark.v_tokenizers.emu3 import EMU3VisionTokenizer


class DummyCoreEmu3Tokenizer:
    def __init__(self, **kwargs):
        self.device = "cpu"
        self.model = None
        self.codebook_size = 131072


class DummyTextTokenizerNoPadding:
    unk_token_id = 0
    boi_token = "<|img_start|>"
    img_token = "<|img_token_start|>"
    eol_token = "<|img_end_of_row|>"
    eof_token = "<|img_end_of_frame|>"
    eoi_token = "<|img_end|>"

    def convert_tokens_to_ids(self, token):
        table = {
            "<|visual token 0|>": 131272,
            "<|img_start|>": 131073,
            "<|img_end|>": 131074,
            "<|img_token_start|>": 131075,
            "<|img_end_of_row|>": 131076,
            "<|img_end_of_frame|>": 131077,
        }
        return table.get(token, self.unk_token_id)


def test_visual_token_format_probed_without_zero_padding(monkeypatch, tmp_path):
    tokenizer_dir = tmp_path / "tok"
    tokenizer_dir.mkdir()
    fake_module = types.ModuleType("Tokenizer.Emu3VisionTokenizer")
    fake_module.Emu3VisionTokenizer = DummyCoreEmu3Tokenizer
    monkeypatch.setitem(sys.modules, "Tokenizer.Emu3VisionTokenizer", fake_module)

    fake_reconstruct_helper = types.ModuleType("emu3_reconstruct_helper")
    fake_reconstruct_helper.VisionTokenRange = lambda first_id, codebook_size: (first_id, codebook_size)
    monkeypatch.setitem(sys.modules, "emu3_reconstruct_helper", fake_reconstruct_helper)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTextTokenizerNoPadding(),
    )

    tokenizer = EMU3VisionTokenizer(
        model_path="/tmp/custom-emu3",
        tokenizer_path=str(tokenizer_dir),
        device="cpu",
    )
    indices = torch.tensor([[11, 12], [13, 14]], dtype=torch.long)
    metadata = {"height": 2, "width": 2, "num_tokens": 4}
    prompt = tokenizer.format_tokens_for_chat(indices, metadata, {})

    assert "<|visual token 11|>" in prompt
    assert "<|visual token 000011|>" not in prompt
    assert tokenizer.visual_token_template == "<|visual token {token}|>"
