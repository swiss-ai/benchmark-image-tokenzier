import sys
import types

from vision_tokenization.qualitative_benchmark.v_tokenizers.emu3 import EMU3VisionTokenizer


class DummyCoreEmu3Tokenizer:
    last_kwargs = None

    def __init__(self, **kwargs):
        DummyCoreEmu3Tokenizer.last_kwargs = dict(kwargs)
        self.device = "cpu"
        self.model = None
        self.codebook_size = 8192


class DummyTextTokenizer:
    unk_token_id = 0
    boi_token = "<|img_start|>"
    img_token = "<|img_token_start|>"
    eol_token = "<|img_end_of_row|>"
    eof_token = "<|img_end_of_frame|>"
    eoi_token = "<|img_end|>"

    def convert_tokens_to_ids(self, token):
        if token == "<|visual token 000000|>":
            return 123
        return self.unk_token_id


def test_emu3_wrapper_passes_model_path(monkeypatch):
    fake_module = types.ModuleType("Tokenizer.Emu3VisionTokenizer")
    fake_module.Emu3VisionTokenizer = DummyCoreEmu3Tokenizer
    monkeypatch.setitem(sys.modules, "Tokenizer.Emu3VisionTokenizer", fake_module)
    fake_reconstruct_helper = types.ModuleType("emu3_reconstruct_helper")
    fake_reconstruct_helper.VisionTokenRange = lambda first_id, codebook_size: (first_id, codebook_size)
    monkeypatch.setitem(sys.modules, "emu3_reconstruct_helper", fake_reconstruct_helper)

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: DummyTextTokenizer())

    tokenizer = EMU3VisionTokenizer(
        model_path="/tmp/custom-emu3",
        tokenizer_path="/tmp/text-tokenizer",
        device="cpu",
    )

    assert DummyCoreEmu3Tokenizer.last_kwargs["model_path"] == "/tmp/custom-emu3"
    assert tokenizer.get_resolution_params()["model_path"] == "/tmp/custom-emu3"
