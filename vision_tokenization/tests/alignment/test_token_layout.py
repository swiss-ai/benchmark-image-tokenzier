import json

import pytest

from vision_tokenization.discrete.emu.token_layout import (
    STRUCTURE_TOKENS,
    resolve_token_ids,
    resolve_token_ids_from_dir,
    vision_band,
)


class _Vocab:
    unk_token_id = 0

    def __init__(self, table):
        self._table = table

    def convert_tokens_to_ids(self, token):
        return self._table.get(token, 0)


def test_resolve_token_ids_maps_names_to_ids():
    table = {tok: 100 + i for i, tok in enumerate(STRUCTURE_TOKENS.values())}
    ids = resolve_token_ids(_Vocab(table), STRUCTURE_TOKENS)
    assert ids == {name: table[tok] for name, tok in STRUCTURE_TOKENS.items()}


def test_resolve_token_ids_refuses_unk():
    with pytest.raises(ValueError, match="UNK"):
        resolve_token_ids(_Vocab({}), {"img_start": "<|img_start|>"})


def test_vision_band_inclusive_range():
    cfg = {"omnimodal_config": {"modalities": [
        {"name": "vision", "offset": 131272, "vocab_size": 131072}]}}
    assert vision_band(cfg) == (131272, 262343)


def test_vision_band_requires_vision_modality():
    with pytest.raises(ValueError, match="vision modality"):
        vision_band({"omnimodal_config": {"modalities": []}})


def test_vision_band_missing_vocab_size_fails_loud():
    cfg = {"omnimodal_config": {"modalities": [{"name": "vision", "offset": 7}]}}
    with pytest.raises(ValueError, match="vocab_size"):
        vision_band(cfg)


def _tokenizer_dir(tmp_path, added_tokens):
    (tmp_path / "tokenizer.json").write_text(
        json.dumps({"added_tokens": added_tokens}), encoding="utf-8")
    return tmp_path


def test_resolve_token_ids_from_dir_reads_added_tokens(tmp_path):
    d = _tokenizer_dir(tmp_path, [{"id": 0, "content": "<unk>"},
                                  {"id": 131073, "content": "<|img_start|>"}])
    assert resolve_token_ids_from_dir(
        d, {"img_start": "<|img_start|>"}) == {"img_start": 131073}


def test_resolve_token_ids_from_dir_reads_low_ids(tmp_path):
    """Apertus 2 puts structure tokens below the base vocab, not above it."""
    d = _tokenizer_dir(tmp_path, [{"id": 27, "content": "<|img_start|>"}])
    assert resolve_token_ids_from_dir(
        d, {"img_start": "<|img_start|>"}) == {"img_start": 27}


def test_tokenizer_identity_distinguishes_generations(tmp_path):
    from vision_tokenization.discrete.emu.token_layout import tokenizer_identity

    def make(name, base_vocab_size, added):
        d = tmp_path / name
        d.mkdir()
        (d / "tokenizer.json").write_text(json.dumps({"added_tokens": added}), encoding="utf-8")
        (d / "tokenizer_config.json").write_text(
            json.dumps({"base_vocab_size": base_vocab_size}), encoding="utf-8")
        return d

    a = tokenizer_identity(make("a", 131072, [{"id": 131073, "content": "<|img_start|>"}]))
    b = tokenizer_identity(make("b", 200064, [{"id": 27, "content": "<|img_start|>"}]))
    assert a["tokenizer_base_vocab_size"] == 131072
    assert b["tokenizer_base_vocab_size"] == 200064
    assert a["tokenizer_sha256"] != b["tokenizer_sha256"]


def test_resolve_token_ids_from_dir_refuses_missing_token(tmp_path):
    d = _tokenizer_dir(tmp_path, [{"id": 0, "content": "<unk>"}])
    with pytest.raises(ValueError, match="missing"):
        resolve_token_ids_from_dir(d, {"img_start": "<|img_start|>"})
