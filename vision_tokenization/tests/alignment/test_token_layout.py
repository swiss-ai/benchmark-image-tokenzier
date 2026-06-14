import pytest

from vision_tokenization.discrete.emu.token_layout import (
    STRUCTURE_TOKENS,
    resolve_token_ids,
    resolve_token_ids_from_config,
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


def test_resolve_token_ids_from_config_uses_added_tokens_decoder():
    cfg = {"added_tokens_decoder": {"0": {"content": "<unk>"},
                                    "131073": {"content": "<|img_start|>"}},
           "unk_token": "<unk>"}
    assert resolve_token_ids_from_config(
        cfg, {"img_start": "<|img_start|>"}) == {"img_start": 131073}


def test_resolve_token_ids_from_config_refuses_missing_token():
    cfg = {"added_tokens_decoder": {"0": {"content": "<unk>"}}, "unk_token": "<unk>"}
    with pytest.raises(ValueError, match="UNK"):
        resolve_token_ids_from_config(cfg, {"img_start": "<|img_start|>"})
