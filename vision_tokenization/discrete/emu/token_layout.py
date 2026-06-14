"""Token-id resolution from a tokenizer's static config — no tokenizer load,
no torch.

These helpers read ids straight out of ``tokenizer_config.json``
(``added_tokens_decoder`` / ``omnimodal_config``) so the alignment scan and the
inline merge can derive the manifest's ``token_layout`` without importing the
EMU encoder (which pulls torch). ``EMUImageOnlyTokenizer`` re-imports
``resolve_token_ids`` / ``vision_band`` / ``STRUCTURE_TOKENS`` from here.
"""

from typing import Dict, Tuple

STRUCTURE_TOKENS = {
    "img_start": "<|img_start|>",
    "img_end": "<|img_end|>",
    "img_token_start": "<|img_token_start|>",
    "eol": "<|img_end_of_row|>",
    "eof": "<|img_end_of_frame|>",
}


def resolve_token_ids(text_tokenizer, tokens: Dict[str, str]) -> Dict[str, int]:
    """Resolve special tokens to ids, refusing any that fall back to UNK."""
    unk_id = text_tokenizer.unk_token_id
    resolved = {}
    for name, token in tokens.items():
        tid = text_tokenizer.convert_tokens_to_ids(token)
        if tid == unk_id:
            raise ValueError(
                f"Special token {token} resolved to UNK (id={unk_id}). "
                f"Ensure the tokenizer vocabulary contains this token."
            )
        resolved[name] = tid
    return resolved


class _ConfigVocab:
    """Duck-typed vocab over tokenizer_config.json's ``added_tokens_decoder``
    so ``resolve_token_ids`` works without loading the tokenizer."""

    def __init__(self, tokenizer_config: dict):
        self._ids = {info["content"]: int(tid)
                     for tid, info in tokenizer_config["added_tokens_decoder"].items()}
        unk = tokenizer_config.get("unk_token")
        if isinstance(unk, dict):
            unk = unk.get("content")
        self.unk_token_id = self._ids.get(unk)

    def convert_tokens_to_ids(self, token: str):
        return self._ids.get(token, self.unk_token_id)


def resolve_token_ids_from_config(
    tokenizer_config: dict, tokens: Dict[str, str],
) -> Dict[str, int]:
    """``resolve_token_ids`` from the config dict alone — no tokenizer load."""
    return resolve_token_ids(_ConfigVocab(tokenizer_config), tokens)


def vision_band(tokenizer_config: dict) -> Tuple[int, int]:
    """Inclusive [lo, hi] id range of vision codebook tokens (omnimodal_config)."""
    omni_cfg = tokenizer_config.get("omnimodal_config", {})
    vision_modality = next(
        (m for m in omni_cfg.get("modalities", []) if m["name"] == "vision"), None
    )
    if vision_modality is None:
        raise ValueError(
            "No vision modality found in tokenizer_config.json omnimodal_config. "
            "Ensure the tokenizer has omnimodal_config.modalities with a 'vision' entry."
        )
    try:
        lo, size = vision_modality["offset"], vision_modality["vocab_size"]
    except KeyError as e:
        raise ValueError(
            f"vision modality in omnimodal_config is missing {e.args[0]!r}; "
            f"expected both 'offset' and 'vocab_size'"
        ) from None
    return lo, lo + size - 1
