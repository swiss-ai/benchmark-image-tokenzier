"""Token-id resolution from a tokenizer's static files — no tokenizer load,
no torch.

Structure-token ids come from ``tokenizer.json``'s ``added_tokens``;
the modality bands come from ``tokenizer_config.json``'s ``omnimodal_config``.
That lets the alignment scan and the inline merge derive the manifest's
``token_layout`` without importing the EMU encoder, which pulls torch.
``EMUImageOnlyTokenizer`` re-imports ``resolve_token_ids`` / ``vision_band`` /
``STRUCTURE_TOKENS`` from here.
"""

import hashlib
import json
import os
from typing import Any, Dict, Tuple

APERTUS_1P5_BASE_VOCAB_SIZE = 131072

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


def resolve_token_ids_from_dir(
    tokenizer_dir, tokens: Dict[str, str],
) -> Dict[str, int]:
    """``resolve_token_ids`` from tokenizer.json's ``added_tokens`` — no tokenizer load."""
    path = os.path.join(tokenizer_dir, "tokenizer.json")
    with open(path, encoding="utf-8") as f:
        ids = {entry["content"]: int(entry["id"]) for entry in json.load(f)["added_tokens"]}
    missing = [token for token in tokens.values() if token not in ids]
    if missing:
        raise ValueError(
            f"{path}: added_tokens is missing {missing}. "
            f"Ensure the tokenizer vocabulary contains these tokens."
        )
    return {name: ids[token] for name, token in tokens.items()}


def tokenizer_identity(tokenizer_dir) -> Dict[str, Any]:
    """Content identity of a tokenizer, for resume safety.

    Carried in the plan fingerprint so pointing an existing output dir at a
    different tokenizer refuses to resume instead of mixing id spaces.
    """
    path = os.path.join(tokenizer_dir, "tokenizer.json")
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    config_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
    with open(config_path, encoding="utf-8") as f:
        base_vocab_size = json.load(f).get("base_vocab_size")
    return {"tokenizer_sha256": digest, "tokenizer_base_vocab_size": base_vocab_size}


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
