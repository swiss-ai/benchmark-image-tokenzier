"""Per-pair DPO text tokenization, factored out so one tokenization feeds three
consumers: the engine (vision + text in one pass), the merge (store-side slice
lengths), and the binidx writer (the flat ``.bin``).

``tokenize_pair_text`` does the text-only work and is vision-free: the prompt's
image slots are positional placeholders into ``prompt_text_ids``, filled with the
deduped vision block only at assembly time. ``pair_lengths`` turns those slots plus
the per-image vision block lengths into the DPO slice boundaries (prompt+chosen and
prompt+rejected). ``enable_thinking`` follows the CHOSEN response (the target
behavior), so chosen and rejected share one prompt prefix.

The ``TokenizedPair`` is flat (plain id arrays + integer slot positions) so the
engine can spill it to parquet and the merge/binidx read it back without a tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision_tokenization.discrete.conversation import apply_conversation_policy
from vision_tokenization.discrete.sft_segments import _has_thinking_content, split_rendered_sft


@dataclass
class TokenizedPair:
    """One preference pair tokenized at the text level. ``prompt_text_ids`` is the
    image-free prompt; ``image_insert_positions[k]`` is the ``prompt_text_ids`` index
    where the k-th media's vision block is spliced in at assembly time."""

    prompt_text_ids: np.ndarray
    image_insert_positions: list[int]
    chosen_ids: np.ndarray
    rejected_ids: np.ndarray
    enable_thinking: bool


def tokenize_pair_text(row, tok, policy, markers) -> TokenizedPair:
    """Render the prompt, tokenize its text spans, and split chosen/rejected off the
    shared prompt prefix. Vision-free: image markers become positional slots."""
    chosen, rejected = str(row["chosen"]), str(row["rejected"])

    # enable_thinking follows the CHOSEN (the target behavior); the shared prompt is
    # rendered with this one value so chosen/rejected split at the same prompt prefix.
    enable_thinking = _has_thinking_content([{"role": "assistant", "content": chosen}])
    norm_prompt = apply_conversation_policy([dict(m) for m in row["prompt"]], policy)

    rendered = tok.apply_chat_template(
        norm_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    present = [m for m in markers if m in rendered]
    text_parts, image_insert_positions, cursor = [], [], 0
    for seg in split_rendered_sft(rendered, present):
        if seg["type"] == "text":
            if not seg["text"]:
                continue
            ids = np.asarray(tok(seg["text"], add_special_tokens=False)["input_ids"], dtype=np.int32)
            text_parts.append(ids)
            cursor += int(len(ids))
        else:
            image_insert_positions.append(cursor)
    prompt_text_ids = np.concatenate(text_parts) if text_parts else np.empty(0, dtype=np.int32)

    prompt_token_ids = tok(rendered, add_special_tokens=False)["input_ids"]
    p = len(prompt_token_ids)

    def _response_ids(answer):
        if any(m in answer for m in markers):
            raise ValueError(f"{row['prompt_id']}: response carries an image marker; chosen/rejected must be text-only")
        full = tok.apply_chat_template(
            norm_prompt + [{"role": "assistant", "content": answer}],
            tokenize=True, add_generation_prompt=False, enable_thinking=enable_thinking)
        if full[:p] != prompt_token_ids:
            raise ValueError(f"{row['prompt_id']}: prompt prefix not stable under apply_chat_template")
        return np.asarray(full[p:], dtype=np.int32)

    return TokenizedPair(
        prompt_text_ids=prompt_text_ids,
        image_insert_positions=image_insert_positions,
        chosen_ids=_response_ids(chosen),
        rejected_ids=_response_ids(rejected),
        enable_thinking=enable_thinking,
    )


def seq_lengths(prompt_text_len: int, vision_total: int, chosen_len: int, rejected_len: int):
    """``(prompt_len, seq_chosen_len, seq_rejected_len)`` — ``prompt_len`` is text + inlined
    vision; the seq lengths are the ``[prompt|chosen]`` / ``[prompt|rejected]`` span lengths.
    The single seq arithmetic shared by the binidx index and the merge's store rows."""
    prompt_len = int(prompt_text_len) + int(vision_total)
    return prompt_len, prompt_len + int(chosen_len), prompt_len + int(rejected_len)


def pair_lengths(tp: TokenizedPair, image_token_lengths, *, prompt_id: str = "") -> dict:
    """The DPO index row: prompt/chosen/rejected lengths, the prompt+chosen and
    prompt+rejected slice lengths, and the in-prompt vision block locations. The
    single offset arithmetic shared by the merge (store rows) and the binidx writer."""
    if len(image_token_lengths) != len(tp.image_insert_positions):
        raise ValueError(
            f"{len(tp.image_insert_positions)} image slots but {len(image_token_lengths)} vision blocks")
    image_offsets, image_lengths, cum_vision = [], [], 0
    for pos, ln in zip(tp.image_insert_positions, image_token_lengths):
        ln = int(ln)
        image_offsets.append(pos + cum_vision)
        image_lengths.append(ln)
        cum_vision += ln
    chosen_len, rejected_len = int(len(tp.chosen_ids)), int(len(tp.rejected_ids))
    prompt_len, seq_chosen, seq_rejected = seq_lengths(
        len(tp.prompt_text_ids), cum_vision, chosen_len, rejected_len)
    return {
        "prompt_id": prompt_id,
        "prompt_len": prompt_len,
        "chosen_len": chosen_len,
        "rejected_len": rejected_len,
        "image_offsets": image_offsets,
        "image_lengths": image_lengths,
        "seq_chosen_len": seq_chosen,
        "seq_rejected_len": seq_rejected,
        "image_tok": int(sum(image_lengths)),
    }


def assemble_from_tokenized(tp: TokenizedPair, images, tokens, *, prompt_id: str = ""):
    """Return (doc:int32 np.ndarray, index_row:dict): splice each media's deduped
    vision block into the prompt slots and concat ``[prompt | chosen | rejected]``.
    ``images[k]`` carries ``token_offset``/``token_length`` into the ``tokens`` store.
    No tokenizer — the text is already tokenized in ``tp``."""
    index_row = pair_lengths(tp, [int(im["token_length"]) for im in images], prompt_id=prompt_id)
    parts, prev = [], 0
    for pos, im in zip(tp.image_insert_positions, images):
        parts.append(tp.prompt_text_ids[prev:pos])
        off, ln = int(im["token_offset"]), int(im["token_length"])
        parts.append(np.asarray(tokens[off : off + ln], dtype=np.int32))
        prev = pos
    parts.append(tp.prompt_text_ids[prev:])
    prompt_ids = np.concatenate(parts)
    if len(prompt_ids) != index_row["prompt_len"]:
        raise RuntimeError(
            f"{prompt_id}: spliced prompt {len(prompt_ids)} != layout {index_row['prompt_len']}")
    doc = np.concatenate([prompt_ids, tp.chosen_ids, tp.rejected_ids])
    return doc, index_row


def assemble_pair(row, tokens, tok, policy, markers):
    """Tokenize one pair's text and assemble its ``.bin`` doc in a single call — the
    all-in-one path the current binidx writer uses. ``tokenize_pair_text`` +
    ``assemble_from_tokenized`` are the same work split for the engine/merge flow."""
    tp = tokenize_pair_text(row, tok, policy, markers)
    return assemble_from_tokenized(tp, list(row["images"]), tokens, prompt_id=row["prompt_id"])


def tokenized_to_row(tp: TokenizedPair, prompt_id: str) -> dict:
    """Flatten a ``TokenizedPair`` to a parquet-friendly row (plain lists) for the
    engine spill. Pairs with ``tokenized_from_row``."""
    return {
        "prompt_id": prompt_id,
        "prompt_text_ids": tp.prompt_text_ids.tolist(),
        "image_insert_positions": [int(x) for x in tp.image_insert_positions],
        "chosen_ids": tp.chosen_ids.tolist(),
        "rejected_ids": tp.rejected_ids.tolist(),
        "enable_thinking": bool(tp.enable_thinking),
    }


def tokenized_from_row(row: dict) -> TokenizedPair:
    """Rebuild a ``TokenizedPair`` from a spilled row (the merge/binidx side)."""
    return TokenizedPair(
        prompt_text_ids=np.asarray(row["prompt_text_ids"], dtype=np.int32),
        image_insert_positions=[int(x) for x in row["image_insert_positions"]],
        chosen_ids=np.asarray(row["chosen_ids"], dtype=np.int32),
        rejected_ids=np.asarray(row["rejected_ids"], dtype=np.int32),
        enable_thinking=bool(row["enable_thinking"]),
    )
