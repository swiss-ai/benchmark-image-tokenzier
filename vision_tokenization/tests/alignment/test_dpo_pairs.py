"""One tokenization feeds engine/merge/binidx. Asserts: (1) ``assemble_pair`` is
byte-identical to the pre-refactor logic, (2) the merge-side ``pair_lengths``
reproduces the binidx index lengths WITHOUT the vision store, (3) ``enable_thinking``
follows the CHOSEN only. Needs transformers + the Apertus tokenizer; skipped otherwise.
"""

import os

import numpy as np
import pytest

TOK_DIR = "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed"


@pytest.fixture(scope="module")
def tok():
    pytest.importorskip("transformers")
    if not os.path.isdir(TOK_DIR):
        pytest.skip("Apertus tokenizer not present")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=True, use_fast=True)


@pytest.fixture(scope="module")
def policy_markers(tok):
    from vision_tokenization.discrete.conversation import ConversationPolicy
    from vision_tokenization.discrete.sft_segments import build_image_marker_candidates
    policy = ConversationPolicy(add_system_message=True)
    return policy, build_image_marker_candidates(text_tokenizer=tok, conversation_policy=policy)


def _row(prompt_text, chosen, rejected, images):
    return {"prompt": [{"role": "user", "content": prompt_text}], "chosen": chosen,
            "rejected": rejected, "images": images, "prompt_id": "p0"}


def _assemble_pair_v0(row, tokens, tok, policy, markers):
    """Frozen copy of the pre-refactor assemble_pair (the golden reference)."""
    from vision_tokenization.discrete.conversation import apply_conversation_policy
    from vision_tokenization.discrete.sft_segments import _has_thinking_content, split_rendered_sft
    chosen, rejected = str(row["chosen"]), str(row["rejected"])
    images = list(row["images"])
    et = _has_thinking_content([{"role": "assistant", "content": chosen}])
    norm_prompt = apply_conversation_policy([dict(m) for m in row["prompt"]], policy)
    rendered = tok.apply_chat_template(norm_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=et)
    present = [m for m in markers if m in rendered]
    chunks, image_offsets, image_lengths = [], [], []
    cursor, img_i = 0, 0
    for seg in split_rendered_sft(rendered, present):
        if seg["type"] == "text":
            if not seg["text"]:
                continue
            arr = np.asarray(tok(seg["text"], add_special_tokens=False)["input_ids"], dtype=np.int32)
        else:
            im = images[img_i]
            img_i += 1
            off, ln = int(im["token_offset"]), int(im["token_length"])
            arr = np.asarray(tokens[off:off + ln], dtype=np.int32)
            image_offsets.append(cursor)
            image_lengths.append(int(len(arr)))
        chunks.append(arr)
        cursor += int(len(arr))
    if img_i != len(images):
        raise ValueError("img segments != image refs")
    prompt_ids = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int32)
    prompt_text_ids = tok(rendered, add_special_tokens=False)["input_ids"]
    p = len(prompt_text_ids)

    def _response_ids(answer):
        full = tok.apply_chat_template(
            norm_prompt + [{"role": "assistant", "content": answer}],
            tokenize=True, add_generation_prompt=False, enable_thinking=et)
        if full[:p] != prompt_text_ids:
            raise ValueError("prompt prefix not stable")
        return np.asarray(full[p:], dtype=np.int32)

    chosen_ids, rejected_ids = _response_ids(chosen), _response_ids(rejected)
    doc = np.concatenate([prompt_ids, chosen_ids, rejected_ids])
    pl, cl, rl = int(len(prompt_ids)), int(len(chosen_ids)), int(len(rejected_ids))
    return doc, {
        "prompt_id": row["prompt_id"], "prompt_len": pl, "chosen_len": cl, "rejected_len": rl,
        "image_offsets": image_offsets, "image_lengths": image_lengths,
        "seq_chosen_len": pl + cl, "seq_rejected_len": pl + rl, "image_tok": int(sum(image_lengths)),
    }


def test_assemble_pair_byte_identical_no_image(tok, policy_markers):
    from vision_tokenization.discrete.dpo_pairs import assemble_pair, pair_lengths, tokenize_pair_text
    policy, markers = policy_markers
    tokens = np.arange(64, dtype=np.int32)
    row = _row("What is 2+2?", "The answer is 4.", "It is 5.", [])

    doc, idx = assemble_pair(row, tokens, tok, policy, markers)
    v0_doc, v0_idx = _assemble_pair_v0(row, tokens, tok, policy, markers)
    assert np.array_equal(doc, v0_doc)
    assert idx == v0_idx
    assert idx["seq_chosen_len"] == idx["prompt_len"] + idx["chosen_len"]
    assert idx["seq_rejected_len"] == idx["prompt_len"] + idx["rejected_len"]
    # merge-side reproduces the lengths with no vision store
    tp = tokenize_pair_text(row, tok, policy, markers)
    assert pair_lengths(tp, [], prompt_id="p0") == idx


def test_assemble_pair_byte_identical_one_image(tok, policy_markers):
    from vision_tokenization.discrete.dpo_pairs import assemble_pair, pair_lengths, tokenize_pair_text
    policy, markers = policy_markers
    tokens = np.arange(100, dtype=np.int32)
    images = [{"media_id": "m", "token_offset": 10, "token_length": 5}]
    row = _row("<|image|>\nDescribe the image.", "A cat.", "A dog.", images)

    doc, idx = assemble_pair(row, tokens, tok, policy, markers)
    v0_doc, v0_idx = _assemble_pair_v0(row, tokens, tok, policy, markers)
    assert np.array_equal(doc, v0_doc)
    assert idx == v0_idx
    assert idx["image_lengths"] == [5] and idx["image_tok"] == 5
    # the spliced vision block (tokens[10:15]) sits at image_offsets[0] in the doc
    off = idx["image_offsets"][0]
    assert np.array_equal(doc[off:off + 5], tokens[10:15])
    # merge-side reproduces the lengths from the vision block length alone
    tp = tokenize_pair_text(row, tok, policy, markers)
    assert pair_lengths(tp, [5], prompt_id="p0") == idx


def test_enable_thinking_follows_chosen_only(tok, policy_markers):
    from vision_tokenization.discrete.dpo_pairs import tokenize_pair_text
    policy, markers = policy_markers
    think = "<think>let me reason</think>\nThe answer is 4."
    plain = "The answer is 4."

    chosen_thinks = tokenize_pair_text(_row("Q?", think, plain, []), tok, policy, markers)
    assert chosen_thinks.enable_thinking is True

    only_rejected_thinks = tokenize_pair_text(_row("Q?", plain, think, []), tok, policy, markers)
    assert only_rejected_thinks.enable_thinking is False
