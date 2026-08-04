"""rl_prompt task fork — the thin parallel to test_dpo_pairs.py. Asserts: (1) the rl
parser keeps prompt+answer+answer_variants, ALLOWS a system turn, and enforces the
marker-count / marker-in-answer checks; (2) ``tokenize_rl_prompt_text`` pins
``enable_thinking`` True and keeps the prompt-prefix stability guard; (3)
``assemble_prompt_only`` reuses ``pair_lengths`` to build a [prompt]-only doc whose
length equals ``prompt_len`` and whose vision splice lands at ``image_offsets``; (4)
the rl index row carries exactly the DELTA-3 fields.

The parser tests are pure (no tokenizer). The tokenize/assemble tests need
transformers + the Apertus tokenizer and skip otherwise, like test_dpo_pairs.py.
The index-schema field check needs torch (megatron import) and skips under pa20.
"""

import os

import numpy as np
import pytest

from vision_tokenization.indexing.alignment.ingest import (
    MARKER,
    MarkerMismatch,
    _parse_rl_prompt_row_with_refs,
)

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


def _src_row(prompt, answer, *, answer_variants=None, source_id="p0"):
    """An rl_prompt source row as the converter emits it (pre-normalization): the
    prompt content carries the dataset-level ``<image>`` marker, not ``<|image|>``."""
    row = {"prompt": prompt, "answer": answer, "source-id": source_id}
    if answer_variants is not None:
        row["answer_variants"] = answer_variants
    return row


def _prompt_row(prompt_text):
    """A post-parse rl row (prompt content already normalized to ``<|image|>``)."""
    return {"prompt": [{"role": "user", "content": prompt_text}], "prompt_id": "p0"}


# --- parser (pure; no tokenizer) ------------------------------------------------

def test_parser_keeps_prompt_answer_and_variants():
    row = _src_row(
        [{"role": "user", "content": "<image>\nWhat is in the picture?"}],
        "a cat",
        answer_variants=["a cat", "cat", "feline"],
    )
    out = _parse_rl_prompt_row_with_refs(row, ["m0"])
    assert out["prompt_id"] == "p0"
    assert out["answer"] == "a cat"
    assert out["answer_variants"] == ["a cat", "cat", "feline"]
    # the <image> marker is rewritten to the canonical <|image|>
    assert "<image>" not in out["prompt"][0]["content"]
    assert out["prompt"][0]["content"].count(MARKER) == 1
    # C1: prompt-only media refs; chosen/rejected empty
    assert out["prompt_media_refs"] == ["m0"]
    assert out["chosen_media_refs"] == [] and out["rejected_media_refs"] == []
    # answer / enable_thinking are NOT parser output (view-/index-carriers)
    assert "chosen" not in out and "rejected" not in out and "enable_thinking" not in out


def test_parser_defaults_variants_to_empty():
    row = _src_row([{"role": "user", "content": "no image here"}], "42")
    out = _parse_rl_prompt_row_with_refs(row, [])
    assert out["answer"] == "42"
    assert out["answer_variants"] == []
    assert out["prompt_media_refs"] == []


def test_parser_allows_system_turn():
    """Real RL data keeps a system instruction (DeepVision: 'answer in \\boxed{}').
    Preference rejects system; the rl path must let it through (the template renders it)."""
    row = _src_row(
        [
            {"role": "system", "content": "Answer the question in \\boxed{}."},
            {"role": "user", "content": "<image>\nCompute the value."},
        ],
        "\\boxed{7}",
    )
    out = _parse_rl_prompt_row_with_refs(row, ["m0"])
    assert [m["role"] for m in out["prompt"]] == ["system", "user"]
    assert out["prompt"][0]["content"] == "Answer the question in \\boxed{}."


def test_parser_marker_count_mismatch_raises():
    row = _src_row([{"role": "user", "content": "<image>\n<image>\nq"}], "ans")
    with pytest.raises(MarkerMismatch, match="markers vs"):
        _parse_rl_prompt_row_with_refs(row, ["m0"])


def test_parser_rejects_marker_in_answer():
    row = _src_row([{"role": "user", "content": "<image>\nq"}], f"sneaky {MARKER} answer")
    with pytest.raises(MarkerMismatch, match="accidental marker in answer"):
        _parse_rl_prompt_row_with_refs(row, ["m0"])


def test_parser_rejects_accidental_marker_in_prompt():
    """A bare <|image|> already in the source prompt (not from <image> rewrite) is a
    smuggled marker — the shared _normalize guard catches it on the rl path too."""
    row = _src_row([{"role": "user", "content": f"text {MARKER} text"}], "ans")
    with pytest.raises(MarkerMismatch, match="accidental"):
        _parse_rl_prompt_row_with_refs(row, [])


# --- tokenize_rl_prompt_text (needs the real tokenizer) -------------------------

def test_tokenize_rl_prompt_enable_thinking_always_true(tok, policy_markers):
    """enable_thinking is a task constant (True), never read from the row — the render
    must always emit the 'Deliberation: enabled' developer block. Contrast with the
    preference path, where it follows the chosen response."""
    from vision_tokenization.discrete.conversation import apply_conversation_policy
    policy, markers = policy_markers

    tp = tokenize_helper(tok, policy, markers, "What is 2+2?")
    assert tp.image_insert_positions == []

    # the prompt-prefix is rendered with enable_thinking=True; compare against the
    # explicit True render and confirm it differs from the False render (the flag bites).
    norm = apply_conversation_policy([{"role": "user", "content": "What is 2+2?"}], policy)
    render_true = tok.apply_chat_template(norm, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    render_false = tok.apply_chat_template(norm, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert render_true != render_false
    # with no images the marker-free render equals the whole render
    assert tp.prompt_text_ids.tolist() == tok(render_true, add_special_tokens=False)["input_ids"]


def test_tokenize_rl_prompt_prefix_stability_one_image(tok, policy_markers):
    """With an image slot, the segment-wise text concat must still reconstruct the
    whole marker-free render — the load-bearing guard that the stored [prompt] tail is
    what an autoregressive rollout conditions on."""
    policy, markers = policy_markers
    tp = tokenize_helper(tok, policy, markers, f"{MARKER}\nDescribe the image.")
    assert len(tp.image_insert_positions) == 1
    # the slot index is a valid position inside prompt_text_ids (text-only stream)
    assert 0 <= tp.image_insert_positions[0] <= len(tp.prompt_text_ids)


def test_tokenize_rl_prompt_stability_guard_fires_on_drift(tok, policy_markers):
    """Force a BPE-boundary drift: a tokenizer whose whole-render tokenization differs
    from the per-segment concat. The guard (no response to catch it, unlike DPO) must
    raise rather than store a [prompt] tail an autoregressive rollout can't reproduce."""
    from vision_tokenization.discrete.dpo_pairs import tokenize_rl_prompt_text
    policy, markers = policy_markers

    seg_texts: list[str] = []

    class _Drift:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __call__(self, text, *args, **kwargs):
            out = self._inner(text, *args, **kwargs)
            # render_prompt_prefix tokenizes each text span first; the guard then
            # tokenizes the whole render WITH markers — perturb only that last call.
            if MARKER in text:
                out["input_ids"] = out["input_ids"][:-1]
            else:
                seg_texts.append(text)
            return out

    with pytest.raises(ValueError, match="prompt prefix not stable"):
        tokenize_rl_prompt_text(
            _prompt_row(f"{MARKER}\nDescribe the image."), _Drift(tok), policy, markers)


def test_tokenize_rl_prompt_tight_marker_no_false_fail(tok, policy_markers):
    """Tight inline markers (text \\n immediately before and after <|image|>, no blank
    line) once false-failed the guard: dropping the marker let the two \\n BPE-merge into
    one \\n\\n token, so the marker-free join drifted from the segment-wise concat. The
    guard now compares against the whole render minus the image token, so it passes."""
    policy, markers = policy_markers
    tp = tokenize_helper(tok, policy, markers, f"AB = ( ) m\n{MARKER}\nChoices:\nA. 300")
    assert len(tp.image_insert_positions) == 1
    assert 0 <= tp.image_insert_positions[0] <= len(tp.prompt_text_ids)


# --- assemble_prompt_only (needs the real tokenizer) ----------------------------

def test_assemble_prompt_only_no_image(tok, policy_markers):
    """[prompt]-only doc: with no images the doc IS prompt_text_ids and
    prompt_len == len(doc); the index_row carries no response lengths."""
    from vision_tokenization.discrete.dpo_pairs import assemble_prompt_only, pair_lengths
    policy, markers = policy_markers
    tokens = np.arange(64, dtype=np.int32)
    tp = tokenize_helper(tok, policy, markers, "What is the capital of France?")

    doc, irow = assemble_prompt_only(tp, [], tokens, prompt_id="p0")
    assert np.array_equal(doc, tp.prompt_text_ids)
    assert irow["prompt_len"] == len(doc)
    assert irow["image_offsets"] == [] and irow["image_lengths"] == [] and irow["image_tok"] == 0
    # pure-prompt index row keys (answer/enable_thinking added later at binidx)
    assert set(irow) == {"prompt_id", "prompt_len", "image_offsets", "image_lengths", "image_tok"}
    # reuses pair_lengths (chosen/rejected default 0)
    pl = pair_lengths(tp, [], prompt_id="p0")
    assert (pl["prompt_len"], pl["image_offsets"], pl["image_tok"]) == (
        irow["prompt_len"], irow["image_offsets"], irow["image_tok"])
    assert pl["chosen_len"] == 0 and pl["rejected_len"] == 0


def test_assemble_prompt_only_one_image(tok, policy_markers):
    """The spliced vision block lands at image_offsets[0]; the whole doc is the prompt
    and prompt_len == len(doc) == text + inlined vision."""
    from vision_tokenization.discrete.dpo_pairs import assemble_prompt_only, prompt_from_row, prompt_to_row
    policy, markers = policy_markers
    tokens = np.arange(100, dtype=np.int32)
    tp = tokenize_helper(tok, policy, markers, f"{MARKER}\nDescribe the image.")
    images = [{"media_id": "m", "token_offset": 10, "token_length": 5}]

    doc, irow = assemble_prompt_only(tp, images, tokens, prompt_id="p0")
    assert irow["prompt_len"] == len(doc)
    assert irow["image_lengths"] == [5] and irow["image_tok"] == 5
    off = irow["image_offsets"][0]
    assert np.array_equal(doc[off:off + 5], tokens[10:15])
    # doc == [text-before | vision | text-after], no response span
    assert len(doc) == len(tp.prompt_text_ids) + 5

    # the spill round-trip preserves the TokenizedPrompt (prompt_from_row(prompt_to_row))
    rt = prompt_from_row(prompt_to_row(tp, "p0"))
    assert np.array_equal(rt.prompt_text_ids, tp.prompt_text_ids)
    assert rt.image_insert_positions == tp.image_insert_positions
    doc2, irow2 = assemble_prompt_only(rt, images, tokens, prompt_id="p0")
    assert np.array_equal(doc2, doc) and irow2 == irow


# --- rl index schema (DELTA-3) --------------------------------------------------

def test_rl_prompt_index_schema_fields():
    """DELTA-3: the rl index schema is exactly these fields. Importing the binidx
    module pulls in megatron (torch); skip where torch is absent (pa20)."""
    pytest.importorskip("torch")
    from vision_tokenization.pipeline.output.dpo_binidx import RL_PROMPT_INDEX_SCHEMA
    assert RL_PROMPT_INDEX_SCHEMA.names == [
        "prompt_id", "prompt_len", "image_offsets", "image_lengths",
        "image_tok", "answer", "answer_variants", "enable_thinking",
    ]
    types = {f.name: f.type for f in RL_PROMPT_INDEX_SCHEMA}
    import pyarrow as pa
    assert types["answer"] == pa.string()
    assert types["answer_variants"] == pa.list_(pa.string())
    assert types["enable_thinking"] == pa.bool_()


def tokenize_helper(tok, policy, markers, prompt_text):
    from vision_tokenization.discrete.dpo_pairs import tokenize_rl_prompt_text
    return tokenize_rl_prompt_text(_prompt_row(prompt_text), tok, policy, markers)
