"""The direct and spill backends must write identical bytes.

Which one runs is decided by a single line in the executor::

    use_spill = multi_image or mode in ("interleave", "alignment")

so the same mode takes a different assembler depending on a config flag. They
share the size formula (common.layout) but still lay out the tokens
independently — image_only builds inline from resolved ids, common.assembly
rebuilds from a StructureTokenIds bundle. Nothing else checks that they agree.

Assembly is tested directly rather than through the pipeline: the divergence is
in the layout, and both entry points take already-encoded codebook indices, so
no GPU or vision tokenizer is involved.
"""

import pytest
import torch

from vision_tokenization.common.assembly import (
    StructureTokenIds,
    encapsulate_image_structure,
    encapsulate_image_structure_batch,
)
from vision_tokenization.discrete.emu.image_only import EMUImageOnlyTokenizer

# Apertus 2 geometry: structure tokens sit low, inside the base vocab, and the
# vision band starts at base_vocab_size.
BOS, EOS = 1, 2
IMG_START, IMG_END, IMG_TOKEN_START, EOL, EOF = 27, 28, 29, 30, 31
VISION_OFFSET = 200064
DIM_TOKENS = {"2*3": [40, 41, 42], "4*4": [50, 51, 52, 53, 54]}


def _direct_assembler():
    """An EMUImageOnlyTokenizer with only the assembly attributes set.

    __init__ loads a text tokenizer and a vision model; neither is reachable
    from encapsulate_image, so the object is built without it.
    """
    tok = object.__new__(EMUImageOnlyTokenizer)
    tok.bos_id, tok.eos_id = BOS, EOS
    tok.img_start_id, tok.img_end_id = IMG_START, IMG_END
    tok.img_token_start_id = IMG_TOKEN_START
    tok.eol_id, tok.eof_id = EOL, EOF
    tok.vision_token_offset = VISION_OFFSET
    tok.dim_cache = dict(DIM_TOKENS)
    return tok


def _shared_ids():
    return StructureTokenIds(
        bos_id=BOS, eos_id=EOS,
        img_start_id=IMG_START, img_end_id=IMG_END,
        img_token_start_id=IMG_TOKEN_START,
        eol_id=EOL, eof_id=EOF,
        vision_token_offset=VISION_OFFSET,
        image_token_id=18,
        dim_tokens_fn=lambda h, w: DIM_TOKENS[f"{h}*{w}"],
    )


@pytest.mark.parametrize("h,w", [(2, 3), (4, 4)])
def test_direct_matches_spill(h, w):
    """encapsulate_image == BOS + encapsulate_image_structure + EOS."""
    indices = torch.arange(h * w, dtype=torch.long)

    direct = _direct_assembler().encapsulate_image(indices.clone(), h, w)
    block = encapsulate_image_structure(indices.clone(), h, w, _shared_ids())
    spill = torch.cat([torch.tensor([BOS]), block, torch.tensor([EOS])])

    assert direct.tolist() == spill.tolist()


@pytest.mark.parametrize("h,w", [(2, 3), (4, 4)])
def test_batch_paths_match_too(h, w):
    """The batched spill assembler must agree with the scalar one."""
    indices = torch.arange(h * w, dtype=torch.long)
    ids = _shared_ids()

    scalar = encapsulate_image_structure(indices.clone(), h, w, ids)
    batched = encapsulate_image_structure_batch(
        indices.clone().unsqueeze(0), h, w, ids)
    assert scalar.tolist() == batched.reshape(-1).tolist()


def test_the_layout_is_what_we_think_it_is():
    """Pin the wire format explicitly, so a change to both assemblers at once
    still has to be deliberate."""
    direct = _direct_assembler().encapsulate_image(
        torch.arange(6, dtype=torch.long), 2, 3)

    assert direct.tolist() == [
        BOS, IMG_START, 40, 41, 42, IMG_TOKEN_START,
        VISION_OFFSET + 0, VISION_OFFSET + 1, VISION_OFFSET + 2, EOL,
        VISION_OFFSET + 3, VISION_OFFSET + 4, VISION_OFFSET + 5, EOL,
        EOF, IMG_END, EOS,
    ]


def test_vision_ids_land_in_the_declared_band():
    """Codebook index k must become base_vocab_size + k, never a structure id."""
    h, w = 4, 4
    direct = _direct_assembler().encapsulate_image(
        torch.arange(h * w, dtype=torch.long), h, w)

    vision = [t for t in direct.tolist() if t >= VISION_OFFSET]
    assert vision == [VISION_OFFSET + k for k in range(h * w)]
    structure = {BOS, EOS, IMG_START, IMG_END, IMG_TOKEN_START, EOL, EOF}
    assert not structure & set(vision)
