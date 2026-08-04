"""Strip <think>...</think> spans from an already-merged Megatron MMIDIDX dataset.

For a `dataset.{bin,idx}` pair, produces `dataset_no_cot.{bin,idx}` in the same
directory (or a custom output path). Reuses the production strip implementation
from `vision_tokenization.pipeline.output.merge` — no logic duplication.

When to use:
- An SFT dataset was merged with CoT preserved (`STRIP_THINKING=false`) and you
  later want a no-CoT capability-filtering variant without re-running the merge.
- A dataset's CoT is uniform/universal (e.g. `google_rsrcc`) and stripping
  leaves just the final answer.
- A dataset has *partial* CoT (e.g. `sensenova_si_8m`, ~0.5%) and you want a
  guaranteed-clean variant for capability filtering.

Usage:
    # Default: writes the stripped variant into
    # `Apertus1p5_sft_capability_filtering/{basename}.{bin,idx}`
    # (canonical `Apertus1p5_sft_tokenized/` is never modified).
    python -m scripts.strip_cot \\
        /capstor/.../Apertus1p5_sft_tokenized/google_rsrcc

    # Custom output:
    python -m scripts.strip_cot \\
        /path/to/foo --output /path/to/bar

The merge's CLI `--strip-thinking` operates on per-rank shards mid-merge; this
script is the post-hoc analogue for an already-merged dataset.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make sibling `vision_tokenization` package importable when run as a script.
# Megatron path is handled by merge._ensure_megatron_importable() at call time.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision_tokenization.pipeline.output.merge import strip_thinking_dataset


# Default output dir for stripped variants. Lands directly in
# capability_filtering so the file is usable without extra symlinking; the
# canonical `Apertus1p5_sft_tokenized/` dir is never modified.
_DEFAULT_OUTPUT_DIR = (
    "/capstor/store/cscs/swissai/infra01/vision-datasets/"
    "Apertus1p5_sft_capability_filtering"
)


def strip_cot(
    input_prefix: str,
    output_prefix: str | None = None,
    *,
    think_id: int,
    end_think_id: int,
) -> str:
    """Strip `<think>...</think>` from a merged dataset; return output prefix.

    Args:
        input_prefix: path without `.bin`/`.idx` extension.
        output_prefix: where to write. Defaults to
            `Apertus1p5_sft_capability_filtering/{basename}.bin/.idx` —
            stripped variants live alongside the symlinked no-CoT-native
            datasets in the capability-filtering dir. The canonical
            `Apertus1p5_sft_tokenized/` dir is never modified.
        think_id: token ID opening a reasoning span,
            in the tokenizer that produced this dataset.
        end_think_id: token ID closing it.

    Returns the output prefix.
    """
    input_prefix = str(input_prefix).rstrip("/")
    if output_prefix is None:
        os.makedirs(_DEFAULT_OUTPUT_DIR, exist_ok=True)
        basename = os.path.basename(input_prefix)
        output_prefix = os.path.join(_DEFAULT_OUTPUT_DIR, basename)

    stats = strip_thinking_dataset(input_prefix, output_prefix,
                                   think_id, end_think_id)
    print(f"input:    {stats.input_count:,} sequences")
    print(f"written:  {stats.written_count:,} sequences")
    print(f"skipped:  {stats.skipped_count:,} sequences (empty after strip)")
    print(f"tokens:   {stats.output_tokens:,} tokens written")
    print(f"output:   {output_prefix}.bin / .idx")
    return output_prefix


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_prefix", help="Dataset prefix (no .bin/.idx suffix).")
    p.add_argument("--output", default=None, help="Output prefix (default: {input}_no_cot).")
    p.add_argument("--think-id", type=int, required=True,
                   help="Reasoning-span opening token ID in the tokenizer that "
                        "produced this dataset")
    p.add_argument("--end-think-id", type=int, required=True,
                   help="Reasoning-span closing token ID")
    args = p.parse_args(argv)

    if not os.path.exists(args.input_prefix + ".bin"):
        sys.stderr.write(f"error: {args.input_prefix}.bin not found\n")
        return 1

    strip_cot(
        args.input_prefix, args.output,
        think_id=args.think_id, end_think_id=args.end_think_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
