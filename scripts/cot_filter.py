"""Filter a tokenized dataset to ONLY the sequences containing a given token id
(default <think>=32), writing them to a new prefix. Built on rewrite_dataset
(same primitive as bucket_by_length) — reads input once, keeps matching seqs.

Usage:
  python scripts/cot_filter.py --input <prefix> --output <prefix> [--token 32]
"""
import argparse, sys
import numpy as np

sys.path.insert(0, "/iopsstor/scratch/cscs/xyixuan/apertus/Megatron-LM")
sys.path.insert(0, "/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier")
from vision_tokenization.pipeline.output.merge import rewrite_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input prefix (no .bin/.idx)")
    ap.add_argument("--output", required=True, help="output prefix (no .bin/.idx)")
    ap.add_argument("--token", type=int, required=True,
                    help="token id to require, in the tokenizer that produced this dataset")
    a = ap.parse_args()
    tok = a.token

    def keep(seq):
        return seq if np.any(np.asarray(seq) == tok) else None

    st = rewrite_dataset(a.input, a.output, keep)
    print(f"cot_filter token={tok}: input={st.input_count:,} written={st.written_count:,} "
          f"dropped={st.skipped_count:,} output_tokens={st.output_tokens:,}", flush=True)


if __name__ == "__main__":
    main()
