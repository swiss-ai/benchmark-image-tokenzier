"""Shared mixins for EMU tokenizers with parallel GPU/CPU tokenization."""

from concurrent.futures import ThreadPoolExecutor
from typing import List

import torch


class ThreadPoolExecutorOwner:
    """Owns a 2-worker ThreadPoolExecutor and the shared flat-text tokenizer.

    Used as a mixin by EMU tokenizer classes that overlap image GPU encoding
    with CPU-side text tokenization. Provides:

    - the executor + context-manager + best-effort destructor so the pool is
      cleaned up even when the tokenizer is constructed directly (e.g. by
      ``executor.py``) without a ``with`` block;
    - ``_tokenize_flat_texts_cpu`` to batch-tokenize a list of pre-extracted
      text spans into individual tensors without adding BOS/EOS wrappers.

    Expects ``self.text_tokenizer`` to be provided by the co-inherited
    ``EMUImageOnlyTokenizer`` parent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="TokenizerPool",
        )

    def close(self) -> None:
        executor = getattr(self, "executor", None)
        if executor is None:
            return
        executor.shutdown(wait=True)
        self.executor = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):  # pragma: no cover — best-effort; executor.py constructs us directly without `with`.
        try:
            self.close()
        except Exception:
            pass

    def _tokenize_flat_texts_cpu(self, flat_texts: List[str]) -> List[torch.Tensor]:
        """Batch-tokenize text spans on CPU without adding BOS/EOS wrappers."""
        if not flat_texts:
            return []
        encoded = self.text_tokenizer(
            flat_texts,
            truncation=False,
            add_special_tokens=False,
            return_tensors=None,
            padding=False,
        )
        return [torch.tensor(ids, dtype=torch.long) for ids in encoded["input_ids"]]
