"""MicroShardWriter — micro-shard lifecycle for distributed tokenization.

Manages IndexedDatasetBuilder for writing token sequences to Megatron
MMIDIDX ``.bin/.idx`` files.  Extracted from the former ``BaseHandler``.

Also provides ``SplitMicroShardWriter`` which routes sequences to two
sub-writers based on a token-length threshold (stage2 / lct).
"""

import logging
import os
from typing import Any, Dict, Tuple

from ...runtime.checkpoint import (
    WorkerStats,
    finalize_shard_writer,
    open_chunk_writer,
)
from ..provenance import sidecar_path

logger = logging.getLogger(__name__)


class MicroShardWriter:
    """Micro-shard writer for vision tokenization output.

    Provides the writer lifecycle (open / checkpoint / finalize) and a
    ``write_sequence`` helper that writes one token sequence and updates
    stats.
    """

    def __init__(self):
        self._builder = None
        self._tmp_bin = None
        self._tmp_idx = None
        self._bin_path = None
        self._idx_path = None
        self._output_dir = None
        self._rank = None
        self._chunk_id = None
        self._vocab_size = None
        self._vision_token_offset = None
        self._emit_prov = False
        self._src_buf = None
        self._src_path = None
        self.chunk_samples = 0

    def setup_writer(
        self, output_dir: str, rank: int, chunk_id: int, tokenizer, emit_prov: bool = False,
    ) -> None:
        """Open an IndexedDatasetBuilder for the current micro-shard."""
        self._output_dir = output_dir
        self._rank = rank
        self._chunk_id = chunk_id
        self._vocab_size = len(tokenizer.text_tokenizer)
        self._vision_token_offset = getattr(tokenizer, "vision_token_offset", None)
        self._emit_prov = emit_prov
        self._open_writer()

    def _open_writer(self):
        self._builder, self._tmp_bin, self._tmp_idx, self._bin_path, self._idx_path = (
            open_chunk_writer(self._output_dir, self._rank, self._chunk_id, self._vocab_size)
        )
        self.chunk_samples = 0
        # Per-chunk provenance buffer, reset in lockstep with chunk_samples so
        # one int64 source row is appended per written sequence (end_document()).
        # The sidecar path is derived once here so the ``.src.npy`` suffix lives
        # only in ``provenance.sidecar_path`` (``_bin_path`` ends with ".bin").
        self._src_buf = [] if self._emit_prov else None
        self._src_path = sidecar_path(self._bin_path[:-4]) if self._emit_prov else None

    def checkpoint_writer(self) -> int:
        """Finalize current chunk, open next. Returns finalized chunk_id."""
        finalize_shard_writer(
            self._builder, self._tmp_bin, self._tmp_idx, self._bin_path, self._idx_path,
            src_buf=self._src_buf, src_path=self._src_path,
        )
        done_chunk = self._chunk_id
        self._chunk_id += 1
        self._open_writer()
        return done_chunk

    def finalize_writer(self) -> None:
        """Finalize the last chunk (even if empty, for consistency)."""
        if self.chunk_samples > 0:
            finalize_shard_writer(
                self._builder, self._tmp_bin, self._tmp_idx, self._bin_path, self._idx_path,
                src_buf=self._src_buf, src_path=self._src_path,
            )
        else:
            for p in (self._tmp_bin, self._tmp_idx):
                if p and os.path.exists(p):
                    os.unlink(p)

    def write_sequence(self, seq_cpu, stats: WorkerStats, source_id=None) -> None:
        """Write one token sequence to the current micro-shard and update stats.

        *source_id* is the source manifest row of this sequence; it is recorded
        in the per-chunk provenance buffer only when provenance is enabled.
        """
        self._builder.add_item(seq_cpu)
        self._builder.end_document()
        n_tokens = seq_cpu.numel()
        stats.samples_processed += 1
        stats.tokens_generated += n_tokens
        if self._vision_token_offset is not None:
            img_tok = int((seq_cpu >= self._vision_token_offset).sum().item())
            stats.image_tokens += img_tok
            stats.text_tokens += n_tokens - img_tok
        else:
            stats.image_tokens += n_tokens
        self.chunk_samples += 1
        if self._src_buf is not None:
            self._src_buf.append(int(source_id))


class SplitMicroShardWriter:
    """Routes sequences to ``stage2/`` or ``lct/`` based on token length.

    Sequences with ``numel() <= seqlen_threshold`` go to ``stage2/``;
    longer sequences go to ``lct/``.  Each bucket has its own independent
    ``MicroShardWriter`` with its own chunk counter.
    """

    def __init__(self, seqlen_threshold: int):
        self._threshold = seqlen_threshold
        self._stage2 = MicroShardWriter()
        self._lct = MicroShardWriter()

    @property
    def chunk_samples(self) -> int:
        return self._stage2.chunk_samples + self._lct.chunk_samples

    def setup_writer(
        self, output_dir: str, rank: int,
        stage2_chunk_id: int, lct_chunk_id: int, tokenizer, emit_prov: bool = False,
    ) -> None:
        os.makedirs(os.path.join(output_dir, "stage2"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "lct"), exist_ok=True)
        self._stage2.setup_writer(
            os.path.join(output_dir, "stage2"), rank, stage2_chunk_id, tokenizer,
            emit_prov=emit_prov,
        )
        self._lct.setup_writer(
            os.path.join(output_dir, "lct"), rank, lct_chunk_id, tokenizer,
            emit_prov=emit_prov,
        )

    def write_sequence(self, seq_cpu, stats: WorkerStats, source_id=None) -> None:
        n = seq_cpu.numel()
        if n <= self._threshold:
            self._stage2.write_sequence(seq_cpu, stats, source_id=source_id)
            stats.stage2_tokens += n
            stats.stage2_samples += 1
        else:
            self._lct.write_sequence(seq_cpu, stats, source_id=source_id)
            stats.lct_tokens += n
            stats.lct_samples += 1

    def checkpoint_writer(self) -> Tuple[int, int]:
        """Finalize sub-writers that have samples, return (stage2_chunk, lct_chunk)."""
        s2 = (
            self._stage2.checkpoint_writer()
            if self._stage2.chunk_samples > 0
            else self._stage2._chunk_id - 1
        )
        lct = (
            self._lct.checkpoint_writer()
            if self._lct.chunk_samples > 0
            else self._lct._chunk_id - 1
        )
        return (s2, lct)

    def finalize_writer(self) -> None:
        self._stage2.finalize_writer()
        self._lct.finalize_writer()

    @property
    def stage2_chunk_id(self) -> int:
        return self._stage2._chunk_id

    @property
    def lct_chunk_id(self) -> int:
        return self._lct._chunk_id

    @property
    def stage2_chunk_samples(self) -> int:
        return self._stage2.chunk_samples

    @property
    def lct_chunk_samples(self) -> int:
        return self._lct.chunk_samples
