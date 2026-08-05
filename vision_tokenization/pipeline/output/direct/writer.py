"""MicroShardWriter — micro-shard lifecycle for distributed tokenization.

Manages IndexedDatasetBuilder for writing token sequences to Megatron
MMIDIDX ``.bin/.idx`` files.  Extracted from the former ``BaseHandler``.

"""

import logging
import os

from ...runtime.checkpoint import (
    WorkerStats,
    finalize_shard_writer,
    open_chunk_writer,
)

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
        self.chunk_samples = 0
        self.chunk_tokens = 0
        # Finalized-chunk records {name, bytes, sequences, tokens}: the
        # writer's own truth of what it shipped, persisted in writer_state
        # across resumes and published in the rank completion manifest.
        self.finalized_files = []

    def setup_writer(self, output_dir: str, rank: int, chunk_id: int, tokenizer) -> None:
        """Open an IndexedDatasetBuilder for the current micro-shard."""
        self._output_dir = output_dir
        self._rank = rank
        self._chunk_id = chunk_id
        self._vocab_size = len(tokenizer.text_tokenizer)
        self._vision_token_offset = getattr(tokenizer, "vision_token_offset", None)
        self._open_writer()

    def _open_writer(self):
        self._builder, self._tmp_bin, self._tmp_idx, self._bin_path, self._idx_path = (
            open_chunk_writer(self._output_dir, self._rank, self._chunk_id, self._vocab_size)
        )
        self.chunk_samples = 0
        self.chunk_tokens = 0

    def restore(self, writer_state: dict) -> None:
        """Restore writer-owned state (finalized-file records) on resume."""
        self.finalized_files = list(writer_state.get("files", []))

    def _record_finalized_chunk(self) -> None:
        self.finalized_files.append({
            "name": os.path.basename(self._bin_path),
            "bytes": os.path.getsize(self._bin_path),
            "sequences": self.chunk_samples,
            "tokens": self.chunk_tokens,
        })

    def checkpoint_writer(self) -> dict:
        """Finalize current chunk, open next. Returns the writer's resume state."""
        finalize_shard_writer(
            self._builder, self._tmp_bin, self._tmp_idx, self._bin_path, self._idx_path
        )
        self._record_finalized_chunk()
        done_chunk = self._chunk_id
        self._chunk_id += 1
        self._open_writer()
        return {"chunk_id": done_chunk, "files": self.finalized_files}

    @staticmethod
    def resume_chunk(writer_state: dict) -> int:
        """First chunk to (re)write given a checkpointed writer state.

        The chunk after the last finalized one is reopened and fully
        overwritten — together with the deterministic plan this gives
        exactly-once output across crash/resume.
        """
        return int(writer_state["chunk_id"]) + 1

    def finalize_writer(self) -> None:
        """Finalize the last chunk (even if empty, for consistency)."""
        if self.chunk_samples > 0:
            finalize_shard_writer(
                self._builder, self._tmp_bin, self._tmp_idx, self._bin_path, self._idx_path
            )
            self._record_finalized_chunk()
        else:
            for p in (self._tmp_bin, self._tmp_idx):
                if p and os.path.exists(p):
                    os.unlink(p)

    def write_sequence(self, seq_cpu, stats: WorkerStats) -> None:
        """Write one token sequence to the current micro-shard and update stats."""
        self._builder.add_item(seq_cpu)
        self._builder.end_document()
        n_tokens = seq_cpu.numel()
        self.chunk_tokens += n_tokens
        stats.samples_processed += 1
        stats.tokens_generated += n_tokens
        if self._vision_token_offset is not None:
            img_tok = int((seq_cpu >= self._vision_token_offset).sum().item())
            stats.image_tokens += img_tok
            stats.text_tokens += n_tokens - img_tok
        else:
            stats.image_tokens += n_tokens
        self.chunk_samples += 1
