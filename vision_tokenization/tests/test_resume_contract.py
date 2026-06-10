"""Crash-resume contract: no loss, no duplication, loud refusal on mismatch.

Covers the protocol that lost data was previously only protected by luck:
writer-owned cursor state, the v2 checkpoint schema, legacy translation,
split-checkpoint refusal, and the plan fingerprint guard.
"""

import pytest
import torch

from vision_tokenization.pipeline.runtime.checkpoint import (
    WorkerStats,
    load_checkpoint,
    save_checkpoint,
    verify_plan_fingerprint,
)
from vision_tokenization.pipeline.output.direct.writer import MicroShardWriter


class _FakeTok:
    text_tokenizer = list(range(200_000))
    vision_token_offset = 100_000


def _write(writer, seqs):
    for s in seqs:
        writer.write_sequence(torch.tensor(s, dtype=torch.int32), WorkerStats())


def _read_all(tmp_path):
    try:
        from megatron.core.datasets.indexed_dataset import IndexedDataset
    except ImportError:
        pytest.skip("megatron not available")
    out = []
    for f in sorted(tmp_path.glob("rank_0000_chunk_*.bin")):
        ds = IndexedDataset(str(f)[:-4])
        out.extend(ds[i].tolist() for i in range(len(ds)))
    return out


SEQS = [[i, i + 1, i + 2] for i in range(0, 30, 3)]  # 10 deterministic seqs


def test_crash_resume_no_loss_no_duplication(tmp_path):
    w = MicroShardWriter()
    w.setup_writer(str(tmp_path), 0, 0, _FakeTok())
    _write(w, SEQS[:5])
    state = w.checkpoint_writer()           # chunk 0 finalized, state persisted
    _write(w, SEQS[5:8])                    # chunk 1 in flight ...
    # ... crash: no finalize, .tmp chunk 1 left behind

    w2 = MicroShardWriter()
    w2.setup_writer(str(tmp_path), 0, MicroShardWriter.resume_chunk(state), _FakeTok())
    _write(w2, SEQS[5:])                    # deterministic replay of 5..9
    w2.finalize_writer()

    assert _read_all(tmp_path) == SEQS      # exactly once, in order


def test_crash_after_finalize_before_checkpoint_overwrites_cleanly(tmp_path):
    w = MicroShardWriter()
    w.setup_writer(str(tmp_path), 0, 0, _FakeTok())
    _write(w, SEQS[:5])
    state = w.checkpoint_writer()
    _write(w, SEQS[5:8])
    w.checkpoint_writer()                   # chunk 1 finalized — but crash before save_checkpoint

    w2 = MicroShardWriter()                 # resume from the OLD state
    w2.setup_writer(str(tmp_path), 0, MicroShardWriter.resume_chunk(state), _FakeTok())
    _write(w2, SEQS[5:])                    # rewrites chunk 1 with a superset
    w2.finalize_writer()

    assert _read_all(tmp_path) == SEQS      # overwrite, not duplicate


def test_v2_checkpoint_roundtrip(tmp_path):
    fp = {"manifest_fingerprint": "abc", "total_batches": 7, "total_tokens": 99}
    save_checkpoint(str(tmp_path), 0, batch_index=42, writer_state={"chunk_id": 3},
                    plan_fingerprint=fp, stats={"tokens_generated": 1}, world_size=4)
    ckpt = load_checkpoint(str(tmp_path), 0)
    assert ckpt["version"] == 2
    assert ckpt["writer"] == {"chunk_id": 3}
    assert ckpt["plan"] == fp
    assert ckpt["batch_index"] == 42


def test_legacy_int_checkpoint_translated(tmp_path):
    torch.save({"batch_index": 9, "chunk_id": 4, "stats": {}, "world_size": 1},
               tmp_path / "rank_0000_checkpoint.pt")
    ckpt = load_checkpoint(str(tmp_path), 0)
    assert ckpt["writer"] == {"chunk_id": 4}
    assert ckpt["plan"] is None             # legacy → fingerprint check skipped
    assert MicroShardWriter.resume_chunk(ckpt["writer"]) == 5


def test_legacy_split_checkpoint_refused(tmp_path):
    torch.save({"batch_index": 9, "chunk_id": (3, 1), "stats": {}, "world_size": 1},
               tmp_path / "rank_0000_checkpoint.pt")
    with pytest.raises(RuntimeError, match="split-mode"):
        load_checkpoint(str(tmp_path), 0)


def test_fingerprint_mismatch_refuses(tmp_path):
    fp = {"manifest_fingerprint": "abc", "total_batches": 7, "total_tokens": 99}
    save_checkpoint(str(tmp_path), 0, batch_index=1, writer_state={"chunk_id": 0},
                    plan_fingerprint=fp, stats={}, world_size=1)
    ckpt = load_checkpoint(str(tmp_path), 0)
    verify_plan_fingerprint(ckpt, fp, rank=0)            # match: accepted
    with pytest.raises(RuntimeError, match="no longer matches"):
        verify_plan_fingerprint(ckpt, {**fp, "total_batches": 8}, rank=0)
    ckpt["plan"] = None                                  # legacy: always accepted
    verify_plan_fingerprint(ckpt, fp, rank=0)


class TestWorldSizeGuard:
    """One output_dir = one run configuration: a world-size change re-splits
    the plan, so resuming into the same directory must fail fast with the
    exact resubmit size."""

    def test_clean_and_matching_dirs_pass(self, tmp_path):
        from vision_tokenization.pipeline.runtime.checkpoint import verify_run_world_size
        verify_run_world_size(str(tmp_path), world_size=4, rank=0)  # empty: ok
        save_checkpoint(str(tmp_path), 0, batch_index=1, writer_state={"chunk_id": 0},
                        plan_fingerprint=None, stats={}, world_size=4)
        verify_run_world_size(str(tmp_path), world_size=4, rank=0)  # same size: ok

    def test_larger_to_smaller_refused_with_resubmit_size(self, tmp_path):
        from vision_tokenization.pipeline.runtime.checkpoint import verify_run_world_size
        save_checkpoint(str(tmp_path), 7, batch_index=3, writer_state={"chunk_id": 0},
                        plan_fingerprint=None, stats={}, world_size=8)
        with pytest.raises(RuntimeError, match="num_gpus=8"):
            verify_run_world_size(str(tmp_path), world_size=4, rank=0)

    def test_smaller_to_larger_refused_with_resubmit_size(self, tmp_path):
        from vision_tokenization.pipeline.runtime.checkpoint import verify_run_world_size
        save_checkpoint(str(tmp_path), 0, batch_index=1, writer_state={"chunk_id": 0},
                        plan_fingerprint=None, stats={}, world_size=4)
        with pytest.raises(RuntimeError, match="num_gpus=4"):
            verify_run_world_size(str(tmp_path), world_size=8, rank=0)

    def test_shard_only_stale_rank_refused(self, tmp_path):
        from vision_tokenization.pipeline.runtime.checkpoint import verify_run_world_size
        (tmp_path / "rank_0005_chunk_0000.bin").touch()  # no checkpoint at all
        with pytest.raises(RuntimeError, match="num_gpus=6"):
            verify_run_world_size(str(tmp_path), world_size=4, rank=0)

    def test_merge_gate_refuses_stale_ranks(self, tmp_path):
        from vision_tokenization.pipeline.output.merge import _stale_ranks
        for r in range(4):  # new 4-rank generation
            save_checkpoint(str(tmp_path), r, batch_index=1, writer_state={"chunk_id": 0},
                            plan_fingerprint=None, stats={}, world_size=4)
        assert _stale_ranks(tmp_path, set(range(8))) == [4, 5, 6, 7]
        assert _stale_ranks(tmp_path, set(range(4))) == []
