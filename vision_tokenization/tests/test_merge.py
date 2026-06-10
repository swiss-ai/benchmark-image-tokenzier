"""Tests for merge utilities: strip_thinking_tokens and rewrite_dataset."""

from pathlib import Path

import os
import numpy as np
import pytest

from vision_tokenization.pipeline.output.merge import (
    _ensure_megatron_importable,
    merge_shards,
    rewrite_dataset,
    strip_thinking_tokens,
)

_ensure_megatron_importable()

# ---------------------------------------------------------------------------
# strip_thinking_tokens — unit tests
# ---------------------------------------------------------------------------

THINK = 32
END_THINK = 33


class TestStripThinkingTokens:
    def test_basic(self):
        tokens = np.array([1, THINK, 99, 100, END_THINK, 2], dtype=np.int32)
        result = strip_thinking_tokens(tokens)
        np.testing.assert_array_equal(result, [1, 2])

    def test_no_think_tokens(self):
        tokens = np.array([1, 2, 3], dtype=np.int32)
        result = strip_thinking_tokens(tokens)
        assert result is tokens  # exact same object, no copy

    def test_multiple_spans(self):
        tokens = np.array(
            [1, THINK, 99, END_THINK, 5, THINK, 88, END_THINK, 2], dtype=np.int32
        )
        result = strip_thinking_tokens(tokens)
        np.testing.assert_array_equal(result, [1, 5, 2])

    def test_unmatched_open(self):
        tokens = np.array([1, THINK, 99, 100], dtype=np.int32)
        result = strip_thinking_tokens(tokens)
        np.testing.assert_array_equal(result, [1])

    def test_orphan_close(self):
        tokens = np.array([END_THINK, 1, 2], dtype=np.int32)
        result = strip_thinking_tokens(tokens)
        np.testing.assert_array_equal(result, [1, 2])

    def test_all_thinking(self):
        tokens = np.array([THINK, 99, END_THINK], dtype=np.int32)
        result = strip_thinking_tokens(tokens)
        assert result is None

    def test_empty_input(self):
        tokens = np.array([], dtype=np.int32)
        result = strip_thinking_tokens(tokens)
        assert result is None

    def test_adjacent_spans(self):
        tokens = np.array([THINK, END_THINK, THINK, END_THINK], dtype=np.int32)
        result = strip_thinking_tokens(tokens)
        assert result is None

    def test_think_at_start(self):
        tokens = np.array([THINK, 99, END_THINK, 5, 6], dtype=np.int32)
        result = strip_thinking_tokens(tokens)
        np.testing.assert_array_equal(result, [5, 6])

    def test_think_at_end(self):
        tokens = np.array([5, 6, THINK, 99, END_THINK], dtype=np.int32)
        result = strip_thinking_tokens(tokens)
        np.testing.assert_array_equal(result, [5, 6])

    def test_custom_ids(self):
        tokens = np.array([1, 50, 99, 51, 2], dtype=np.int32)
        result = strip_thinking_tokens(tokens, think_id=50, end_think_id=51)
        np.testing.assert_array_equal(result, [1, 2])

    def test_orphan_close_mid_sequence(self):
        tokens = np.array([1, END_THINK, 2, END_THINK, 3], dtype=np.int32)
        result = strip_thinking_tokens(tokens)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_preserves_dtype(self):
        tokens = np.array([1, THINK, 99, END_THINK, 2], dtype=np.int32)
        result = strip_thinking_tokens(tokens)
        assert result.dtype == np.int32

    def test_repeated_think_inside_span(self):
        """Repeated <think> inside a span is plain content — not a new span."""
        tokens = np.array(
            [10, THINK, 20, THINK, 30, END_THINK, 40], dtype=np.int32
        )
        result = strip_thinking_tokens(tokens)
        np.testing.assert_array_equal(result, [10, 40])

    def test_repeated_think_then_unmatched(self):
        """Second <think> inside span, only one </think> — exits correctly."""
        tokens = np.array(
            [10, THINK, THINK, THINK, END_THINK, 40, 50], dtype=np.int32
        )
        result = strip_thinking_tokens(tokens)
        np.testing.assert_array_equal(result, [10, 40, 50])

    def test_close_before_open_then_span(self):
        """Orphan close, then a proper span."""
        tokens = np.array(
            [END_THINK, 1, THINK, 99, END_THINK, 2], dtype=np.int32
        )
        result = strip_thinking_tokens(tokens)
        np.testing.assert_array_equal(result, [1, 2])


# ---------------------------------------------------------------------------
# rewrite_dataset — integration tests
# ---------------------------------------------------------------------------

def _build_test_shards(tmp_path, sequences, dtype=np.int32):
    """Build a single rank shard with the given sequences."""
    from vision_tokenization.formats.megatron import IndexedDatasetBuilder

    prefix = str(tmp_path / "rank_0000_chunk_0000")
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=dtype)
    for seq in sequences:
        builder.add_item(np.array(seq, dtype=dtype))
        builder.end_document()
    builder.finalize(prefix + ".idx")
    # Default gating requires the rank completion marker
    (tmp_path / "rank_0000").mkdir(exist_ok=True)
    (tmp_path / "rank_0000" / "_SUCCESS").touch()
    return prefix

def _build_test_shards_multimodal(tmp_path, sequences, modes, dtype=np.int32):
    """Build a single rank shard with sequence modes."""
    from vision_tokenization.formats.megatron import IndexedDatasetBuilder

    prefix = str(tmp_path / "rank_0000_chunk_0000")
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=dtype, multimodal=True)
    for seq, mode in zip(sequences, modes):
        builder.add_item(np.array(seq, dtype=dtype), mode=mode)
        builder.end_document()
    builder.finalize(prefix + ".idx")
    return prefix


def _read_all_sequences(prefix):
    """Read all sequences from a merged dataset."""
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    dataset = IndexedDataset(prefix)
    return [np.array(dataset[i]).copy() for i in range(len(dataset))]


def _read_all_sequences_with_modes(prefix):
    """Read all sequences and modes from a multimodal merged dataset."""
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    dataset = IndexedDataset(prefix, multimodal=True)
    results = []
    for i in range(len(dataset)):
        seq, mode = dataset[i]
        results.append((np.array(seq).copy(), int(mode)))
    return results


class TestRewriteDataset:
    def test_basic_strip(self, tmp_path):
        sequences = [
            [1, THINK, 99, 100, END_THINK, 2],     # → [1, 2]
            [10, 20, 30],                            # no think → unchanged
            [THINK, 50, END_THINK],                  # all think → skipped
            [5, THINK, 60, END_THINK, 6, 7],         # → [5, 6, 7]
        ]
        _build_test_shards(tmp_path, sequences)
        merged = merge_shards(tmp_path, output_name="merged")
        assert merged is not None

        no_cot_prefix = str(tmp_path / "no_cot")
        stats = rewrite_dataset(str(merged), no_cot_prefix, strip_thinking_tokens)

        assert stats.input_count == 4
        assert stats.written_count == 3
        assert stats.skipped_count == 1
        assert stats.output_tokens == 2 + 3 + 3  # [1,2] + [10,20,30] + [5,6,7]

        rewritten = _read_all_sequences(no_cot_prefix)
        assert len(rewritten) == 3
        np.testing.assert_array_equal(rewritten[0], [1, 2])
        np.testing.assert_array_equal(rewritten[1], [10, 20, 30])
        np.testing.assert_array_equal(rewritten[2], [5, 6, 7])

    def test_preserves_sequence_order(self, tmp_path):
        sequences = [
            [100, THINK, 1, END_THINK, 200],
            [300, 400],
            [500, THINK, 2, END_THINK, 600],
        ]
        _build_test_shards(tmp_path, sequences)
        merged = merge_shards(tmp_path, output_name="merged")

        no_cot_prefix = str(tmp_path / "no_cot")
        rewrite_dataset(str(merged), no_cot_prefix, strip_thinking_tokens)

        rewritten = _read_all_sequences(no_cot_prefix)
        np.testing.assert_array_equal(rewritten[0], [100, 200])
        np.testing.assert_array_equal(rewritten[1], [300, 400])
        np.testing.assert_array_equal(rewritten[2], [500, 600])

    def test_preserves_dtype(self, tmp_path):
        sequences = [[1, THINK, 99, END_THINK, 2]]
        _build_test_shards(tmp_path, sequences, dtype=np.int32)
        merged = merge_shards(tmp_path, output_name="merged")

        no_cot_prefix = str(tmp_path / "no_cot")
        rewrite_dataset(str(merged), no_cot_prefix, strip_thinking_tokens)

        from megatron.core.datasets.indexed_dataset import IndexedDataset
        ds = IndexedDataset(no_cot_prefix)
        assert ds.index.dtype == np.int32

    def test_preserves_sequence_modes(self, tmp_path):
        """Multimodal datasets: sequence modes must be preserved through rewrite."""
        sequences = [
            [1, THINK, 99, END_THINK, 2],
            [10, 20, 30],
            [THINK, 50, END_THINK],
        ]
        modes = [0, 1, 0]
        # Build a multimodal dataset directly (merge_shards uses bulk add_index
        # which doesn't handle multimodal, so we create the "merged" file ourselves)
        _build_test_shards_multimodal(tmp_path, sequences, modes)
        merged_prefix = str(tmp_path / "rank_0000_chunk_0000")

        no_cot_prefix = str(tmp_path / "no_cot")
        stats = rewrite_dataset(merged_prefix, no_cot_prefix, strip_thinking_tokens)

        assert stats.written_count == 2
        assert stats.skipped_count == 1

        results = _read_all_sequences_with_modes(no_cot_prefix)
        assert len(results) == 2
        np.testing.assert_array_equal(results[0][0], [1, 2])
        assert results[0][1] == 0
        np.testing.assert_array_equal(results[1][0], [10, 20, 30])
        assert results[1][1] == 1

    def test_error_on_existing_output(self, tmp_path):
        sequences = [[1, 2, 3]]
        _build_test_shards(tmp_path, sequences)
        merged = merge_shards(tmp_path, output_name="merged")

        no_cot_prefix = str(tmp_path / "no_cot")
        rewrite_dataset(str(merged), no_cot_prefix, strip_thinking_tokens)

        with pytest.raises(FileExistsError):
            rewrite_dataset(str(merged), no_cot_prefix, strip_thinking_tokens)

    def test_regression_no_cot_matches_merged_minus_think(self, tmp_path):
        """The no-CoT variant must match merged exactly, minus think spans."""
        sequences = [
            [1, 2, 3],
            [10, THINK, 50, 51, END_THINK, 20],
            [30, 40],
            [THINK, 99, END_THINK],
            [60, THINK, 70, END_THINK, 80, THINK, 90, END_THINK, 100],
        ]
        _build_test_shards(tmp_path, sequences)
        merged = merge_shards(tmp_path, output_name="merged")

        no_cot_prefix = str(tmp_path / "no_cot")
        rewrite_dataset(str(merged), no_cot_prefix, strip_thinking_tokens)

        merged_seqs = _read_all_sequences(str(merged))
        no_cot_seqs = _read_all_sequences(no_cot_prefix)

        expected = []
        for seq in merged_seqs:
            stripped = strip_thinking_tokens(seq)
            if stripped is not None:
                expected.append(stripped)

        assert len(no_cot_seqs) == len(expected)
        for actual, exp in zip(no_cot_seqs, expected):
            np.testing.assert_array_equal(actual, exp)


# ---------------------------------------------------------------------------
# CLI — smoke tests
# ---------------------------------------------------------------------------

class TestCLI:
    def test_strip_thinking_flag_with_defaults(self, tmp_path):
        """--strip-thinking works with default IDs, no tokenizer needed."""
        from vision_tokenization.pipeline.output.merge import main

        sequences = [
            [1, THINK, 99, END_THINK, 2],
            [10, 20],
        ]
        _build_test_shards(tmp_path, sequences)

        ret = main([str(tmp_path), "--strip-thinking"])
        assert ret == 0

        assert (tmp_path / "merged.bin").exists()
        assert (tmp_path / "merged.idx").exists()
        assert (tmp_path / "merged_no_cot.bin").exists()
        assert (tmp_path / "merged_no_cot.idx").exists()

        no_cot_seqs = _read_all_sequences(str(tmp_path / "merged_no_cot"))
        assert len(no_cot_seqs) == 2
        np.testing.assert_array_equal(no_cot_seqs[0], [1, 2])
        np.testing.assert_array_equal(no_cot_seqs[1], [10, 20])

    def test_resolve_thinking_ids(self, tmp_path):
        """--resolve-thinking-ids resolves from the real tokenizer."""
        from vision_tokenization.pipeline.output.merge import main

        tokenizer_path = (
            "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/"
            "apertus_emu3.5_wavtok_instruct"
        )
        if not Path(tokenizer_path).exists():
            pytest.skip("Tokenizer not available")

        sequences = [[1, THINK, 99, END_THINK, 2]]
        _build_test_shards(tmp_path, sequences)

        ret = main([
            str(tmp_path), "--strip-thinking",
            "--resolve-thinking-ids", "--tokenizer-path", tokenizer_path,
        ])
        assert ret == 0

        no_cot_seqs = _read_all_sequences(str(tmp_path / "merged_no_cot"))
        np.testing.assert_array_equal(no_cot_seqs[0], [1, 2])


class TestBandViews:
    """Band .idx views must load through Megatron's NATIVE reader and return
    exactly the sequences whose lengths fall in the band."""

    def test_band_view_readback_native(self, tmp_path):
        try:
            from megatron.core.datasets.indexed_dataset import IndexedDataset
        except ImportError:
            pytest.skip("megatron not available")
        from vision_tokenization.pipeline.output.merge import split_bands

        seqs = [[1] * 5, [2] * 10, [3] * 3, [4] * 20, [5] * 10]
        prefix = _build_test_shards(tmp_path, seqs)
        # bands: <=8 and >8 (toy edges)
        out = split_bands(prefix, edges=[8])
        assert set(out) == {"0k", "gt0k"}

        # split_bands creates each view's .bin alias (hardlink) itself.
        for name, (n, tok, idx_path) in out.items():
            view_prefix = idx_path[:-4]
            assert os.path.samefile(view_prefix + ".bin", prefix + ".bin")
            ds = IndexedDataset(view_prefix)
            assert len(ds) == n
            got = sorted(ds[i].tolist() for i in range(len(ds)))
            want = sorted(s for s in seqs if (len(s) <= 8) == (name == "0k"))
            assert got == want
            assert sum(len(ds[i]) for i in range(len(ds))) == tok


class TestBandSafety:
    """The three band-surface contracts: retro-banding works on an existing
    merge, bad edges are refused loudly, and re-banding is idempotent."""

    def test_bands_apply_to_already_merged_dataset(self, tmp_path):
        from vision_tokenization.pipeline.output.merge import merge_shards

        seqs = [[1] * 5, [2] * 12, [3] * 3]
        _build_test_shards(tmp_path, seqs)
        first = merge_shards(tmp_path, shuffle=False)          # plain merge, no bands
        assert first is not None and not list(tmp_path.glob("merged_*k.idx"))
        again = merge_shards(tmp_path, shuffle=False, bands=[8])  # retro-band
        assert again == first
        assert (tmp_path / "merged_0k.idx").exists()
        assert (tmp_path / "merged_0k.bin").exists()           # alias created too

    def test_unsorted_edges_refused(self):
        from vision_tokenization.pipeline.output.merge import _validate_edges
        with pytest.raises(ValueError, match="ascending"):
            _validate_edges([16384, 8192])
        with pytest.raises(ValueError, match="ascending"):
            _validate_edges([0, 8192])

    def test_name_colliding_edges_refused(self):
        from vision_tokenization.pipeline.output.merge import _validate_edges
        with pytest.raises(ValueError, match="collide"):
            _validate_edges([8192, 8704])  # both floor to "8k"
        assert _validate_edges(["8192", "16384"]) == [8192, 16384]  # CLI strings ok

    def test_rebanding_is_idempotent(self, tmp_path):
        from vision_tokenization.pipeline.output.merge import merge_shards, split_bands

        seqs = [[1] * 5, [2] * 12]
        _build_test_shards(tmp_path, seqs)
        merge_shards(tmp_path, shuffle=False, bands=[8])
        prefix = str(tmp_path / "merged")
        out1 = split_bands(prefix, [8])
        out2 = split_bands(prefix, [8])                        # re-run: alias refreshed, same result
        assert out1 == out2
