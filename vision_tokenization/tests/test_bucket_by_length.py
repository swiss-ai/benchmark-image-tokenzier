"""Tests for bucket_by_length's length filter."""

import numpy as np

from vision_tokenization.pipeline.output.bucket_by_length import make_length_transform


def _seq(n):
    return np.arange(n, dtype=np.int32)


class TestLengthTransform:
    def test_keeps_sequence_inside_the_band(self):
        t = make_length_transform(10, 20)
        for n in (10, 15, 20):
            assert t(_seq(n)) is not None, n

    def test_drops_below_min_and_above_max(self):
        t = make_length_transform(10, 20)
        assert t(_seq(9)) is None
        assert t(_seq(21)) is None

    def test_bounds_are_inclusive(self):
        t = make_length_transform(4, 6)
        assert len(t(_seq(4))) == 4
        assert len(t(_seq(6))) == 6
        assert t(_seq(3)) is None
        assert t(_seq(7)) is None

    def test_returns_the_sequence_unmodified(self):
        """The bucketer filters; it must never rewrite token ids."""
        t = make_length_transform(0, 100)
        seq = np.array([1, 200064, 27, 2], dtype=np.int32)
        out = t(seq)
        assert out is seq
        np.testing.assert_array_equal(out, [1, 200064, 27, 2])

    def test_empty_sequence_dropped_when_min_positive(self):
        assert make_length_transform(1, 10)(_seq(0)) is None

    def test_min_zero_keeps_empty(self):
        assert make_length_transform(0, 10)(_seq(0)) is not None
