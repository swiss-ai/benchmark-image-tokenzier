"""Exact-dims batching for alignment media: ``media_id -> block`` must be a pure
function of (bytes, store generation), so NO spillover cluster-mean dims here
(that path is a pretrain throughput optimization; spec invariant 3). Images are
grouped by exact ``(h, w)`` and chunked to ``batch_size``; stragglers form short
batches that keep their exact dims.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ExactDimBatch:
    member_indices: list      # indices into the unique_media list
    resize_height: int
    resize_width: int


def plan_exact_dim_batches(dims, batch_size: int) -> list:
    groups = defaultdict(list)
    for idx, h, w in dims:
        groups[(h, w)].append(idx)
    batches = []
    for (h, w) in sorted(groups):
        members = sorted(groups[(h, w)])
        for s in range(0, len(members), batch_size):
            batches.append(ExactDimBatch(members[s:s + batch_size], h, w))
    return batches
