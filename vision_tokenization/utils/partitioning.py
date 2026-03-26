"""Shared partitioning utilities for splitting work across workers."""

from __future__ import annotations

from typing import List, Sequence, TypeVar

T = TypeVar("T")


def weighted_contiguous_split(
    items: Sequence[T],
    costs: Sequence[float],
    num_workers: int,
) -> List[List[T]]:
    """Split *items* into *num_workers* contiguous chunks balanced by *costs*.

    Contiguous assignment preserves locality (shard/tar ordering), while
    weighting by cost reduces long-tail stragglers on skewed workloads.
    """
    if num_workers <= 0:
        raise ValueError("num_workers must be > 0")
    n = len(items)
    if n == 0:
        return [[] for _ in range(num_workers)]

    splits: List[List[T]] = []
    start = 0

    # Precompute suffix sums for O(1) target_cost lookups
    suffix = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + costs[i]

    while start < n and len(splits) < num_workers - 1:
        remaining_workers = num_workers - len(splits)
        if (n - start) < remaining_workers:
            break

        target_cost = suffix[start] / remaining_workers
        running_cost = 0.0
        end = start

        while end < n:
            next_cost = running_cost + costs[end]
            can_cut = end > start
            must_leave = (n - (end + 1)) < (remaining_workers - 1)

            if (
                can_cut
                and not must_leave
                and abs(target_cost - running_cost) < abs(target_cost - next_cost)
            ):
                break

            running_cost = next_cost
            end += 1

            if (n - end) == (remaining_workers - 1):
                break

        splits.append(list(items[start:end]))
        start = end

    splits.append(list(items[start:]))
    while len(splits) < num_workers:
        splits.append([])
    return splits
