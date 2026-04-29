"""Shared in-order parallel execution utility for manifest scanners."""

from __future__ import annotations

import logging
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Sentinel indicating a failed future whose error was handled by error_fn.
_FAILED = object()


def run_ordered_pool(
    n_items: int,
    submit_fn: Callable[[ProcessPoolExecutor, int], Future],
    emit_fn: Callable[[int, Any], None],
    num_workers: int,
    *,
    max_in_flight_factor: int = 2,
    error_fn: Optional[Callable[[int, BaseException], None]] = None,
    progress_fn: Optional[Callable[[int, int], None]] = None,
    progress_interval: int = 100,
) -> None:
    """Execute *n_items* tasks in parallel, emitting results in index order.

    Args:
        n_items: Total number of work items.
        submit_fn: ``submit_fn(pool, idx)`` submits item *idx* and returns
            the :class:`~concurrent.futures.Future`.
        emit_fn: ``emit_fn(idx, result)`` is called exactly once per
            successful item, in strict index order (0, 1, 2, ...).
        num_workers: Number of :class:`ProcessPoolExecutor` workers.
        max_in_flight_factor: Multiplier on *num_workers* for the in-flight
            future cap (default 2).
        error_fn: Optional ``error_fn(idx, exception)`` for non-fatal error
            handling.  When provided, worker exceptions are passed here
            instead of propagating.  When *None*, any worker exception
            terminates the pool.
        progress_fn: Optional ``progress_fn(completed, n_items)`` callback.
        progress_interval: Call *progress_fn* every N completions (and on the
            last item).
    """
    if n_items == 0:
        return

    pool = ProcessPoolExecutor(max_workers=num_workers)
    pool_shutdown = False
    max_in_flight = max(1, num_workers * max_in_flight_factor)
    next_submit_idx = 0
    next_emit_idx = 0
    future_to_idx: dict[Future, int] = {}
    completed_results: dict[int, Any] = {}

    def _shutdown_pool(*, wait_for_workers: bool, cancel_futures: bool) -> None:
        nonlocal pool_shutdown
        if pool_shutdown:
            return
        pool.shutdown(wait=wait_for_workers, cancel_futures=cancel_futures)
        pool_shutdown = True

    try:
        # Prime the pump.
        while next_submit_idx < n_items and len(future_to_idx) < max_in_flight:
            future = submit_fn(pool, next_submit_idx)
            future_to_idx[future] = next_submit_idx
            next_submit_idx += 1

        while next_emit_idx < n_items:
            while next_emit_idx in completed_results:
                result = completed_results.pop(next_emit_idx)
                if result is not _FAILED:
                    emit_fn(next_emit_idx, result)
                next_emit_idx += 1
                if progress_fn and (
                    next_emit_idx % progress_interval == 0
                    or next_emit_idx == n_items
                ):
                    progress_fn(next_emit_idx, n_items)

            if next_emit_idx >= n_items:
                break

            if not future_to_idx:
                raise RuntimeError(
                    f"Ordered pool stalled at item {next_emit_idx}/{n_items} "
                    f"with no futures in flight"
                )

            done, _ = wait(tuple(future_to_idx), return_when=FIRST_COMPLETED)
            for future in done:
                idx = future_to_idx.pop(future)
                try:
                    completed_results[idx] = future.result()
                except Exception as exc:
                    if error_fn is None:
                        raise
                    error_fn(idx, exc)
                    completed_results[idx] = _FAILED

            while next_submit_idx < n_items and len(future_to_idx) < max_in_flight:
                future = submit_fn(pool, next_submit_idx)
                future_to_idx[future] = next_submit_idx
                next_submit_idx += 1
    except BaseException:
        for future in future_to_idx:
            future.cancel()
        _shutdown_pool(wait_for_workers=False, cancel_futures=True)
        raise
    else:
        _shutdown_pool(wait_for_workers=True, cancel_futures=False)
