"""Batch prefetcher with optional multi-worker I/O for the tokenization loop."""

import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from queue import Queue
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_SENTINEL = None


@dataclass
class PrefetchResult:
    """Result from the prefetch thread for one batch.

    Without a ``prepare`` hook, ``images``/``texts`` are exactly what the
    loader returned (``List[Optional[PIL.Image]]``, ``Optional[List]``).
    With a hook, they are whatever the hook returned — e.g. the executor's
    hook puts a ``PreparedBatch`` in ``images`` (filtered + CPU-preprocessed,
    ready for one H2D copy) and ``None`` in ``texts``.
    """

    batch_index: int
    assignment: object
    images: object
    texts: object
    timing: dict  # {"load_ms": float}
    error: Optional[Exception] = None


class BatchPrefetcher:
    """Prefetches load_batch in background thread(s).

    Yields ``PrefetchResult`` for each batch.  When a per-batch load or
    augment call fails, the result carries ``error`` instead of data so
    the main loop can apply its existing retry/skip logic.

    Memory is bounded: at most ``queue_size + num_workers`` decoded
    batches are kept in memory at any time.
    """

    def __init__(self, data_loader, prepare=None, queue_size=2, num_workers=1):
        """*prepare* is an optional ``(images, texts, assignment) -> (images, texts)``
        hook run in the worker thread after loading — e.g. CPU-side image
        preprocessing, so it overlaps with GPU work instead of serializing
        on the consumer thread."""
        self._loader = data_loader
        self._prepare = prepare
        self._queue: Queue = Queue(maxsize=queue_size)
        self._thread: Optional[threading.Thread] = None
        self._num_workers = max(1, num_workers)

    def _load_one(self, batch_index, ba):
        """Load one batch. Called from thread pool workers."""
        try:
            t0 = time.perf_counter()
            images, texts = self._loader.load_batch(
                ba.sample_indices, group_slices=ba.group_slices,
            )
            if self._prepare is not None:
                images, texts = self._prepare(images, texts, ba)
            load_ms = (time.perf_counter() - t0) * 1000

            return PrefetchResult(
                batch_index=batch_index,
                assignment=ba,
                images=images,
                texts=texts,
                timing={"load_ms": load_ms},
            )
        except Exception as exc:
            return PrefetchResult(
                batch_index=batch_index,
                assignment=ba,
                images=None,
                texts=None,
                timing={},
                error=exc,
            )

    def _worker(self, batches, start):
        """Dispatcher thread: load batches and enqueue in order.

        Uses a bounded sliding window of futures so that at most
        ``num_workers + queue_size`` decoded batches exist at once,
        preventing unbounded memory growth from eager submission.
        """
        try:
            max_pending = self._num_workers + self._queue.maxsize
            batch_iter = iter(enumerate(batches[start:], start=start))

            if self._num_workers <= 1:
                # Fast path: no thread pool overhead, just sequential I/O.
                for batch_index, ba in batch_iter:
                    result = self._load_one(batch_index, ba)
                    self._queue.put(result)
            else:
                # Bounded parallel: sliding window of futures with ordered drain.
                with ThreadPoolExecutor(max_workers=self._num_workers) as pool:
                    pending: Dict[Future, int] = {}
                    ready: Dict[int, PrefetchResult] = {}
                    next_emit = start

                    def _submit_next() -> bool:
                        try:
                            idx, ba = next(batch_iter)
                        except StopIteration:
                            return False
                        pending[pool.submit(self._load_one, idx, ba)] = idx
                        return True

                    # Seed the initial sliding window.
                    for _ in range(max_pending):
                        if not _submit_next():
                            break

                    while pending:
                        done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                        for future in done:
                            batch_index = pending.pop(future)
                            ready[batch_index] = future.result()

                        while next_emit in ready:
                            self._queue.put(ready.pop(next_emit))
                            next_emit += 1

                            while len(pending) + len(ready) < max_pending:
                                if not _submit_next():
                                    break
        finally:
            self._queue.put(_SENTINEL)

    def iter_batches(self, batches, start=0):
        self._thread = threading.Thread(
            target=self._worker, args=(batches, start), daemon=True,
        )
        self._thread.start()
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            yield item

    def shutdown(self):
        if self._thread is not None:
            self._thread.join(timeout=10)
