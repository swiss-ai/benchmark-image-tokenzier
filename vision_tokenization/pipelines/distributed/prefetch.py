"""Batch prefetcher with optional multi-worker I/O for the tokenization loop."""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from typing import Optional

logger = logging.getLogger(__name__)

_SENTINEL = None


@dataclass
class PrefetchResult:
    """Result from the prefetch thread for one batch."""

    batch_index: int
    assignment: object
    images: object  # List[Optional[PIL.Image]]
    texts: object  # Optional[List]
    timing: dict  # {"load_s": float, "augment_s": float}
    error: Optional[Exception] = None


class BatchPrefetcher:
    """Prefetches load_batch + augment in background thread(s).

    Yields ``PrefetchResult`` for each batch.  When a per-batch load or
    augment call fails, the result carries ``error`` instead of data so
    the main loop can apply its existing retry/skip logic.

    With ``num_workers > 1`` a :class:`~concurrent.futures.ThreadPoolExecutor`
    parallelises I/O-bound ``load_batch`` calls while a single dispatcher
    thread feeds results into the output queue **in order**.
    """

    def __init__(self, data_loader, augmenter=None, queue_size=2, num_workers=1):
        self._loader = data_loader
        self._augmenter = augmenter
        self._queue: Queue = Queue(maxsize=queue_size)
        self._thread: Optional[threading.Thread] = None
        self._num_workers = max(1, num_workers)

    def _load_one(self, batch_index, ba):
        """Load + augment one batch.  Called from thread pool workers."""
        try:
            t0 = time.perf_counter()
            images, texts = self._loader.load_batch(
                ba.sample_indices, group_slices=ba.group_slices,
            )
            load_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            if self._augmenter is not None:
                images = self._augmenter.augment_batch(images)
            augment_s = time.perf_counter() - t1

            return PrefetchResult(
                batch_index=batch_index,
                assignment=ba,
                images=images,
                texts=texts,
                timing={"load_s": load_s, "augment_s": augment_s},
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
        """Dispatcher thread: load batches via thread pool and enqueue in order."""
        try:
            with ThreadPoolExecutor(max_workers=self._num_workers) as pool:
                for result in pool.map(
                    lambda args: self._load_one(*args),
                    enumerate(batches[start:], start=start),
                ):
                    self._queue.put(result)
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
