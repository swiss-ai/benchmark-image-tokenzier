"""TokenizationHandler — single tokenizer-agnostic handler.

Calls ``tokenizer.tokenize_batch()`` and writes results via
``MicroShardWriter``.  Works with any tokenizer that implements the
``tokenize_batch(images, resize_size, text=, group_slices=)`` interface.
"""

import logging
import time
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch

from ...runtime.checkpoint import WorkerStats
from .writer import MicroShardWriter

logger = logging.getLogger(__name__)


class TokenizationHandler:
    """Tokenizer-agnostic handler for the distributed tokenization pipeline.

    Owns: call ``tokenizer.tokenize_batch()`` on pre-filtered input,
    write sequences, track stats.  Does NOT know about EMU tokens,
    encapsulation, or conversation policy.
    """

    def __init__(self, writer: MicroShardWriter, needs_text: bool):
        self.writer = writer
        self.needs_text = needs_text

    # -- Delegate writer lifecycle to the writer object --

    @property
    def chunk_samples(self):
        return self.writer.chunk_samples

    def setup_writer(self, *args, **kwargs):
        self.writer.setup_writer(*args, **kwargs)

    def checkpoint_writer(self):
        return self.writer.checkpoint_writer()

    def finalize_writer(self):
        self.writer.finalize_writer()

    # -- Core processing --

    def process_batch(
        self,
        images: List[Optional[Image.Image]],
        resize_size: Tuple[int, int],
        tokenizer,
        stats: WorkerStats,
        device: str,
        texts: Optional[List[Any]] = None,
        group_slices: Optional[np.ndarray] = None,
        timing_enabled: bool = False,
    ) -> dict:
        """Tokenize a batch and write results to micro-shard.

        Args:
            images: List of PIL images (None entries are skipped).
            resize_size: Batch-wide resize target.
            tokenizer: Any tokenizer implementing ``tokenize_batch()``.
            stats: WorkerStats to update.
            device: CUDA device string (unused here, kept for interface).
            texts: Optional text data (captions, conversations).
            group_slices: Optional ``(num_groups, 2)`` array for multi-image.
            timing_enabled: When true, populate wall-clock timings for the
                tokenization and write stages. GPU tokenization time is
                measured with CUDA events.

        Returns:
            Timing dict with keys ``tokenize_wall_ms``, ``tokenize_gpu_ms``,
            ``write_ms`` (zeros when *timing_enabled* is false).
        """
        timings = {
            "tokenize_wall_ms": 0.0,
            "tokenize_gpu_ms": 0.0,
            "write_ms": 0.0,
        }

        # Input is already filtered by the executor's prefetch prepare hook —
        # None entries never reach this layer.
        if len(images) == 0:
            return timings

        if timing_enabled:
            tokenize_start = time.perf_counter()
            # CUDA events conflict with torch.compile reduce-overhead
            # (CUDA graph capture). Use sync + wall-clock instead.
            use_cuda_events = (
                torch.cuda.is_available()
                and str(device).startswith("cuda")
                and not getattr(tokenizer, "torch_compile", False)
            )
            cuda_device = torch.device(device) if use_cuda_events else None
            start_event = None
            end_event = None
            if use_cuda_events:
                torch.cuda.synchronize(cuda_device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                with torch.cuda.device(cuda_device):
                    start_event.record()

        try:
            # Call tokenizer — tokenizer-agnostic interface
            token_sequences = tokenizer.tokenize_batch(
                images,
                resize_size,
                text=texts if self.needs_text else None,
                group_slices=group_slices,
            )
        finally:
            if timing_enabled:
                if start_event is not None and end_event is not None:
                    with torch.cuda.device(cuda_device):
                        end_event.record()
                    torch.cuda.synchronize(cuda_device)
                    timings["tokenize_gpu_ms"] = start_event.elapsed_time(end_event)
                else:
                    # Sync + wall-clock as GPU time proxy when CUDA events disabled
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    timings["tokenize_gpu_ms"] = (time.perf_counter() - tokenize_start) * 1000
                timings["tokenize_wall_ms"] = (time.perf_counter() - tokenize_start) * 1000

        # Write results (skip None entries from multi-image skips)
        if timing_enabled:
            write_start = time.perf_counter()
        for seq in token_sequences:
            if seq is None:
                stats.samples_skipped += 1
                continue
            self.writer.write_sequence(seq.cpu() if seq.is_cuda else seq, stats)
        if timing_enabled:
            timings["write_ms"] = (time.perf_counter() - write_start) * 1000
        return timings

