"""TokenizationHandler — single tokenizer-agnostic handler.

Calls ``tokenizer.tokenize_batch()`` and writes results via
``MicroShardWriter``.  Works with any tokenizer that implements the
``tokenize_batch(images, resize_size, text=, group_slices=)`` interface
(see ``vokenizers.base.BaseTokenizer``).
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

    Owns: filter None images, call ``tokenizer.tokenize_batch()``,
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
        source_ids: Optional[np.ndarray] = None,
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
            source_ids: Optional per-image source manifest rows (flat, parallel
                to *images*).  When provided, filtered in lockstep and recorded
                per written sequence for provenance.

        Returns:
            Timing dict with keys ``tokenize_wall_ms``, ``tokenize_gpu_ms``,
            ``write_ms`` (zeros when *timing_enabled* is false).
        """
        timings = {
            "tokenize_wall_ms": 0.0,
            "tokenize_gpu_ms": 0.0,
            "write_ms": 0.0,
        }

        valid_images, valid_texts, valid_slices, valid_source_ids = self._filter_none(
            images, texts, group_slices, stats, source_ids,
        )

        if not valid_images:
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
                valid_images,
                resize_size,
                text=valid_texts if self.needs_text else None,
                group_slices=valid_slices,
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

        # Write results (skip None entries from multi-image skips). The
        # provenance-off path is kept identical to the original loop (no extra
        # allocation); when on, valid_source_ids is 1:1 with token_sequences so a
        # None sequence simply drops its source id alongside it.
        if timing_enabled:
            write_start = time.perf_counter()
        if valid_source_ids is None:
            for seq in token_sequences:
                if seq is None:
                    stats.samples_skipped += 1
                    continue
                self.writer.write_sequence(seq.cpu() if seq.is_cuda else seq, stats)
        else:
            for seq, sid in zip(token_sequences, valid_source_ids):
                if seq is None:
                    stats.samples_skipped += 1
                    continue
                self.writer.write_sequence(seq.cpu() if seq.is_cuda else seq, stats, source_id=sid)
        if timing_enabled:
            timings["write_ms"] = (time.perf_counter() - write_start) * 1000
        return timings

    @staticmethod
    def _filter_none(images, texts, group_slices, stats, source_ids=None):
        """Filter out None images (and their paired texts / group entries).

        *source_ids* (when provided) is a flat array parallel to *images*; it is
        filtered in lockstep and reduced to one id per surviving group (the
        group's first image — its lowest-component-index member).

        Returns:
            (valid_images, valid_texts, valid_slices, valid_source_ids)
            where ``valid_source_ids`` is ``None`` when *source_ids* is None.
        """
        if group_slices is not None:
            # Multi-image: skip entire group if ANY image or text is None
            valid_flat_images = []
            valid_texts = []
            valid_slices = []
            valid_source_ids = [] if source_ids is not None else None

            for g_idx, (start, end) in enumerate(group_slices):
                start, end = int(start), int(end)
                group_images = images[start:end]
                text = texts[g_idx] if texts is not None else None

                if (texts is not None and text is None) or any(
                    img is None for img in group_images
                ):
                    stats.samples_skipped += 1
                    continue

                new_start = len(valid_flat_images)
                valid_flat_images.extend(group_images)
                valid_slices.append((new_start, len(valid_flat_images)))
                if texts is not None:
                    valid_texts.append(text)
                if valid_source_ids is not None:
                    valid_source_ids.append(int(source_ids[start]))

            valid_slices_arr = np.array(valid_slices, dtype=np.int64) if valid_slices else None
            return (
                valid_flat_images,
                valid_texts if texts is not None else None,
                valid_slices_arr,
                valid_source_ids,
            )

        # Single-image path
        if texts is not None:
            valid_images = []
            valid_texts = []
            valid_source_ids = [] if source_ids is not None else None
            for i, img in enumerate(images):
                if img is not None and texts[i] is not None:
                    valid_images.append(img)
                    valid_texts.append(texts[i])
                    if valid_source_ids is not None:
                        valid_source_ids.append(int(source_ids[i]))
                else:
                    stats.samples_skipped += 1
            return valid_images, valid_texts, None, valid_source_ids

        # Image-only (no text). Keep the original comprehension on the
        # provenance-off path; only build the parallel id list when needed.
        if source_ids is None:
            valid_images = [img for img in images if img is not None]
            stats.samples_skipped += len(images) - len(valid_images)
            return valid_images, None, None, None
        valid_images = []
        valid_source_ids = []
        for i, img in enumerate(images):
            if img is not None:
                valid_images.append(img)
                valid_source_ids.append(int(source_ids[i]))
        stats.samples_skipped += len(images) - len(valid_images)
        return valid_images, None, None, valid_source_ids
