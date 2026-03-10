"""TokenizationHandler — single tokenizer-agnostic handler.

Calls ``tokenizer.tokenize_batch()`` and writes results via
``MicroShardWriter``.  Works with any tokenizer that implements the
``tokenize_batch(images, resize_size, text=, group_slices=)`` interface
(see ``vokenizers.base.BaseTokenizer``).
"""

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image

from vision_tokenization.pipelines.distributed.checkpoint import WorkerStats
from vision_tokenization.pipelines.distributed.writer import MicroShardWriter

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

    def setup_writer(self, output_dir, rank, chunk_id, tokenizer):
        self.writer.setup_writer(output_dir, rank, chunk_id, tokenizer)

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
    ) -> None:
        """Tokenize a batch and write results to micro-shard.

        Args:
            images: List of PIL images (None entries are skipped).
            resize_size: Batch-wide resize target.
            tokenizer: Any tokenizer implementing ``tokenize_batch()``.
            stats: WorkerStats to update.
            device: CUDA device string (unused here, kept for interface).
            texts: Optional text data (captions, conversations).
            group_slices: Optional ``(num_groups, 2)`` array for multi-image.
        """
        valid_images, valid_texts, valid_slices = self._filter_none(
            images, texts, group_slices, stats,
        )

        if not valid_images:
            return

        # Call tokenizer — tokenizer-agnostic interface
        token_sequences = tokenizer.tokenize_batch(
            valid_images,
            resize_size,
            text=valid_texts if self.needs_text else None,
            group_slices=valid_slices,
        )

        # Write results (skip None entries from multi-image skips)
        for seq in token_sequences:
            if seq is None:
                stats.samples_skipped += 1
                continue
            self.writer.write_sequence(seq.cpu() if seq.is_cuda else seq, stats)

    @staticmethod
    def _filter_none(images, texts, group_slices, stats):
        """Filter out None images (and their paired texts / group entries).

        Returns:
            (valid_images, valid_texts, valid_slices)
        """
        if group_slices is not None:
            # Multi-image: skip entire group if ANY image or text is None
            valid_flat_images = []
            valid_texts = []
            valid_slices = []

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

            valid_slices_arr = np.array(valid_slices, dtype=np.int64) if valid_slices else None
            return (
                valid_flat_images,
                valid_texts if texts is not None else None,
                valid_slices_arr,
            )

        # Single-image path
        if texts is not None:
            valid_images = []
            valid_texts = []
            for i, img in enumerate(images):
                if img is not None and texts[i] is not None:
                    valid_images.append(img)
                    valid_texts.append(texts[i])
                else:
                    stats.samples_skipped += 1
            return valid_images, valid_texts, None

        # Image-only (no text)
        valid_images = [img for img in images if img is not None]
        stats.samples_skipped += len(images) - len(valid_images)
        return valid_images, None, None
