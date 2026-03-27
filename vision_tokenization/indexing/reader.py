"""Random-access tar reader with LRU file handle cache.

Thread-safe: each thread gets its own LRU cache of file handles via
``threading.local()``, so multiple prefetch workers can read from the
same ``TarRandomAccessReader`` instance without lock contention.
"""

import logging
import threading
from collections import OrderedDict
from io import BytesIO
from typing import List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


def _is_common_decode_error(err: BaseException) -> bool:
    """Return True for expected image-corruption failures.

    These errors are common in large web-scale datasets and are not actionable
    enough to warrant a full traceback on every occurrence.  We still keep the
    one-line warning so operators can see which raw file/offset is bad.
    """
    if isinstance(err, UnidentifiedImageError):
        return True

    msg = str(err).lower()
    return any(
        marker in msg
        for marker in (
            "image file is truncated",
            "cannot identify image file",
            "broken data stream",
            "truncated file read",
        )
    )


def _log_read_failure(tar_path: str, offset: int, err: BaseException) -> None:
    """Log image-read failures with concise output for expected corruption.

    Corrupted or truncated images are expected occasionally in web-scale
    datasets, so keep those warnings to one line. Unexpected exceptions still
    include the traceback to aid debugging.
    """
    if _is_common_decode_error(err):
        logger.warning(
            "Failed to read image at offset %d in %s: %s",
            offset,
            tar_path,
            err,
        )
        return

    logger.warning(
        "Failed to read image at offset %d in %s",
        offset,
        tar_path,
        exc_info=True,
    )


class TarRandomAccessReader:
    """Read individual images from tar files by byte offset.

    Maintains a **per-thread** LRU cache of open file handles so that
    repeated reads from the same tar file reuse the same ``open()``
    handle, and multiple threads can read concurrently without races.

    Usage::

        with TarRandomAccessReader() as reader:
            img = reader.read_image(tar_path, offset, size)
    """

    def __init__(self, max_open_files: int = 32):
        self.max_open_files = max_open_files
        self._local = threading.local()
        # Track all thread-local handle dicts for cleanup in close()
        self._all_handles_lock = threading.Lock()
        self._all_handles: list = []

    # -- context manager ---------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -- internal ----------------------------------------------------------
    def _get_handles(self) -> OrderedDict:
        """Return the per-thread LRU handle dict, creating it on first access."""
        if not hasattr(self._local, "handles"):
            self._local.handles = OrderedDict()
            with self._all_handles_lock:
                self._all_handles.append(self._local.handles)
        return self._local.handles

    def _get_handle(self, tar_path: str):
        """Return an open file handle for *tar_path*, creating or promoting it in the LRU."""
        handles = self._get_handles()
        if tar_path in handles:
            handles.move_to_end(tar_path)
            return handles[tar_path]

        # Evict oldest if at capacity
        while len(handles) >= self.max_open_files:
            _, old_fh = handles.popitem(last=False)
            old_fh.close()

        fh = open(tar_path, "rb")
        handles[tar_path] = fh
        return fh

    # -- public API --------------------------------------------------------
    def read_bytes(
        self,
        tar_path: str,
        offset_data: int,
        file_size: int,
    ) -> bytes:
        """Read raw bytes from a tar by seeking to its byte offset.

        Args:
            tar_path: Path to the tar file.
            offset_data: Byte offset of the file content (``TarInfo.offset_data``).
            file_size: Size of the file in bytes.

        Returns:
            The raw bytes of the file.
        """
        fh = self._get_handle(tar_path)
        fh.seek(offset_data)
        return fh.read(file_size)

    def read_image(
        self,
        tar_path: str,
        offset_data: int,
        file_size: int,
    ) -> Image.Image:
        """Read a single image from a tar by seeking to its byte offset.

        Args:
            tar_path: Path to the tar file.
            offset_data: Byte offset of the file content (``TarInfo.offset_data``).
            file_size: Size of the file in bytes.

        Returns:
            A PIL Image.
        """
        fh = self._get_handle(tar_path)
        fh.seek(offset_data)
        data = fh.read(file_size)
        img = Image.open(BytesIO(data))
        img.load()  # eager decode — force JPEG/PNG decompress now
        return img

    def read_batch(
        self,
        refs: List[Tuple[str, int, int]],
    ) -> List[Optional[Image.Image]]:
        """Read a batch of images, returning ``None`` for failed reads.

        Args:
            refs: List of ``(tar_path, offset_data, file_size)`` tuples.

        Returns:
            List of PIL Images (or ``None`` on failure), same order as *refs*.
        """
        results: List[Optional[Image.Image]] = []
        for tar_path, offset, size in refs:
            try:
                results.append(self.read_image(tar_path, offset, size))
            except Exception as err:
                _log_read_failure(tar_path, offset, err)
                results.append(None)
        return results

    def close(self):
        """Close all cached file handles across all threads."""
        with self._all_handles_lock:
            for handles in self._all_handles:
                for fh in handles.values():
                    fh.close()
                handles.clear()
            self._all_handles.clear()
