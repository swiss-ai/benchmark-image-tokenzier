"""Random-access tar reader with LRU file handle cache."""

import logging
from collections import OrderedDict
from io import BytesIO
from typing import List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


class TarRandomAccessReader:
    """Read individual images from tar files by byte offset.

    Maintains an LRU cache of open file handles so that repeated reads
    from the same tar file reuse the same ``open()`` handle.

    Usage::

        with TarRandomAccessReader() as reader:
            img = reader.read_image(tar_path, offset, size)
    """

    def __init__(self, max_open_files: int = 32):
        self.max_open_files = max_open_files
        # OrderedDict used as LRU: most-recently-used at the end
        self._handles: OrderedDict[str, object] = OrderedDict()

    # -- context manager ---------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -- internal ----------------------------------------------------------
    def _get_handle(self, tar_path: str):
        """Return an open file handle for *tar_path*, creating or promoting it in the LRU."""
        if tar_path in self._handles:
            self._handles.move_to_end(tar_path)
            return self._handles[tar_path]

        # Evict oldest if at capacity
        while len(self._handles) >= self.max_open_files:
            _, old_fh = self._handles.popitem(last=False)
            old_fh.close()

        fh = open(tar_path, "rb")
        self._handles[tar_path] = fh
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
        return Image.open(BytesIO(data))

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
            except Exception:
                logger.warning(f"Failed to read image at offset {offset} in {tar_path}", exc_info=True)
                results.append(None)
        return results

    def close(self):
        """Close all cached file handles."""
        for fh in self._handles.values():
            fh.close()
        self._handles.clear()
