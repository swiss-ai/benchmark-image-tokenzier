"""Fast JSON I/O via orjson with file helpers.

Drop-in replacement for common ``json`` patterns::

    from vision_tokenization.utils.json import json_loads, json_dumps, json_load, json_dump

orjson is ~10x faster than stdlib json for both serialization and
deserialization, which matters for JSONL manifests and large metadata files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Union

import orjson


def json_loads(data: Union[str, bytes]) -> Any:
    """Deserialize JSON string/bytes to Python object."""
    return orjson.loads(data)


def json_dumps(obj: Any, *, indent: bool = False, default=None) -> str:
    """Serialize Python object to JSON string.

    Args:
        obj: Object to serialize.
        indent: Pretty-print with 2-space indentation.
        default: Callable for non-serializable types (like ``str``).
    """
    opts = orjson.OPT_NON_STR_KEYS
    if indent:
        opts |= orjson.OPT_INDENT_2
    return orjson.dumps(obj, option=opts, default=default).decode("utf-8")


def json_load(path: Union[str, Path]) -> Any:
    """Read and deserialize a JSON file."""
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def json_dump(obj: Any, path: Union[str, Path], *, indent: bool = True, default=None) -> None:
    """Serialize and write a Python object to a JSON file."""
    opts = orjson.OPT_NON_STR_KEYS
    if indent:
        opts |= orjson.OPT_INDENT_2
    with open(path, "wb") as f:
        f.write(orjson.dumps(obj, option=opts, default=default))


def json_dump_atomic(obj: Any, path: Union[str, Path], *, indent: bool = True, default=None) -> None:
    """Write a JSON file via tmp + fsync + os.replace.

    For commit-record files (e.g. an alignment store's manifest.json, written
    last): a crash mid-write leaves the prior file intact, never a partial one.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(json_dumps(obj, indent=indent, default=default))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
