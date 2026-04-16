"""Shared layer registry.

Provides a global key-value store for data produced by each pipeline step.
Each step registers its output under a named key so that downstream steps can
retrieve it without the caller having to pass objects through every function.

Mirrors the design of oemer's ``layers.py``.

Example
-------
>>> from orm import layers
>>> layers.register_layer("original_image", img_bgr)
>>> img = layers.get_layer("original_image")
"""
import logging
from typing import Any, List

_logger = logging.getLogger(__name__)

_layers: dict = {}
_access_count: dict = {}


def register_layer(name: str, data: Any) -> None:
    """Register *data* under *name*.

    Logs a warning and returns early if the name is already taken.
    Call :func:`delete_layer` first if you need to overwrite an entry.
    """
    if name in _layers:
        _logger.warning(
            "Layer '%s' is already registered. Delete it first or use a different name.",
            name,
        )
        return
    _layers[name] = data
    _access_count[name] = 0


def get_layer(name: str) -> Any:
    """Return the data registered under *name*.

    Raises :class:`KeyError` when the name has not been registered.
    """
    if name not in _layers:
        raise KeyError(f"Layer '{name}' not registered. Available: {list_layers()}")
    _access_count[name] += 1
    return _layers[name]


def delete_layer(name: str) -> None:
    """Remove a layer from the registry (no-op if not found)."""
    _layers.pop(name, None)
    _access_count.pop(name, None)


def list_layers() -> List[str]:
    """Return the names of all currently registered layers."""
    return list(_layers.keys())


def clear() -> None:
    """Remove *all* registered layers (useful between pipeline runs)."""
    _layers.clear()
    _access_count.clear()


def show_access_count() -> None:
    """Print how many times each layer has been accessed."""
    print(_access_count)
