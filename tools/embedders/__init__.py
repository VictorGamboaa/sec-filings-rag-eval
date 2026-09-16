"""Embedding backends, selected by config.

The interface and its contract live in :mod:`tools.embedders.base`. Import from
here::

    from tools.embedders import get_embedder
    embedder = get_embedder(config, stage)

CLI::

    python -m tools.embedders --list    # registered providers (no deps needed)
    python -m tools.embedders --smoke   # embed 3 strings, verify the contract
"""

from __future__ import annotations

from tools.embedders.base import (
    Embedder,
    EmbedderError,
    available_embedders,
    get_embedder,
    register_embedder,
)

__all__ = [
    "Embedder",
    "EmbedderError",
    "available_embedders",
    "get_embedder",
    "register_embedder",
]
