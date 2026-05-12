"""Persistent + hot-reloadable holders for :class:`UserContext`.

Two implementations:

- :class:`FileBackedContextStore` — production. Reads the tree under a root
  directory; on every :meth:`get` it inspects the max ``mtime`` across the
  tree and reloads if anything changed since the last materialisation. No
  background thread, no dependency on ``watchdog``; the API/agent simply
  asks for the context when it needs it and gets the latest.
- :class:`InMemoryContextStore` — used by tests / Phase 4 unit tests.

Both implement the :class:`ContextStore` ``Protocol`` so the agent layer
holds a reference to the protocol, never to the concrete type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from tailor_core.context.loader import load_user_context
from tailor_core.context.models import UserContext

if TYPE_CHECKING:
    from pathlib import Path


class ContextStore(Protocol):
    """The single operation the agent depends on."""

    def get(self) -> UserContext: ...


class FileBackedContextStore:
    """Loads from a directory tree and re-loads on file mutation."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._cached: UserContext | None = None
        self._cached_signature: tuple[int, float] = (0, 0.0)

    def get(self) -> UserContext:
        signature = self._signature()
        if self._cached is None or signature != self._cached_signature:
            self._cached = load_user_context(self._root)
            self._cached_signature = signature
        return self._cached

    def _signature(self) -> tuple[int, float]:
        """Cheap fingerprint of the tree. Changes if any file is added,
        removed, or modified.
        """
        if not self._root.is_dir():
            return (0, 0.0)
        count = 0
        max_mtime = 0.0
        for path in self._root.rglob("*"):
            if path.is_file():
                count += 1
                mtime = path.stat().st_mtime
                max_mtime = max(max_mtime, mtime)
        return (count, max_mtime)


class InMemoryContextStore:
    """Returns the context handed in at construction (or set later)."""

    def __init__(self, context: UserContext | None = None) -> None:
        self._context = context or UserContext()

    def get(self) -> UserContext:
        return self._context

    def set(self, context: UserContext) -> None:
        self._context = context
