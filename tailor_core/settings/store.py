"""Persistent storage for runtime settings.

Two implementations:

- :class:`SqliteSettingsStore` -- production. SQLite at
  ``~/.tailor_core/tailor_core.db`` by default; consumer apps override with
  their own path. Owner-only file permissions (``0600``).
- :class:`InMemorySettingsStore` -- used by tests. Identical semantics, no I/O.

Both implement the :class:`SettingsStore` ``Protocol`` so callers hold a
reference to the protocol, never to the concrete type.

The whole stack is generic over a :class:`BaseRuntimeSettings` subclass so a
consumer's subclass (e.g. ``resumeai.settings.RuntimeSettings`` adding a
``template_name`` field) round-trips intact through the store.
"""

from __future__ import annotations

import sqlite3
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from tailor_core.settings.models import BaseRuntimeSettings

DEFAULT_DB_PATH = Path("~/.tailor_core/tailor_core.db").expanduser()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runtime_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    settings_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


class SettingsStore[SettingsT: BaseRuntimeSettings](Protocol):
    """The storage contract every Settings backend must satisfy."""

    def get_runtime_settings(self) -> SettingsT: ...
    def set_runtime_settings(self, settings: SettingsT) -> None: ...


class SqliteSettingsStore[SettingsT: BaseRuntimeSettings]:
    """SQLite-backed store. Default DB path: ``~/.tailor_core/tailor_core.db``."""

    def __init__(
        self,
        *,
        settings_cls: type[SettingsT],
        db_path: Path | None = None,
    ) -> None:
        self._settings_cls = settings_cls
        self._db_path = db_path or DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so the connection survives FastAPI's
        # worker-threadpool dispatch.
        self._conn = sqlite3.connect(self._db_path, isolation_level=None, check_same_thread=False)
        self._conn.executescript(_SCHEMA_SQL)
        self._db_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def close(self) -> None:
        self._conn.close()

    def get_runtime_settings(self) -> SettingsT:
        row = self._conn.execute(
            "SELECT settings_json FROM runtime_settings WHERE id = 1"
        ).fetchone()
        if row is None:
            return self._settings_cls()
        return self._settings_cls.model_validate_json(row[0])

    def set_runtime_settings(self, settings: SettingsT) -> None:
        self._conn.execute(
            """INSERT INTO runtime_settings (id, settings_json, updated_at)
               VALUES (1, ?, ?)
               ON CONFLICT(id) DO UPDATE
                 SET settings_json = excluded.settings_json,
                     updated_at = excluded.updated_at""",
            (settings.model_dump_json(), _utcnow_iso()),
        )


class InMemorySettingsStore[SettingsT: BaseRuntimeSettings]:
    """In-memory store with identical semantics. For tests."""

    def __init__(self, *, settings_cls: type[SettingsT]) -> None:
        self._settings_cls = settings_cls
        self._settings: SettingsT = settings_cls()

    def get_runtime_settings(self) -> SettingsT:
        return self._settings

    def set_runtime_settings(self, settings: SettingsT) -> None:
        self._settings = settings
