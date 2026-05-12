"""SQLite + in-memory storage for uploaded context files."""

from __future__ import annotations

import secrets
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from tailor_core.context_files.models import ContextFile, ContextFileKind

DEFAULT_DB_PATH = Path("~/.tailor_core/tailor_core.db").expanduser()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS context_files (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    uploaded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_context_files_uploaded_at
    ON context_files (uploaded_at DESC);
"""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def generate_context_file_id() -> str:
    return f"ctx_{secrets.token_urlsafe(8)}"


class ContextFileStore(Protocol):
    """Storage contract for uploaded files."""

    def add(
        self,
        *,
        name: str,
        kind: ContextFileKind,
        extracted_text: str,
        byte_size: int,
        tags: tuple[str, ...] = (),
        note: str = "",
    ) -> ContextFile: ...
    def get(self, file_id: str) -> ContextFile | None: ...
    def list_all(self) -> list[ContextFile]: ...
    def remove(self, file_id: str) -> bool: ...


class SqliteContextFileStore:
    """SQLite-backed implementation. Shares the Settings/runs DB by default."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, isolation_level=None, check_same_thread=False)
        self._conn.executescript(_SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    def add(
        self,
        *,
        name: str,
        kind: ContextFileKind,
        extracted_text: str,
        byte_size: int,
        tags: tuple[str, ...] = (),
        note: str = "",
    ) -> ContextFile:
        record = ContextFile(
            id=generate_context_file_id(),
            name=name,
            kind=kind,
            extracted_text=extracted_text,
            byte_size=byte_size,
            tags=tags,
            uploaded_at=_utcnow(),
            note=note,
        )
        self._conn.execute(
            """INSERT INTO context_files (id, payload, uploaded_at)
               VALUES (?, ?, ?)""",
            (record.id, record.model_dump_json(), record.uploaded_at.isoformat()),
        )
        return record

    def get(self, file_id: str) -> ContextFile | None:
        row = self._conn.execute(
            "SELECT payload FROM context_files WHERE id = ?", (file_id,)
        ).fetchone()
        if row is None:
            return None
        return ContextFile.model_validate_json(row[0])

    def list_all(self) -> list[ContextFile]:
        rows = self._conn.execute(
            "SELECT payload FROM context_files ORDER BY uploaded_at DESC"
        ).fetchall()
        return [ContextFile.model_validate_json(row[0]) for row in rows]

    def remove(self, file_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM context_files WHERE id = ?", (file_id,))
        return cursor.rowcount > 0


class InMemoryContextFileStore:
    """In-memory store with identical semantics. For tests."""

    def __init__(self) -> None:
        self._files: dict[str, ContextFile] = {}

    def add(
        self,
        *,
        name: str,
        kind: ContextFileKind,
        extracted_text: str,
        byte_size: int,
        tags: tuple[str, ...] = (),
        note: str = "",
    ) -> ContextFile:
        record = ContextFile(
            id=generate_context_file_id(),
            name=name,
            kind=kind,
            extracted_text=extracted_text,
            byte_size=byte_size,
            tags=tags,
            uploaded_at=_utcnow(),
            note=note,
        )
        self._files[record.id] = record
        return record

    def get(self, file_id: str) -> ContextFile | None:
        return self._files.get(file_id)

    def list_all(self) -> list[ContextFile]:
        return sorted(self._files.values(), key=lambda f: f.uploaded_at, reverse=True)

    def remove(self, file_id: str) -> bool:
        return self._files.pop(file_id, None) is not None
