"""Persistent + in-memory storage for :class:`Run` records.

The whole stack is generic over the consumer's ``TailoredT`` so a
subclass of ``Run[TailoredResume]`` round-trips through the SQLite +
in-memory backends with full schema validation.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ValidationError

from tailor_core.runs.models import Run, RunStatus

_log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("~/.tailor_core/tailor_core.db").expanduser()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_updated_at ON runs (updated_at DESC);
"""


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RunsStore[TailoredT: BaseModel](Protocol):
    """Storage contract for run records."""

    def get(self, run_id: str) -> Run[TailoredT] | None: ...
    def save(self, run: Run[TailoredT]) -> None: ...
    def list_recent(self, limit: int = 20) -> list[Run[TailoredT]]: ...
    def clear(self) -> int: ...


class SqliteRunsStore[TailoredT: BaseModel]:
    """SQLite-backed implementation. Default db at
    ``~/.tailor_core/tailor_core.db``; consumer apps override with their
    own path so each app's data lands in one file."""

    def __init__(
        self,
        *,
        tailored_cls: type[TailoredT],
        db_path: Path | None = None,
    ) -> None:
        # Parameterise on the concrete Run[TailoredT] so model_validate_json
        # restores ``tailored`` to the right pydantic type. Without this,
        # the bound generic would be erased and validation would fall back
        # to ``Any``.
        self._run_cls: type[Run[TailoredT]] = Run[tailored_cls]  # type: ignore[valid-type]
        self._db_path = db_path or DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so the connection survives FastAPI's
        # worker-threadpool dispatch. SQLite serialises writes at the C
        # level; per-call usage is safe at our concurrency profile.
        self._conn = sqlite3.connect(self._db_path, isolation_level=None, check_same_thread=False)
        self._conn.executescript(_SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    def get(self, run_id: str) -> Run[TailoredT] | None:
        row = self._conn.execute("SELECT payload FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        try:
            return self._run_cls.model_validate_json(row[0])
        except ValidationError as exc:
            _log.warning("skipping unparseable run %r: %s", run_id, exc)
            return None

    def save(self, run: Run[TailoredT]) -> None:
        payload = run.model_dump_json()
        self._conn.execute(
            """INSERT INTO runs (id, payload, created_at, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE
                 SET payload = excluded.payload,
                     updated_at = excluded.updated_at""",
            (
                run.id,
                payload,
                run.created_at.isoformat(),
                run.updated_at.isoformat(),
            ),
        )

    def list_recent(self, limit: int = 20) -> list[Run[TailoredT]]:
        rows = self._conn.execute(
            "SELECT id, payload FROM runs ORDER BY updated_at DESC LIMIT ?",
            (max(0, limit),),
        ).fetchall()
        # Tolerate rows that no longer validate against the current schema
        # (e.g. a renamed enum value). Skipping the row beats 500-ing the
        # whole runs page; the bad record is logged so we can decide
        # whether to wipe it.
        runs: list[Run[TailoredT]] = []
        for row_id, payload in rows:
            try:
                runs.append(self._run_cls.model_validate_json(payload))
            except ValidationError as exc:
                _log.warning("skipping unparseable run row %r: %s", row_id, exc)
        return runs

    def clear(self) -> int:
        """Delete every run record. Returns the count that was removed.

        Per-run output directories on disk (``runs/<run_id>/...``) are
        NOT touched -- the user may want to keep the PDF artefacts even
        after clearing the run history. ``rm -rf runs/`` to wipe those.
        """
        cursor = self._conn.execute("DELETE FROM runs")
        return cursor.rowcount


class InMemoryRunsStore[TailoredT: BaseModel]:
    """In-memory store with identical semantics. For tests."""

    def __init__(self) -> None:
        self._runs: dict[str, Run[TailoredT]] = {}

    def get(self, run_id: str) -> Run[TailoredT] | None:
        return self._runs.get(run_id)

    def save(self, run: Run[TailoredT]) -> None:
        self._runs[run.id] = run

    def list_recent(self, limit: int = 20) -> list[Run[TailoredT]]:
        ordered = sorted(self._runs.values(), key=lambda r: r.updated_at, reverse=True)
        return ordered[: max(0, limit)]

    def clear(self) -> int:
        count = len(self._runs)
        self._runs.clear()
        return count


def update_run[TailoredT: BaseModel](
    store: RunsStore[TailoredT],
    run_id: str,
    *,
    status: RunStatus | None = None,
    detail: str | None = None,
    error: str | None = None,
    requirements: object | None = None,
    tailored: object | None = None,
    result: object | None = None,
    verification: object | None = None,
    vision_verification: object | None = None,
) -> Run[TailoredT]:
    """Mutate a stored run by replacing fields the caller supplied.

    Returns the updated, persisted run. Raises ``KeyError`` if the run is
    missing -- callers should always have created the run via ``save`` first.
    """
    current = store.get(run_id)
    if current is None:
        raise KeyError(f"unknown run {run_id!r}")

    updates: dict[str, object] = {"updated_at": _utcnow()}
    if status is not None:
        updates["status"] = status
    if detail is not None:
        updates["detail"] = detail
    if error is not None:
        updates["error"] = error
    if requirements is not None:
        updates["requirements"] = requirements
    if tailored is not None:
        updates["tailored"] = tailored
    if result is not None:
        updates["result"] = result
    if verification is not None:
        updates["verification"] = verification
    if vision_verification is not None:
        updates["vision_verification"] = vision_verification

    new_run = current.model_copy(update=updates)
    store.save(new_run)
    return new_run
