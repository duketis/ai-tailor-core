"""SqliteRunsStore + InMemoryRunsStore behavioural tests.

Parameterised over a ``StubTailored`` subclass so the Generic store
machinery is exercised end-to-end.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tailor_core.runs.models import Run, RunStatus, TailorRequest
from tailor_core.runs.store import (
    InMemoryRunsStore,
    RunsStore,
    SqliteRunsStore,
    update_run,
)
from tests.unit.runs._stub_tailored import StubTailored

if TYPE_CHECKING:
    from collections.abc import Iterator


def _make_run(*, run_id: str = "run_a", offset_minutes: int = 0) -> Run[StubTailored]:
    when = datetime(2026, 5, 9, tzinfo=UTC) + timedelta(minutes=offset_minutes)
    return Run(
        id=run_id,
        request=TailorRequest(jd_text="text"),
        status=RunStatus.PENDING,
        created_at=when,
        updated_at=when,
    )


@pytest.fixture(params=["sqlite", "memory"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[RunsStore[StubTailored]]:
    if request.param == "sqlite":
        s: RunsStore[StubTailored] = SqliteRunsStore(
            tailored_cls=StubTailored, db_path=tmp_path / "runs.db"
        )
        try:
            yield s
        finally:
            s.close()  # type: ignore[attr-defined]
    else:
        yield InMemoryRunsStore[StubTailored]()


def test_save_then_get_round_trips(store: RunsStore[StubTailored]) -> None:
    run = _make_run()
    store.save(run)
    assert store.get(run.id) == run


def test_get_returns_none_for_unknown_id(store: RunsStore[StubTailored]) -> None:
    assert store.get("never-saved") is None


def test_save_overwrites_on_second_save(store: RunsStore[StubTailored]) -> None:
    store.save(_make_run())
    store.save(_make_run().model_copy(update={"status": RunStatus.SUCCEEDED}))
    loaded = store.get("run_a")
    assert loaded is not None
    assert loaded.status is RunStatus.SUCCEEDED


def test_list_recent_orders_by_updated_at_descending(
    store: RunsStore[StubTailored],
) -> None:
    older = _make_run(run_id="run_old", offset_minutes=0)
    newer = _make_run(run_id="run_new", offset_minutes=5)
    store.save(older)
    store.save(newer)

    listed = store.list_recent()
    assert [r.id for r in listed] == ["run_new", "run_old"]


def test_list_recent_respects_limit(store: RunsStore[StubTailored]) -> None:
    for i in range(3):
        store.save(_make_run(run_id=f"run_{i}", offset_minutes=i))

    listed = store.list_recent(limit=2)
    assert len(listed) == 2


def test_list_recent_with_zero_limit_returns_empty(
    store: RunsStore[StubTailored],
) -> None:
    store.save(_make_run())
    assert store.list_recent(limit=0) == []


def test_list_recent_with_negative_limit_returns_empty(
    store: RunsStore[StubTailored],
) -> None:
    store.save(_make_run())
    assert store.list_recent(limit=-1) == []


# -- update_run helper ------------------------------------------------------


def test_update_run_replaces_supplied_fields(store: RunsStore[StubTailored]) -> None:
    store.save(_make_run())
    updated = update_run(
        store,
        "run_a",
        status=RunStatus.TAILORING,
        detail="thinking",
    )
    assert updated.status is RunStatus.TAILORING
    assert updated.detail == "thinking"
    assert updated.updated_at > updated.created_at


def test_update_run_records_error(store: RunsStore[StubTailored]) -> None:
    store.save(_make_run())
    updated = update_run(store, "run_a", error="boom")
    assert updated.error == "boom"


def test_update_run_raises_for_unknown_id(store: RunsStore[StubTailored]) -> None:
    with pytest.raises(KeyError):
        update_run(store, "missing", status=RunStatus.SUCCEEDED)


def test_update_run_can_attach_jd_requirements_and_tailored(
    store: RunsStore[StubTailored],
) -> None:
    from tailor_core.jd.models import JobRequirements  # noqa: PLC0415
    from tailor_core.runs.models import RenderResult  # noqa: PLC0415

    store.save(_make_run())
    requirements = JobRequirements(title="Engineer")
    tailored = StubTailored(label="x")
    result = RenderResult(doc_id="d", doc_url="file:///x.pdf")

    updated = update_run(
        store, "run_a", requirements=requirements, tailored=tailored, result=result
    )
    assert updated.requirements == requirements
    assert updated.tailored == tailored
    assert updated.result == result


# -- SQLite-specific persistence -------------------------------------------


def test_sqlite_store_is_usable_from_a_worker_thread(tmp_path: Path) -> None:
    """FastAPI dispatches sync route handlers to a threadpool -- the
    connection must survive cross-thread access (``check_same_thread=False``).
    Regression test for a 500 on /runs and /runs/{id}."""
    import threading  # noqa: PLC0415

    db_path = tmp_path / "runs.db"
    store: SqliteRunsStore[StubTailored] = SqliteRunsStore(
        tailored_cls=StubTailored, db_path=db_path
    )
    try:
        store.save(_make_run())
        result: list[Run[StubTailored] | None] = []

        def fetch_in_thread() -> None:
            result.append(store.get("run_a"))

        thread = threading.Thread(target=fetch_in_thread)
        thread.start()
        thread.join(timeout=2)

        assert result and result[0] is not None
    finally:
        store.close()


def test_sqlite_persists_across_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.db"
    a: SqliteRunsStore[StubTailored] = SqliteRunsStore(tailored_cls=StubTailored, db_path=db_path)
    a.save(_make_run())
    a.close()

    b: SqliteRunsStore[StubTailored] = SqliteRunsStore(tailored_cls=StubTailored, db_path=db_path)
    try:
        loaded = b.get("run_a")
        assert loaded is not None
    finally:
        b.close()


# -- forward-compatibility: tolerate stale rows that no longer validate -----


def _inject_unparseable_row(db_path: Path, row_id: str = "stale_row") -> None:
    """Write a row whose payload contains a value the current schema rejects.

    Mimics pre-pivot data: an old ``RenderStatus`` value (``"failed"``) that
    is no longer part of the enum. The current loader must skip it without
    bringing down the runs page or run-detail endpoint.
    """
    import json as _json  # noqa: PLC0415
    import sqlite3 as _sqlite3  # noqa: PLC0415

    when = "2026-05-09T00:00:00+00:00"
    bad_payload = _json.dumps(
        {
            "id": row_id,
            "request": {"jd_text": "text"},
            "status": "succeeded",
            "created_at": when,
            "updated_at": when,
            "result": {
                "doc_id": "x",
                "doc_url": "file:///x",
                "pdf_size_bytes": 0,
                "diffs": [{"kind": "summary", "status": "failed"}],
            },
        }
    )
    conn = _sqlite3.connect(db_path, isolation_level=None)
    conn.execute(
        "INSERT INTO runs (id, payload, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (row_id, bad_payload, when, when),
    )
    conn.close()


def test_list_recent_skips_unparseable_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.db"
    store: SqliteRunsStore[StubTailored] = SqliteRunsStore(
        tailored_cls=StubTailored, db_path=db_path
    )
    try:
        store.save(_make_run(run_id="good"))
        _inject_unparseable_row(db_path, row_id="stale_row")
        # The bad row is silently skipped; the good row still surfaces.
        ids = [r.id for r in store.list_recent()]
        assert ids == ["good"]
    finally:
        store.close()


def test_clear_removes_every_row_and_returns_count(
    store: RunsStore[StubTailored],
) -> None:
    store.save(_make_run(run_id="a"))
    store.save(_make_run(run_id="b"))
    store.save(_make_run(run_id="c"))
    assert len(store.list_recent()) == 3

    deleted = store.clear()
    assert deleted == 3
    assert store.list_recent() == []
    assert store.get("a") is None


def test_clear_on_empty_store_returns_zero(store: RunsStore[StubTailored]) -> None:
    assert store.clear() == 0


def test_update_run_persists_verification_field(store: RunsStore[StubTailored]) -> None:
    """Covers the ``verification`` branch in ``update_run``."""
    from tailor_core.verifier.models import (  # noqa: PLC0415
        VerificationResult,
        VerificationStatus,
    )

    store.save(_make_run())
    verification = VerificationResult(
        status=VerificationStatus.PASSED,
        summary="ok",
        rationale="all good",
    )
    updated = update_run(store, "run_a", verification=verification)
    assert updated.verification == verification


def test_update_run_persists_vision_verification(store: RunsStore[StubTailored]) -> None:
    """Covers the ``vision_verification`` branch in ``update_run``."""
    from tailor_core.verifier.models import (  # noqa: PLC0415
        VerificationResult,
        VerificationStatus,
    )

    store.save(_make_run())
    vision = VerificationResult(
        status=VerificationStatus.CONCERNS,
        summary="one widow",
        rationale="small layout issue.",
    )
    updated = update_run(store, "run_a", vision_verification=vision)
    assert updated.vision_verification == vision


def test_get_returns_none_for_unparseable_row(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.db"
    store: SqliteRunsStore[StubTailored] = SqliteRunsStore(
        tailored_cls=StubTailored, db_path=db_path
    )
    try:
        _inject_unparseable_row(db_path, row_id="stale_row")
        # ``get`` on a stale row degrades to "not found" rather than 500-ing.
        assert store.get("stale_row") is None
    finally:
        store.close()
