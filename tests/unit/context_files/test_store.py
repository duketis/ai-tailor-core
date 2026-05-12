"""ContextFileStore tests — both backends."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tailor_core.context_files.models import ContextFile, ContextFileKind
from tailor_core.context_files.store import (
    ContextFileStore,
    InMemoryContextFileStore,
    SqliteContextFileStore,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(params=["sqlite", "memory"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[ContextFileStore]:
    if request.param == "sqlite":
        s = SqliteContextFileStore(db_path=tmp_path / "context.db")
        try:
            yield s
        finally:
            s.close()
    else:
        yield InMemoryContextFileStore()


def test_add_returns_record_with_generated_id(store: ContextFileStore) -> None:
    file = store.add(
        name="notes.md",
        kind=ContextFileKind.MARKDOWN,
        extracted_text="hello",
        byte_size=5,
    )
    assert file.id.startswith("ctx_")
    assert file.name == "notes.md"
    assert file.uploaded_at is not None


def test_get_by_id_returns_persisted_record(store: ContextFileStore) -> None:
    file = store.add(
        name="x.txt",
        kind=ContextFileKind.TEXT,
        extracted_text="t",
        byte_size=1,
    )
    assert store.get(file.id) == file


def test_get_returns_none_for_unknown_id(store: ContextFileStore) -> None:
    assert store.get("never-saved") is None


def test_list_all_orders_most_recent_first(store: ContextFileStore) -> None:
    import time  # noqa: PLC0415

    a = store.add(name="a.txt", kind=ContextFileKind.TEXT, extracted_text="a", byte_size=1)
    time.sleep(0.001)
    b = store.add(name="b.txt", kind=ContextFileKind.TEXT, extracted_text="b", byte_size=1)
    time.sleep(0.001)
    c = store.add(name="c.txt", kind=ContextFileKind.TEXT, extracted_text="c", byte_size=1)

    listed = store.list_all()

    assert [f.id for f in listed] == [c.id, b.id, a.id]


def test_list_all_returns_empty_when_no_records(store: ContextFileStore) -> None:
    assert store.list_all() == []


def test_remove_returns_true_when_record_existed(store: ContextFileStore) -> None:
    file = store.add(name="x.txt", kind=ContextFileKind.TEXT, extracted_text="t", byte_size=1)
    assert store.remove(file.id) is True
    assert store.get(file.id) is None


def test_remove_returns_false_for_unknown_id(store: ContextFileStore) -> None:
    assert store.remove("never-saved") is False


def test_add_preserves_tags_and_note(store: ContextFileStore) -> None:
    file = store.add(
        name="x.txt",
        kind=ContextFileKind.TEXT,
        extracted_text="t",
        byte_size=1,
        tags=("project:x", "role:eng"),
        note="why",
    )
    loaded = store.get(file.id)
    assert loaded is not None
    assert loaded.tags == ("project:x", "role:eng")
    assert loaded.note == "why"


# -- SQLite-specific persistence --------------------------------------------


def test_sqlite_persists_across_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "context.db"
    a = SqliteContextFileStore(db_path=db_path)
    file = a.add(name="x.txt", kind=ContextFileKind.TEXT, extracted_text="t", byte_size=1)
    a.close()
    b = SqliteContextFileStore(db_path=db_path)
    try:
        assert b.get(file.id) is not None
    finally:
        b.close()


def test_sqlite_store_is_usable_from_a_worker_thread(tmp_path: Path) -> None:
    """FastAPI dispatches sync routes to a threadpool — connection must
    survive cross-thread access."""
    import threading  # noqa: PLC0415

    db_path = tmp_path / "context.db"
    store = SqliteContextFileStore(db_path=db_path)
    try:
        store.add(name="x.txt", kind=ContextFileKind.TEXT, extracted_text="t", byte_size=1)
        result: list[list[ContextFile]] = []

        def fetch() -> None:
            result.append(store.list_all())

        thread = threading.Thread(target=fetch)
        thread.start()
        thread.join(timeout=2)

        assert result and len(result[0]) == 1
    finally:
        store.close()
