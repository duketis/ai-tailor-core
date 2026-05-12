"""ContextStore implementations — file-backed hot reload + in-memory."""

from __future__ import annotations

import time
from pathlib import Path

from tailor_core.context.models import (
    Contact,
    ResumeBase,
    UserContext,
    WorkHistoryEntry,
)
from tailor_core.context.store import FileBackedContextStore, InMemoryContextStore

# -- InMemoryContextStore ---------------------------------------------------


def test_in_memory_returns_supplied_context() -> None:
    ctx = UserContext(resume=ResumeBase(name="Alex", contact=Contact(email="alex@example.com")))
    store = InMemoryContextStore(ctx)

    assert store.get() is ctx


def test_in_memory_defaults_to_empty_context() -> None:
    store = InMemoryContextStore()
    assert store.get().is_empty()


def test_in_memory_set_replaces_context() -> None:
    store = InMemoryContextStore()
    new = UserContext(work_history=(WorkHistoryEntry(slug="x", title="Eng", company="Acme"),))
    store.set(new)
    assert store.get() is new


# -- FileBackedContextStore -------------------------------------------------


def _seed_basic_context(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "resume.yaml").write_text("name: Alex\ncontact:\n  email: alex@example.com\n")


def test_file_backed_returns_empty_when_root_missing(tmp_path: Path) -> None:
    store = FileBackedContextStore(tmp_path / "missing")
    assert store.get().is_empty()


def test_file_backed_caches_between_calls(tmp_path: Path) -> None:
    _seed_basic_context(tmp_path)
    store = FileBackedContextStore(tmp_path)

    first = store.get()
    second = store.get()

    assert first is second  # cache hit returns the same instance


def test_file_backed_reloads_when_file_modified(tmp_path: Path) -> None:
    _seed_basic_context(tmp_path)
    store = FileBackedContextStore(tmp_path)

    first = store.get()
    assert first.resume is not None
    assert first.resume.name == "Alex"

    # Bump mtime explicitly so the test isn't flaky on filesystems with
    # 1-second mtime resolution.
    (tmp_path / "resume.yaml").write_text(
        "name: Alex Renamed\ncontact:\n  email: alex@example.com\n"
    )
    new_mtime = time.time() + 5
    import os  # noqa: PLC0415

    os.utime(tmp_path / "resume.yaml", (new_mtime, new_mtime))

    second = store.get()
    assert second is not first
    assert second.resume is not None
    assert second.resume.name == "Alex Renamed"


def test_file_backed_reloads_when_new_file_added(tmp_path: Path) -> None:
    _seed_basic_context(tmp_path)
    store = FileBackedContextStore(tmp_path)

    first = store.get()
    assert first.work_history == ()

    wh = tmp_path / "work_history"
    wh.mkdir()
    (wh / "acme.md").write_text("---\ntitle: Eng\ncompany: Acme\n---\n\nbody\n")

    second = store.get()
    assert len(second.work_history) == 1


def test_file_backed_reloads_when_root_appears_after_construction(
    tmp_path: Path,
) -> None:
    """Caller may construct the store before they've made the directory."""
    root = tmp_path / "user_context"
    store = FileBackedContextStore(root)

    first = store.get()
    assert first.is_empty()

    _seed_basic_context(root)
    second = store.get()
    assert not second.is_empty()
