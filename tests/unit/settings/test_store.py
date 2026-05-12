"""SqliteSettingsStore + InMemorySettingsStore behavioural tests.

These tests use a small ``_DemoSettings`` subclass of
:class:`BaseRuntimeSettings` so we exercise the Generic store as a real
consumer would: a custom keys (``feature`` here) round-trips through both
backends untouched.
"""

from __future__ import annotations

import stat
from pathlib import Path

from tailor_core.settings.models import BaseRuntimeSettings
from tailor_core.settings.store import (
    InMemorySettingsStore,
    SqliteSettingsStore,
)


class _DemoSettings(BaseRuntimeSettings):
    """Demo subclass with one app-specific key, used purely by tests."""

    feature: str = "default"


def test_sqlite_default_get_returns_default_settings(tmp_path: Path) -> None:
    """First access with no prior write returns a default settings instance."""
    store = SqliteSettingsStore(settings_cls=_DemoSettings, db_path=tmp_path / "rt.db")
    try:
        assert store.get_runtime_settings() == _DemoSettings()
    finally:
        store.close()


def test_sqlite_set_then_get_round_trips(tmp_path: Path) -> None:
    store = SqliteSettingsStore(settings_cls=_DemoSettings, db_path=tmp_path / "rt.db")
    try:
        custom = _DemoSettings(feature="alt", model="claude-sonnet-4-6")
        store.set_runtime_settings(custom)
        assert store.get_runtime_settings() == custom
    finally:
        store.close()


def test_sqlite_set_is_idempotent_via_upsert(tmp_path: Path) -> None:
    """Calling ``set`` twice updates the single row rather than failing."""
    store = SqliteSettingsStore(settings_cls=_DemoSettings, db_path=tmp_path / "rt.db")
    try:
        store.set_runtime_settings(_DemoSettings(feature="a"))
        store.set_runtime_settings(_DemoSettings(feature="b"))
        assert store.get_runtime_settings().feature == "b"
    finally:
        store.close()


def test_sqlite_persists_across_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "rt.db"
    a = SqliteSettingsStore(settings_cls=_DemoSettings, db_path=db_path)
    a.set_runtime_settings(_DemoSettings(model="alt-model"))
    a.close()

    b = SqliteSettingsStore(settings_cls=_DemoSettings, db_path=db_path)
    try:
        assert b.get_runtime_settings().model == "alt-model"
    finally:
        b.close()


def test_sqlite_db_file_is_owner_only_perms(tmp_path: Path) -> None:
    """``0600`` permissions so other users on a shared box can't read settings."""
    db_path = tmp_path / "rt.db"
    store = SqliteSettingsStore(settings_cls=_DemoSettings, db_path=db_path)
    try:
        mode = db_path.stat().st_mode & 0o777
        assert mode == stat.S_IRUSR | stat.S_IWUSR
    finally:
        store.close()


def test_in_memory_starts_with_defaults() -> None:
    store = InMemorySettingsStore(settings_cls=_DemoSettings)
    assert store.get_runtime_settings() == _DemoSettings()


def test_in_memory_set_then_get_round_trips() -> None:
    store = InMemorySettingsStore(settings_cls=_DemoSettings)
    custom = _DemoSettings(feature="x", model="m")
    store.set_runtime_settings(custom)
    assert store.get_runtime_settings() == custom
