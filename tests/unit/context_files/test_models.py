"""Frozen-model + enum-value sanity for ContextFile."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tailor_core.context_files.models import ContextFile, ContextFileKind


def _file(**overrides: object) -> ContextFile:
    base = {
        "id": "ctx_x",
        "name": "notes.md",
        "kind": ContextFileKind.MARKDOWN,
        "extracted_text": "hello",
        "byte_size": 5,
        "uploaded_at": datetime(2026, 5, 9, tzinfo=UTC),
    }
    base.update(overrides)
    return ContextFile(**base)  # type: ignore[arg-type]


def test_context_file_round_trips() -> None:
    file = _file(tags=("project:x",), note="why")
    parsed = ContextFile.model_validate_json(file.model_dump_json())
    assert parsed == file


def test_context_file_rejects_blank_id() -> None:
    with pytest.raises(ValidationError):
        _file(id="")


def test_context_file_rejects_blank_name() -> None:
    with pytest.raises(ValidationError):
        _file(name="")


def test_context_file_rejects_negative_byte_size() -> None:
    with pytest.raises(ValidationError):
        _file(byte_size=-1)


def test_context_file_kind_values_are_stable() -> None:
    """Pinning the wire-level enum values; routes serialise these."""
    assert ContextFileKind.PDF.value == "pdf"
    assert ContextFileKind.CSV.value == "csv"
    assert ContextFileKind.TEXT.value == "text"
    assert ContextFileKind.MARKDOWN.value == "markdown"
