"""Run + RunEvent + TailorRequest validation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tailor_core.runs.models import (
    RenderDiff,
    RenderResult,
    RenderStatus,
    Run,
    RunEvent,
    RunStatus,
    TailorRequest,
)
from tests.unit.runs._stub_tailored import StubTailored


def test_tailor_request_accepts_url_only() -> None:
    req = TailorRequest(jd_url="https://example.com/jd")
    assert req.jd_url == "https://example.com/jd"


def test_tailor_request_accepts_text_only() -> None:
    req = TailorRequest(jd_text="Senior Engineer ...")
    assert req.jd_text == "Senior Engineer ..."


def test_tailor_request_rejects_both_set() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        TailorRequest(jd_url="https://x", jd_text="text")


def test_tailor_request_rejects_neither_set() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        TailorRequest()


def test_tailor_request_rejects_whitespace_only_text() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        TailorRequest(jd_text="   \n  ")


def test_tailor_request_verified_context_defaults_none() -> None:
    req = TailorRequest(jd_text="Senior Engineer")
    assert req.verified_context is None


def test_tailor_request_carries_verified_context() -> None:
    req = TailorRequest(jd_text="Senior Engineer", verified_context="jobai: 1,126 tests")
    assert req.verified_context == "jobai: 1,126 tests"


def test_run_status_terminal_set() -> None:
    assert RunStatus.SUCCEEDED.is_terminal
    assert RunStatus.FAILED.is_terminal
    assert not RunStatus.PENDING.is_terminal
    assert not RunStatus.TAILORING.is_terminal


def test_run_round_trips_through_json() -> None:
    now = datetime(2026, 5, 9, tzinfo=UTC)
    run: Run[StubTailored] = Run(
        id="run_x",
        request=TailorRequest(jd_text="text"),
        status=RunStatus.PENDING,
        created_at=now,
        updated_at=now,
    )
    parsed = Run[StubTailored].model_validate_json(run.model_dump_json())
    assert parsed == run


def test_run_round_trips_with_tailored_payload() -> None:
    now = datetime(2026, 5, 9, tzinfo=UTC)
    run: Run[StubTailored] = Run(
        id="run_x",
        request=TailorRequest(jd_text="text"),
        status=RunStatus.SUCCEEDED,
        created_at=now,
        updated_at=now,
        tailored=StubTailored(label="round-trip"),
    )
    parsed = Run[StubTailored].model_validate_json(run.model_dump_json())
    assert parsed.tailored == StubTailored(label="round-trip")


def test_run_event_round_trips() -> None:
    event = RunEvent(
        run_id="run_x",
        status=RunStatus.TAILORING,
        detail="thinking",
        at=datetime(2026, 5, 9, tzinfo=UTC),
    )
    assert RunEvent.model_validate_json(event.model_dump_json()) == event


def test_render_result_round_trips() -> None:
    result = RenderResult(
        doc_id="run_x",
        doc_url="file:///tmp/run_x/resume.pdf",
        pdf_size_bytes=1024,
        diffs=(
            RenderDiff(
                kind="summary",
                heading="Summary",
                status=RenderStatus.REPLACED,
                before_chars=100,
                after_chars=150,
            ),
        ),
    )
    assert RenderResult.model_validate_json(result.model_dump_json()) == result


def test_render_result_rejects_negative_pdf_size() -> None:
    with pytest.raises(ValidationError):
        RenderResult(doc_id="x", doc_url="file:///x.pdf", pdf_size_bytes=-1)


def test_render_result_requires_doc_id_and_url() -> None:
    with pytest.raises(ValidationError):
        RenderResult(doc_id="", doc_url="x")
    with pytest.raises(ValidationError):
        RenderResult(doc_id="x", doc_url="")


def test_render_diff_requires_non_empty_kind() -> None:
    with pytest.raises(ValidationError):
        RenderDiff(kind="", status=RenderStatus.SKIPPED_EMPTY)


def test_render_status_values_are_stable() -> None:
    """Pinning the wire-level enum values; the API will serialise these."""
    assert RenderStatus.REPLACED.value == "replaced"
    assert RenderStatus.SKIPPED_EMPTY.value == "skipped_empty"
    assert RenderStatus.NOT_FOUND.value == "not_found"
