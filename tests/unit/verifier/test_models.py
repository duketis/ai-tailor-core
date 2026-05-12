"""Frozen-model + enum-value sanity for the verifier output shapes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tailor_core.verifier.models import (
    IssueSeverity,
    VerificationIssue,
    VerificationResult,
    VerificationStatus,
)


def test_issue_round_trips() -> None:
    issue = VerificationIssue(
        severity=IssueSeverity.WARN,
        category="missing_must_have",
        message="JD requires AU citizenship; tailored output doesn't mention it.",
        suggestion="Add a 'AU citizen' line to must-haves.",
    )
    parsed = VerificationIssue.model_validate_json(issue.model_dump_json())
    assert parsed == issue


def test_issue_rejects_blank_category() -> None:
    with pytest.raises(ValidationError):
        VerificationIssue(severity=IssueSeverity.INFO, category="", message="msg")


def test_issue_rejects_blank_message() -> None:
    with pytest.raises(ValidationError):
        VerificationIssue(severity=IssueSeverity.INFO, category="x", message="")


def test_result_round_trips() -> None:
    result = VerificationResult(
        status=VerificationStatus.CONCERNS,
        summary="Two warns surfaced.",
        issues=(
            VerificationIssue(
                severity=IssueSeverity.WARN,
                category="missing_skill",
                message="Postgres not in skills list.",
            ),
        ),
        rationale="Skills section was tightened too aggressively.",
    )
    assert VerificationResult.model_validate_json(result.model_dump_json()) == result


def test_status_values_are_stable() -> None:
    """Pinning the wire-level enum values; the API/UI surfaces these."""
    assert VerificationStatus.PASSED.value == "passed"
    assert VerificationStatus.CONCERNS.value == "concerns"
    assert VerificationStatus.FAILED.value == "failed"


def test_severity_values_are_stable() -> None:
    assert IssueSeverity.INFO.value == "info"
    assert IssueSeverity.WARN.value == "warn"
    assert IssueSeverity.ERROR.value == "error"
