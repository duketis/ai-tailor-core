"""Tests for the generic verifier scaffold helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tailor_core.llm.client import FakeLLMClient
from tailor_core.verifier.models import (
    IssueSeverity,
    VerificationIssue,
    VerificationResult,
    VerificationStatus,
)
from tailor_core.verifier.scaffold import (
    VerifierError,
    check_pdf_length,
    evaluate_judgement,
    fallback_concerns_result,
    merge_issue,
    parse_verifier_response,
)

_PASSED_PAYLOAD: dict[str, Any] = {
    "status": "passed",
    "summary": "Output cleanly addresses the brief.",
    "issues": [],
    "rationale": "All must-haves present.",
}

_CONCERNS_PAYLOAD: dict[str, Any] = {
    "status": "concerns",
    "summary": "Two warns surfaced.",
    "issues": [
        {
            "severity": "warn",
            "category": "missing_skill",
            "message": "Postgres not in skills list.",
            "suggestion": "Add Postgres explicitly.",
        }
    ],
    "rationale": "Skills section trimmed too aggressively.",
}


# -- parser --------------------------------------------------------------


def test_parser_accepts_passed_payload() -> None:
    result = parse_verifier_response(json.dumps(_PASSED_PAYLOAD))
    assert result.status is VerificationStatus.PASSED
    assert result.issues == ()


def test_parser_accepts_concerns_payload_with_issues() -> None:
    result = parse_verifier_response(json.dumps(_CONCERNS_PAYLOAD))
    assert result.status is VerificationStatus.CONCERNS
    assert len(result.issues) == 1
    assert result.issues[0].severity is IssueSeverity.WARN
    assert result.issues[0].category == "missing_skill"


def test_parser_strips_markdown_fence() -> None:
    fenced = f"```json\n{json.dumps(_PASSED_PAYLOAD)}\n```"
    result = parse_verifier_response(fenced)
    assert result.status is VerificationStatus.PASSED


def test_parser_strips_unlabelled_fence() -> None:
    fenced = f"```\n{json.dumps(_PASSED_PAYLOAD)}\n```"
    assert parse_verifier_response(fenced).status is VerificationStatus.PASSED


def test_parser_rejects_empty_response() -> None:
    with pytest.raises(VerifierError, match="empty"):
        parse_verifier_response("")


def test_parser_rejects_invalid_json() -> None:
    with pytest.raises(VerifierError, match="not valid JSON"):
        parse_verifier_response("{not json")


def test_parser_rejects_non_object() -> None:
    with pytest.raises(VerifierError, match="not a JSON object"):
        parse_verifier_response("[1, 2]")


def test_parser_rejects_schema_violation() -> None:
    bad = dict(_PASSED_PAYLOAD)
    del bad["status"]
    with pytest.raises(VerifierError, match="schema validation"):
        parse_verifier_response(json.dumps(bad))


# -- evaluate_judgement --------------------------------------------------


def test_evaluate_judgement_sends_system_and_user_prompts_to_llm() -> None:
    llm = FakeLLMClient(default_response=json.dumps(_PASSED_PAYLOAD))
    result = evaluate_judgement(
        system_prompt="SYS",
        user_prompt="USER",
        llm=llm,
    )
    assert result.status is VerificationStatus.PASSED
    assert llm.calls == [("SYS", "USER", None)]


def test_evaluate_judgement_forwards_model_override() -> None:
    llm = FakeLLMClient(default_response=json.dumps(_PASSED_PAYLOAD))
    evaluate_judgement(
        system_prompt="SYS",
        user_prompt="USER",
        llm=llm,
        model="claude-sonnet-4-6",
    )
    assert llm.calls[0][2] == "claude-sonnet-4-6"


def test_evaluate_judgement_propagates_parse_errors() -> None:
    llm = FakeLLMClient(default_response="not valid json")
    with pytest.raises(VerifierError):
        evaluate_judgement(system_prompt="SYS", user_prompt="USER", llm=llm)


# -- merge_issue ---------------------------------------------------------


def _passed() -> VerificationResult:
    return VerificationResult(status=VerificationStatus.PASSED, summary="clean")


def _warn(category: str = "x") -> VerificationIssue:
    return VerificationIssue(severity=IssueSeverity.WARN, category=category, message="something")


def _error(category: str = "x") -> VerificationIssue:
    return VerificationIssue(severity=IssueSeverity.ERROR, category=category, message="bad")


def test_merge_warn_into_passed_promotes_to_concerns() -> None:
    result = merge_issue(_passed(), _warn())
    assert result.status is VerificationStatus.CONCERNS
    assert len(result.issues) == 1


def test_merge_error_into_passed_promotes_to_failed() -> None:
    result = merge_issue(_passed(), _error())
    assert result.status is VerificationStatus.FAILED


def test_merge_warn_into_concerns_stays_concerns() -> None:
    base = VerificationResult(status=VerificationStatus.CONCERNS, summary="s")
    result = merge_issue(base, _warn())
    assert result.status is VerificationStatus.CONCERNS


def test_merge_error_into_concerns_promotes_to_failed() -> None:
    base = VerificationResult(status=VerificationStatus.CONCERNS, summary="s")
    result = merge_issue(base, _error())
    assert result.status is VerificationStatus.FAILED


def test_merge_warn_into_failed_stays_failed() -> None:
    base = VerificationResult(status=VerificationStatus.FAILED, summary="s")
    result = merge_issue(base, _warn())
    assert result.status is VerificationStatus.FAILED


def test_merge_info_into_passed_stays_passed() -> None:
    base = _passed()
    info = VerificationIssue(severity=IssueSeverity.INFO, category="x", message="fyi")
    result = merge_issue(base, info)
    assert result.status is VerificationStatus.PASSED
    assert len(result.issues) == 1


# -- check_pdf_length ----------------------------------------------------


def _write_pdf(path: Path, *, pages: int) -> None:
    from pypdf import PdfWriter  # noqa: PLC0415

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=595, height=842)
    with path.open("wb") as fh:
        writer.write(fh)


def test_check_pdf_length_returns_none_under_target(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    _write_pdf(pdf, pages=2)
    assert check_pdf_length(pdf, target_max_pages=3) is None


def test_check_pdf_length_returns_none_at_target(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    _write_pdf(pdf, pages=3)
    assert check_pdf_length(pdf, target_max_pages=3) is None


def test_check_pdf_length_returns_warn_issue_over_target(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    _write_pdf(pdf, pages=4)
    issue = check_pdf_length(pdf, target_max_pages=3)
    assert issue is not None
    assert issue.severity is IssueSeverity.WARN
    assert issue.category == "page_overflow"
    assert "4 pages" in issue.message
    assert "≤3" in issue.message


def test_check_pdf_length_returns_none_on_unreadable_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"definitely not a pdf")
    assert check_pdf_length(pdf, target_max_pages=3) is None


def test_check_pdf_length_uses_custom_message_when_supplied(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    _write_pdf(pdf, pages=2)
    overflow_pdf = tmp_path / "big.pdf"
    _write_pdf(overflow_pdf, pages=4)
    issue = check_pdf_length(
        overflow_pdf,
        target_max_pages=1,
        overflow_message="Cover letter is too long: 4 pages.",
        overflow_suggestion="Trim to a single page.",
    )
    assert issue is not None
    assert issue.message == "Cover letter is too long: 4 pages."
    assert issue.suggestion == "Trim to a single page."


# -- fallback ------------------------------------------------------------


def test_fallback_concerns_result_synthesises_warn_issue() -> None:
    result = fallback_concerns_result("verifier blew up")
    assert result.status is VerificationStatus.CONCERNS
    assert len(result.issues) == 1
    assert result.issues[0].severity is IssueSeverity.WARN
    assert result.issues[0].category == "verifier_failure"
    assert "verifier blew up" in result.issues[0].message


def test_fallback_concerns_result_round_trips() -> None:
    result = fallback_concerns_result("boom")
    assert VerificationResult.model_validate_json(result.model_dump_json()) == result
