"""Validation behaviour of the context models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tailor_core.context.models import (
    Contact,
    CoverLetterEntry,
    Education,
    GitAuditEntry,
    ProjectEntry,
    ResumeBase,
    UserContext,
    WorkHistoryEntry,
)


def test_resume_base_requires_non_empty_name() -> None:
    with pytest.raises(ValidationError):
        ResumeBase(name="", contact=Contact(email="x@example.com"))


def test_resume_base_round_trips() -> None:
    base = ResumeBase(
        name="Alex",
        headline="Engineer",
        contact=Contact(email="alex@example.com", location="Melbourne"),
        skills=("Python", "Postgres"),
        education=(
            Education(
                institution="Sample U",
                degree="BEng",
                field="SE",
                year_start=2014,
                year_end=2017,
            ),
        ),
        certifications=("AWS SAA",),
    )
    parsed = ResumeBase.model_validate_json(base.model_dump_json())
    assert parsed == base


def test_work_history_entry_requires_title_and_company() -> None:
    with pytest.raises(ValidationError):
        WorkHistoryEntry(slug="x", title="", company="Acme")
    with pytest.raises(ValidationError):
        WorkHistoryEntry(slug="x", title="Eng", company="")


def test_user_context_is_empty_by_default() -> None:
    ctx = UserContext()
    assert ctx.is_empty()


def test_user_context_not_empty_when_resume_present() -> None:
    ctx = UserContext(resume=ResumeBase(name="Alex", contact=Contact(email="a@example.com")))
    assert not ctx.is_empty()


def test_user_context_not_empty_when_only_work_history() -> None:
    ctx = UserContext(work_history=(WorkHistoryEntry(slug="a", title="Eng", company="Acme"),))
    assert not ctx.is_empty()


def test_user_context_not_empty_when_only_git_audit() -> None:
    ctx = UserContext(git_audit=(GitAuditEntry(slug="a", repo="acme/x"),))
    assert not ctx.is_empty()


def test_user_context_not_empty_when_only_cover_letters() -> None:
    ctx = UserContext(cover_letters=(CoverLetterEntry(slug="a"),))
    assert not ctx.is_empty()


def test_project_entry_requires_non_empty_name() -> None:
    with pytest.raises(ValidationError):
        ProjectEntry(slug="x", name="")


def test_user_context_not_empty_when_only_projects() -> None:
    ctx = UserContext(projects=(ProjectEntry(slug="a", name="a"),))
    assert not ctx.is_empty()
