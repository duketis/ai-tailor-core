"""Validation behaviour of the JD models + enums."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tailor_core.jd.models import (
    EmploymentType,
    FetchedJD,
    JobRequirements,
    RemoteType,
    RoleType,
    Seniority,
)


def test_job_requirements_defaults_unknown_enums_and_empty_tuples() -> None:
    req = JobRequirements(title="Senior Software Engineer")

    assert req.role_type is RoleType.UNKNOWN
    assert req.seniority is Seniority.UNKNOWN
    assert req.employment_type is EmploymentType.UNKNOWN
    assert req.remote_type is RemoteType.UNKNOWN
    assert req.required_skills == ()
    assert req.must_haves == ()
    assert req.company is None
    assert req.location is None
    assert req.source_url is None


def test_job_requirements_rejects_empty_title() -> None:
    with pytest.raises(ValidationError):
        JobRequirements(title="")


def test_job_requirements_round_trips_through_json() -> None:
    req = JobRequirements(
        title="Senior Engineer",
        company="Acme",
        location="Melbourne",
        role_type=RoleType.ENGINEERING,
        seniority=Seniority.SENIOR,
        required_skills=("Python", "Postgres"),
        nice_to_have_skills=("Rust",),
        must_haves=("AU citizen",),
        employer_vocabulary=("first principles",),
        raw_text="...",
        source_url="https://example.com/jd",
    )
    parsed = JobRequirements.model_validate_json(req.model_dump_json())
    assert parsed == req


def test_fetched_jd_round_trips() -> None:
    fetched = FetchedJD(
        source_url="https://example.com/jd",
        raw_html="<html></html>",
        cleaned_text="text",
        fetched_at=datetime(2026, 5, 9, tzinfo=UTC),
        ats="greenhouse",
    )
    assert FetchedJD.model_validate_json(fetched.model_dump_json()) == fetched
