"""Tests for the rule-based JD extractors."""

from __future__ import annotations

import pytest

from tailor_core.jd.deterministic import (
    extract_employment_type,
    extract_location,
    extract_remote_type,
    extract_role_type,
    extract_seniority,
)
from tailor_core.jd.models import EmploymentType, RemoteType, RoleType, Seniority

# -- location ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Based in Melbourne", "Melbourne"),
        ("Sydney CBD office", "Sydney"),
        ("Brisbane or remote", "Brisbane"),
        ("Located in NSW", "NSW"),
        ("Hybrid VIC role", "VIC"),
    ],
)
def test_extract_location_recognises_au_cities_and_states(text: str, expected: str) -> None:
    assert extract_location(text) == expected


def test_extract_location_returns_none_when_no_match() -> None:
    assert extract_location("Fully remote, North America preferred") is None


def test_extract_location_prefers_city_over_state() -> None:
    assert extract_location("Sydney NSW") == "Sydney"


def test_extract_location_is_case_insensitive() -> None:
    assert extract_location("based in melbourne") == "Melbourne"


# -- seniority ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Senior Software Engineer", Seniority.SENIOR),
        ("Sr. Backend Developer", Seniority.SENIOR),
        ("Staff Engineer", Seniority.STAFF),
        ("Principal Engineer, Platform", Seniority.PRINCIPAL),
        ("Tech Lead, Payments", Seniority.PRINCIPAL),
        ("Junior Frontend Developer", Seniority.JUNIOR),
        ("Graduate Engineer Programme", Seniority.JUNIOR),
        ("Intermediate / Mid-level Developer", Seniority.MID),
    ],
)
def test_extract_seniority(text: str, expected: Seniority) -> None:
    assert extract_seniority(text) == expected


def test_extract_seniority_returns_unknown_when_no_keyword() -> None:
    assert extract_seniority("Software Engineer") is Seniority.UNKNOWN


# -- role type ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Senior Software Engineer", RoleType.ENGINEERING),
        ("Backend Developer", RoleType.ENGINEERING),
        ("Site Reliability Engineer", RoleType.ENGINEERING),
        ("Data Scientist (NLP)", RoleType.DATA),
        ("ML Engineer", RoleType.DATA),
        ("Senior Product Designer", RoleType.DESIGN),
        ("Product Manager, Growth", RoleType.PRODUCT),
        ("Account Executive — Mid Market", RoleType.SALES),
        ("Growth Marketing Lead", RoleType.MARKETING),
        ("Operations Manager", RoleType.OPERATIONS),
        ("Customer Support Specialist", RoleType.SUPPORT),
        ("Recruiter, Engineering", RoleType.HR),
        ("Finance Manager", RoleType.FINANCE),
        ("Legal Counsel", RoleType.LEGAL),
        ("Registered Nurse", RoleType.HEALTHCARE),
    ],
)
def test_extract_role_type(text: str, expected: RoleType) -> None:
    assert extract_role_type(text) == expected


def test_extract_role_type_unknown_when_no_signal() -> None:
    assert extract_role_type("Cosmic Wrangler — galactic standards") is RoleType.UNKNOWN


# -- employment type ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Full-time, permanent role", EmploymentType.FULL_TIME),
        ("Permanent position", EmploymentType.FULL_TIME),
        ("12 month fixed-term contract", EmploymentType.CONTRACT),
        ("Contractor opportunity", EmploymentType.CONTRACT),
        ("Part-time (3 days/week)", EmploymentType.PART_TIME),
        ("Casual position", EmploymentType.CASUAL),
        ("Summer Internship 2026", EmploymentType.INTERN),
    ],
)
def test_extract_employment_type(text: str, expected: EmploymentType) -> None:
    assert extract_employment_type(text) == expected


def test_extract_employment_type_unknown() -> None:
    assert extract_employment_type("Software Engineer") is EmploymentType.UNKNOWN


# -- remote type -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Fully-remote role", RemoteType.REMOTE),
        ("Remote-first culture", RemoteType.REMOTE),
        ("WFH supported", RemoteType.REMOTE),
        ("Hybrid (3 days in office)", RemoteType.HYBRID),
        ("On-site, Sydney CBD", RemoteType.ONSITE),
        ("In office Tuesdays + Thursdays", RemoteType.ONSITE),
    ],
)
def test_extract_remote_type(text: str, expected: RemoteType) -> None:
    assert extract_remote_type(text) == expected


def test_extract_remote_type_unknown() -> None:
    assert extract_remote_type("Software Engineer at Acme") is RemoteType.UNKNOWN


def test_remote_type_hybrid_takes_precedence_over_remote() -> None:
    """A 'hybrid' role often mentions some remote days; we must not mis-label."""
    assert extract_remote_type("Hybrid role with remote Fridays") is RemoteType.HYBRID
