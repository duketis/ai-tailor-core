"""Rule-based extractors for the fields we don't need an LLM for.

Cheap, deterministic, easy to test. Each returns either a typed enum value
or ``None`` when the signal isn't present.
"""

from __future__ import annotations

import re

from tailor_core.jd.models import EmploymentType, RemoteType, RoleType, Seniority

# -- Location ----------------------------------------------------------------

_AU_CITIES: tuple[str, ...] = (
    "Sydney",
    "Melbourne",
    "Brisbane",
    "Perth",
    "Adelaide",
    "Canberra",
    "Hobart",
    "Darwin",
    "Newcastle",
    "Wollongong",
    "Gold Coast",
    "Geelong",
)
_AU_STATES: tuple[str, ...] = (
    "NSW",
    "VIC",
    "QLD",
    "WA",
    "SA",
    "ACT",
    "TAS",
    "NT",
)


def extract_location(text: str) -> str | None:
    """Pick the first AU city / state mention. Returns canonical capitalisation."""
    for city in _AU_CITIES:
        if re.search(rf"\b{re.escape(city)}\b", text, re.IGNORECASE):
            return city
    # State-only mentions count when no city matched.
    for state in _AU_STATES:
        if re.search(rf"\b{state}\b", text):
            return state
    return None


# -- Seniority ---------------------------------------------------------------

# Order matters — check most-specific first.
_SENIORITY_PATTERNS: tuple[tuple[Seniority, re.Pattern[str]], ...] = (
    (Seniority.PRINCIPAL, re.compile(r"\b(principal|lead engineer|tech lead)\b", re.IGNORECASE)),
    (Seniority.STAFF, re.compile(r"\bstaff\b", re.IGNORECASE)),
    (Seniority.SENIOR, re.compile(r"\b(senior|sr\.?)\b", re.IGNORECASE)),
    (Seniority.JUNIOR, re.compile(r"\b(junior|jr\.?|graduate|grad)\b", re.IGNORECASE)),
    (Seniority.MID, re.compile(r"\b(mid[- ]level|intermediate)\b", re.IGNORECASE)),
)


def extract_seniority(text: str) -> Seniority:
    for level, pattern in _SENIORITY_PATTERNS:
        if pattern.search(text):
            return level
    return Seniority.UNKNOWN


# -- Role type ---------------------------------------------------------------

_ROLE_TYPE_PATTERNS: tuple[tuple[RoleType, re.Pattern[str]], ...] = (
    (
        RoleType.ENGINEERING,
        re.compile(
            r"\b(software engineer|developer|backend|frontend|full[- ]stack|sre|"
            r"site reliability|platform engineer|devops|architect)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RoleType.DATA,
        re.compile(
            r"\b(data engineer|data scientist|data analyst|analytics engineer|"
            r"machine learning|ml engineer|ai engineer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RoleType.DESIGN,
        re.compile(
            r"\b(designer|ux|ui designer|product designer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RoleType.PRODUCT,
        re.compile(
            r"\b(product manager|product owner|pm)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RoleType.SALES,
        re.compile(
            r"\b(sales|account executive|account manager|bdr|sdr|business development)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RoleType.MARKETING,
        re.compile(
            r"\b(marketing|content strategist|seo|growth)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RoleType.OPERATIONS,
        re.compile(
            r"\b(operations|ops manager|coo|chief of staff)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RoleType.SUPPORT,
        re.compile(
            r"\b(customer support|customer success|technical support|help desk)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RoleType.HR,
        re.compile(
            r"\b(recruiter|talent acquisition|people operations|hr partner)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RoleType.FINANCE,
        re.compile(
            r"\b(accountant|finance manager|fp&a|controller|cfo)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RoleType.LEGAL,
        re.compile(
            r"\b(lawyer|solicitor|legal counsel|paralegal)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RoleType.HEALTHCARE,
        re.compile(
            r"\b(nurse|doctor|physician|clinician|paramedic|allied health)\b",
            re.IGNORECASE,
        ),
    ),
)


def extract_role_type(text: str) -> RoleType:
    for role, pattern in _ROLE_TYPE_PATTERNS:
        if pattern.search(text):
            return role
    return RoleType.UNKNOWN


# -- Employment type ---------------------------------------------------------

_EMPLOYMENT_PATTERNS: tuple[tuple[EmploymentType, re.Pattern[str]], ...] = (
    (EmploymentType.INTERN, re.compile(r"\b(intern|internship)\b", re.IGNORECASE)),
    (EmploymentType.CASUAL, re.compile(r"\bcasual\b", re.IGNORECASE)),
    (EmploymentType.PART_TIME, re.compile(r"\bpart[- ]time\b", re.IGNORECASE)),
    (
        EmploymentType.CONTRACT,
        re.compile(
            r"\b(contract|contractor|fixed[- ]term)\b",
            re.IGNORECASE,
        ),
    ),
    (
        EmploymentType.FULL_TIME,
        re.compile(
            r"\b(full[- ]time|permanent)\b",
            re.IGNORECASE,
        ),
    ),
)


def extract_employment_type(text: str) -> EmploymentType:
    for emp, pattern in _EMPLOYMENT_PATTERNS:
        if pattern.search(text):
            return emp
    return EmploymentType.UNKNOWN


# -- Remote type -------------------------------------------------------------

_REMOTE_PATTERNS: tuple[tuple[RemoteType, re.Pattern[str]], ...] = (
    (RemoteType.HYBRID, re.compile(r"\bhybrid\b", re.IGNORECASE)),
    (
        RemoteType.REMOTE,
        re.compile(
            r"\b(fully[- ]remote|remote[- ]first|work[- ]from[- ]home|wfh|remote)\b",
            re.IGNORECASE,
        ),
    ),
    (
        RemoteType.ONSITE,
        re.compile(
            r"\b(on[- ]site|onsite|in[- ]office|in office)\b",
            re.IGNORECASE,
        ),
    ),
)


def extract_remote_type(text: str) -> RemoteType:
    for remote, pattern in _REMOTE_PATTERNS:
        if pattern.search(text):
            return remote
    return RemoteType.UNKNOWN
