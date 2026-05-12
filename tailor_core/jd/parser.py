"""Orchestrators: cleaned text → ``JobRequirements`` (and URL convenience)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tailor_core.jd.deterministic import (
    extract_employment_type,
    extract_location,
    extract_remote_type,
    extract_role_type,
    extract_seniority,
)
from tailor_core.jd.fetcher import fetch_jd
from tailor_core.jd.llm_extractor import extract_with_llm
from tailor_core.jd.models import JobRequirements

if TYPE_CHECKING:
    import httpx

    from tailor_core.llm.client import LLMClient


def parse_jd_text(
    cleaned_text: str,
    *,
    llm: LLMClient,
    source_url: str | None = None,
) -> JobRequirements:
    """Combine deterministic + LLM extraction into a :class:`JobRequirements`."""
    extraction = extract_with_llm(cleaned_text, llm)

    return JobRequirements(
        title=extraction.title,
        company=extraction.company,
        location=extract_location(cleaned_text),
        role_type=extract_role_type(cleaned_text),
        seniority=extract_seniority(cleaned_text),
        employment_type=extract_employment_type(cleaned_text),
        remote_type=extract_remote_type(cleaned_text),
        required_skills=tuple(extraction.required_skills),
        nice_to_have_skills=tuple(extraction.nice_to_have_skills),
        must_haves=tuple(extraction.must_haves),
        employer_vocabulary=tuple(extraction.employer_vocabulary),
        raw_text=cleaned_text,
        source_url=source_url,
    )


def parse_jd_url(
    url: str,
    *,
    llm: LLMClient,
    http_client: httpx.Client | None = None,
) -> JobRequirements:
    """Fetch a JD URL and parse it in one go."""
    fetched = fetch_jd(url, http_client=http_client)
    return parse_jd_text(fetched.cleaned_text, llm=llm, source_url=fetched.source_url)
