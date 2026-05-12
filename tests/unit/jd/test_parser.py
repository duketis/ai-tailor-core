"""End-to-end JD parser tests (deterministic + LLM merged)."""

from __future__ import annotations

import json

import httpx
import respx

from tailor_core.jd.models import EmploymentType, RemoteType, RoleType, Seniority
from tailor_core.jd.parser import parse_jd_text, parse_jd_url
from tailor_core.llm.client import FakeLLMClient

_LLM_RESPONSE = {
    "title": "Senior Software Engineer",
    "company": "Acme Corp",
    "required_skills": ["Python", "Postgres"],
    "nice_to_have_skills": ["Rust"],
    "must_haves": ["AU citizen"],
    "employer_vocabulary": ["first principles"],
}


_JD_TEXT = (
    "Senior Software Engineer at Acme Corp\n"
    "\n"
    "Hybrid role based in Melbourne. Full-time, permanent.\n"
    "We're looking for a senior backend engineer with strong Python and "
    "Postgres experience. You'll work in a high-trust environment using "
    "first principles thinking.\n"
)


def test_parse_jd_text_combines_deterministic_and_llm_fields() -> None:
    llm = FakeLLMClient(default_response=json.dumps(_LLM_RESPONSE))

    req = parse_jd_text(_JD_TEXT, llm=llm, source_url="https://example.com/jd")

    # LLM-extracted fields:
    assert req.title == "Senior Software Engineer"
    assert req.company == "Acme Corp"
    assert req.required_skills == ("Python", "Postgres")
    assert req.nice_to_have_skills == ("Rust",)
    assert req.must_haves == ("AU citizen",)
    assert req.employer_vocabulary == ("first principles",)

    # Deterministic fields:
    assert req.location == "Melbourne"
    assert req.role_type is RoleType.ENGINEERING
    assert req.seniority is Seniority.SENIOR
    assert req.employment_type is EmploymentType.FULL_TIME
    assert req.remote_type is RemoteType.HYBRID

    # Provenance:
    assert req.raw_text == _JD_TEXT
    assert req.source_url == "https://example.com/jd"


def test_parse_jd_text_without_source_url_leaves_it_none() -> None:
    llm = FakeLLMClient(default_response=json.dumps(_LLM_RESPONSE))

    req = parse_jd_text("Plain text", llm=llm)

    assert req.source_url is None


@respx.mock
def test_parse_jd_url_fetches_then_parses() -> None:
    url = "https://boards.greenhouse.io/acme/jobs/123"
    html = (
        "<html><body><main><h1>Senior Software Engineer</h1>"
        "<p>Hybrid Melbourne role. Python and Postgres required.</p></main></body></html>"
    )
    respx.get(url).mock(return_value=httpx.Response(200, text=html))
    llm = FakeLLMClient(default_response=json.dumps(_LLM_RESPONSE))

    req = parse_jd_url(url, llm=llm)

    assert req.title == "Senior Software Engineer"
    assert req.location == "Melbourne"
    assert req.source_url == url
    assert req.role_type is RoleType.ENGINEERING


@respx.mock
def test_parse_jd_url_uses_supplied_http_client() -> None:
    url = "https://example.com/jd"
    respx.get(url).mock(
        return_value=httpx.Response(200, text="<body><main>Senior Engineer</main></body>")
    )
    llm = FakeLLMClient(default_response=json.dumps(_LLM_RESPONSE))
    client = httpx.Client(timeout=2.0)
    try:
        req = parse_jd_url(url, llm=llm, http_client=client)
    finally:
        client.close()
    assert req.source_url == url
