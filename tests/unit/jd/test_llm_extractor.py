"""Tests for ``extract_with_llm`` and the LLM-response parser."""

from __future__ import annotations

import json

import pytest

from tailor_core.jd.llm_extractor import (
    SYSTEM_PROMPT,
    ExtractionError,
    _parse_extraction_payload,
    extract_with_llm,
)
from tailor_core.llm.client import FakeLLMClient

_VALID_RESPONSE = {
    "title": "Senior Software Engineer",
    "company": "Acme Corp",
    "required_skills": ["Python", "Postgres"],
    "nice_to_have_skills": ["Rust"],
    "must_haves": ["AU citizen"],
    "employer_vocabulary": ["first principles", "high-trust environment"],
}


def test_extract_with_llm_passes_system_prompt_and_text() -> None:
    llm = FakeLLMClient(default_response=json.dumps(_VALID_RESPONSE))

    result = extract_with_llm("the JD text", llm)

    assert result.title == "Senior Software Engineer"
    assert llm.calls[0][0] == SYSTEM_PROMPT
    assert llm.calls[0][1] == "the JD text"


def test_extract_with_llm_returns_validated_extraction() -> None:
    llm = FakeLLMClient(default_response=json.dumps(_VALID_RESPONSE))

    result = extract_with_llm("text", llm)

    assert result.company == "Acme Corp"
    assert result.required_skills == ["Python", "Postgres"]
    assert result.nice_to_have_skills == ["Rust"]
    assert result.must_haves == ["AU citizen"]
    assert result.employer_vocabulary == ["first principles", "high-trust environment"]


def test_extract_with_llm_handles_null_company() -> None:
    payload = dict(_VALID_RESPONSE, company=None)
    llm = FakeLLMClient(default_response=json.dumps(payload))

    result = extract_with_llm("text", llm)
    assert result.company is None


def test_extract_with_llm_strips_markdown_fences() -> None:
    fenced = f"```json\n{json.dumps(_VALID_RESPONSE)}\n```"
    llm = FakeLLMClient(default_response=fenced)

    result = extract_with_llm("text", llm)
    assert result.title == "Senior Software Engineer"


def test_extract_with_llm_strips_unlabelled_code_fence() -> None:
    fenced = f"```\n{json.dumps(_VALID_RESPONSE)}\n```"
    llm = FakeLLMClient(default_response=fenced)

    result = extract_with_llm("text", llm)
    assert result.title == "Senior Software Engineer"


def test_parse_extraction_payload_raises_on_empty() -> None:
    with pytest.raises(ExtractionError, match="empty"):
        _parse_extraction_payload("")


def test_parse_extraction_payload_raises_on_invalid_json() -> None:
    with pytest.raises(ExtractionError, match="not valid JSON"):
        _parse_extraction_payload("{not json")


def test_parse_extraction_payload_raises_on_non_object() -> None:
    with pytest.raises(ExtractionError, match="not a JSON object"):
        _parse_extraction_payload("[1, 2, 3]")


def test_parse_extraction_payload_raises_on_missing_required_field() -> None:
    payload = dict(_VALID_RESPONSE)
    del payload["title"]
    with pytest.raises(ExtractionError, match="schema validation"):
        _parse_extraction_payload(json.dumps(payload))


def test_parse_extraction_payload_ignores_extra_fields() -> None:
    """The model returns extra fields sometimes; we tolerate them."""
    payload = dict(_VALID_RESPONSE, extra_field="ignored")
    result = _parse_extraction_payload(json.dumps(payload))
    assert result.title == "Senior Software Engineer"
