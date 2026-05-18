"""Single-shot LLM extraction for the open-ended JD fields.

The deterministic extractors handle the closed enums (location, seniority,
role type, employment type, remote type). This module asks the LLM for the
open-ended ones (title, company, skills, must-haves, employer vocabulary)
and parses its JSON response.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

if TYPE_CHECKING:
    from tailor_core.llm.client import LLMClient


# A thin or unresolved JD (eg a blind apply-portal URL) can leave the LLM
# with no advertised title to extract. A blank title must NOT hard-fail the
# whole tailoring pipeline -- the downstream resume/cover-letter agents can
# still produce a useful artefact from the skills + must-haves. Coerce to
# this placeholder so the run proceeds (fail-soft) rather than crashing
# every retry identically.
TITLE_FALLBACK = "the advertised role"


SYSTEM_PROMPT = """\
You extract structured data from job descriptions for a resume-tailoring tool.

Given the cleaned plain-text JD that follows, respond with ONLY a single JSON
object (no markdown fences, no commentary, no preamble). The object must
match this schema exactly:

{
  "title": "string — the job title as advertised",
  "company": "string or null — the hiring company, or null if not stated",
  "required_skills": ["array of short strings — technical or professional \
skills explicitly required"],
  "nice_to_have_skills": ["array of short strings — preferred / 'bonus' skills"],
  "must_haves": ["array of short strings — non-skill requirements like \
'AU citizen', '5+ years industry experience', 'security clearance'"],
  "employer_vocabulary": ["array of short strings — distinctive phrases the \
JD repeats that a tailored resume should echo (eg 'customer-obsessed', \
'first-principles', 'high-trust environment')"]
}

Rules:
- Skills: prefer the exact wording the JD uses ('TypeScript' not 'TS'; \
'Postgres' if they wrote 'Postgres', 'PostgreSQL' if they wrote 'PostgreSQL').
- Don't fabricate. If a field has no signal in the JD, return [] (or null \
for company). Don't guess.
- Keep arrays under 20 items; pick the most prominent.
- Output the JSON object and nothing else.
"""


class ExtractionError(RuntimeError):
    """Raised when the LLM response can't be parsed into the expected shape."""


class _LLMExtraction(BaseModel):
    """Internal validation shape for the LLM response."""

    model_config = ConfigDict(extra="ignore")

    title: str = Field(default=TITLE_FALLBACK, min_length=1)
    company: str | None = None
    required_skills: list[str] = []
    nice_to_have_skills: list[str] = []
    must_haves: list[str] = []
    employer_vocabulary: list[str] = []

    @field_validator("title", mode="before")
    @classmethod
    def _coerce_blank_title(cls, value: object) -> str:
        """Trim the title; coerce a missing/blank/non-string title to a fallback.

        Runs before ``min_length`` so a thin JD that yields an empty title
        produces a usable placeholder instead of a hard ``ExtractionError``
        that crashes every retry identically.
        """
        if not isinstance(value, str):
            return TITLE_FALLBACK
        trimmed = value.strip()
        return trimmed or TITLE_FALLBACK


# Some models still wrap JSON in ```json fences despite system instructions.
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def extract_with_llm(text: str, llm: LLMClient) -> _LLMExtraction:
    """Run the extraction prompt and return a validated extraction record."""
    raw = llm.complete(system=SYSTEM_PROMPT, user=text).strip()
    return _parse_extraction_payload(raw)


def _parse_extraction_payload(raw: str) -> _LLMExtraction:
    """Parse the model's text response into a validated extraction record."""
    if not raw:
        raise ExtractionError("LLM returned an empty response")

    payload = _FENCE_RE.sub(r"\1", raw).strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ExtractionError(
            f"LLM response was not valid JSON: {exc.msg} — got {payload[:200]!r}"
        ) from exc

    if not isinstance(data, dict):
        raise ExtractionError(f"LLM response was not a JSON object — got {type(data).__name__}")

    try:
        return _LLMExtraction.model_validate(data)
    except ValidationError as exc:
        raise ExtractionError(f"LLM response failed schema validation: {exc}") from exc


__all__ = [
    "SYSTEM_PROMPT",
    "TITLE_FALLBACK",
    "ExtractionError",
    "extract_with_llm",
]
