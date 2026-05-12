"""LLM-judge scaffolding -- text-mode verification helpers.

The contract:

1. Consumer composes a ``SYSTEM_PROMPT`` describing the QC rubric and a
   ``user_prompt`` carrying the input to review (a JD + the tailored
   output, eg).
2. :func:`evaluate_judgement` sends both to the LLM, parses the JSON
   response, and returns a :class:`VerificationResult`.
3. Consumer optionally calls :func:`check_pdf_length` against the
   rendered PDF and :func:`merge_issue` to fold the result in.
4. If the LLM call itself fails, :func:`fallback_concerns_result`
   synthesises a non-blocking ``CONCERNS`` result so the run still
   surfaces an output to the user.

This module is intentionally generic; nothing here knows the difference
between a resume verifier and a cover-letter verifier.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from pydantic import ValidationError

from tailor_core.verifier.models import (
    IssueSeverity,
    VerificationIssue,
    VerificationResult,
    VerificationStatus,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tailor_core.llm.client import LLMClient


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


class VerifierError(RuntimeError):
    """Raised when the verifier's response can't be parsed."""


def evaluate_judgement(
    *,
    system_prompt: str,
    user_prompt: str,
    llm: LLMClient,
    model: str | None = None,
) -> VerificationResult:
    """Run a single LLM-judge pass and parse the response.

    Raises :class:`VerifierError` if the LLM returns text we can't parse
    into a :class:`VerificationResult`. Callers can wrap that with
    :func:`fallback_concerns_result` to keep the run non-blocking.
    """
    raw = llm.complete(system=system_prompt, user=user_prompt, model=model)
    return parse_verifier_response(raw)


def parse_verifier_response(raw: str) -> VerificationResult:
    """Parse the model's text response into a :class:`VerificationResult`.

    Strips an optional Markdown fence, decodes the JSON body, and
    validates against the :class:`VerificationResult` schema.
    """
    text = raw.strip()
    if not text:
        raise VerifierError("verifier returned an empty response")

    payload = _FENCE_RE.sub(r"\1", text).strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise VerifierError(
            f"verifier response was not valid JSON: {exc.msg} — got {payload[:200]!r}"
        ) from exc

    if not isinstance(data, dict):
        raise VerifierError(f"verifier response was not a JSON object — got {type(data).__name__}")

    try:
        return VerificationResult.model_validate(data)
    except ValidationError as exc:
        raise VerifierError(f"verifier response failed schema validation: {exc}") from exc


def merge_issue(result: VerificationResult, new_issue: VerificationIssue) -> VerificationResult:
    """Append a programmatic issue to an LLM-verifier result.

    Promotes the status if the new issue is more severe than the
    existing finding (a ``warn`` added to a ``passed`` result flips
    the result to ``concerns``).
    """
    bumped_status = result.status
    is_warn = new_issue.severity is IssueSeverity.WARN
    is_error = new_issue.severity is IssueSeverity.ERROR
    if is_warn and result.status is VerificationStatus.PASSED:
        bumped_status = VerificationStatus.CONCERNS
    elif is_error and result.status is not VerificationStatus.FAILED:
        bumped_status = VerificationStatus.FAILED
    return result.model_copy(
        update={
            "status": bumped_status,
            "issues": (*result.issues, new_issue),
        }
    )


def check_pdf_length(
    pdf_path: Path,
    *,
    target_max_pages: int,
    overflow_message: str | None = None,
    overflow_suggestion: str | None = None,
) -> VerificationIssue | None:
    """Programmatically check rendered PDF length against the target.

    Returns ``None`` when the PDF is at or under ``target_max_pages``.
    Returns a ``warn``-severity :class:`VerificationIssue` when it
    overflows. Failures reading the PDF degrade silently (return
    ``None``) -- the LLM verifier already pinged on the run, no point
    fabricating a second failure on top.

    ``overflow_message`` / ``overflow_suggestion`` let consumers
    customise the surfaced wording; the defaults are app-agnostic.
    """
    from pypdf import PdfReader  # noqa: PLC0415
    from pypdf.errors import PdfReadError  # noqa: PLC0415

    try:
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
    except (OSError, PdfReadError):
        return None
    if page_count <= target_max_pages:
        return None

    message = overflow_message or (
        f"Rendered PDF is {page_count} pages; target is ≤{target_max_pages}."
    )
    suggestion = overflow_suggestion or "Trim content and re-render."
    return VerificationIssue(
        severity=IssueSeverity.WARN,
        category="page_overflow",
        message=message,
        suggestion=suggestion,
    )


def fallback_concerns_result(reason: str) -> VerificationResult:
    """Synthesise a ``CONCERNS`` result when the verifier itself fails.

    Used by orchestrators: if the LLM call errors or the response is
    malformed, we don't want to block the whole run on QC infrastructure
    -- the user gets a CONCERNS-status run with the failure reason as
    the only issue, and can read the underlying tailored output
    themselves.
    """
    return VerificationResult(
        status=VerificationStatus.CONCERNS,
        summary="Verifier itself failed — review the rendered doc manually.",
        issues=(
            VerificationIssue(
                severity=IssueSeverity.WARN,
                category="verifier_failure",
                message=reason,
                suggestion="Re-run; if this keeps happening, raise an issue on the repo.",
            ),
        ),
        rationale="QC pass couldn't complete; surfacing as CONCERNS rather than blocking.",
    )
