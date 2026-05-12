"""Frozen Pydantic models for the verifier's output."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class VerificationStatus(StrEnum):
    """The verifier's headline judgment."""

    PASSED = "passed"
    """Agent's output is acceptable. No blocking issues."""

    CONCERNS = "concerns"
    """Output is usable but the verifier flagged worth-knowing issues."""

    FAILED = "failed"
    """Output has at least one blocking issue (fabrication, missing
    must-have, etc.). The user should review before sending."""


class IssueSeverity(StrEnum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class VerificationIssue(BaseModel):
    """One concern the verifier raised."""

    model_config = ConfigDict(frozen=True)

    severity: IssueSeverity
    category: str = Field(min_length=1)
    message: str = Field(min_length=1)
    suggestion: str = ""


class VerificationResult(BaseModel):
    """The full output of one verification pass."""

    model_config = ConfigDict(frozen=True)

    status: VerificationStatus
    summary: str = ""
    issues: tuple[VerificationIssue, ...] = ()
    rationale: str = ""
