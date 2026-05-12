"""Run + RunEvent + TailorRequest + RenderResult pydantic models.

``Run`` is generic over the consumer's tailored-output type so resumeai
holds ``Run[TailoredResume]`` and coverletterai holds
``Run[TailoredCoverLetter]``; pydantic v2 serialises both transparently.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tailor_core.jd.models import JobRequirements
from tailor_core.verifier.models import VerificationResult


class RunStatus(StrEnum):
    """Lifecycle states a run moves through. Strictly forward-progressing."""

    PENDING = "pending"
    FETCHING_JD = "fetching_jd"
    PARSING_JD = "parsing_jd"
    LOADING_CONTEXT = "loading_context"
    TAILORING = "tailoring"
    RENDERING = "rendering"
    VERIFYING = "verifying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (RunStatus.SUCCEEDED, RunStatus.FAILED)


class TailorRequest(BaseModel):
    """Request body for a tailoring run.

    Either ``jd_url`` (we'll fetch it) or ``jd_text`` (paste-in) must be
    supplied. Consumers subclass to add app-specific fields (eg
    coverletterai adds ``resume_run_id`` / ``resume_payload`` so the
    letter can be grounded in a previously-tailored resume); ``extra=
    'allow'`` lets the subclass's extra keys flow through when an
    instance is stored inside ``Run.request`` typed as the base.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    jd_url: str | None = None
    jd_text: str | None = None
    model: str | None = None

    @model_validator(mode="after")
    def _exactly_one_jd_source(self) -> Self:
        has_url = bool(self.jd_url and self.jd_url.strip())
        has_text = bool(self.jd_text and self.jd_text.strip())
        if has_url == has_text:
            raise ValueError("supply exactly one of jd_url or jd_text")
        return self


class RunEvent(BaseModel):
    """One progress event. Published as the run moves through the pipeline."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    status: RunStatus
    detail: str = ""
    at: datetime


# -- renderer shapes -------------------------------------------------------


class RenderStatus(StrEnum):
    """One per section we attempted to render."""

    REPLACED = "replaced"
    """Section was found in the template; old content removed, new content inserted."""

    SKIPPED_EMPTY = "skipped_empty"
    """The structured output had nothing to render for this section."""

    NOT_FOUND = "not_found"
    """No heading in the template matched any of this section's aliases."""


class RenderDiff(BaseModel):
    """One diff entry per section the renderer attempted."""

    model_config = ConfigDict(frozen=True)

    kind: str = Field(min_length=1)
    heading: str | None = None
    status: RenderStatus
    before_chars: int = 0
    after_chars: int = 0


class RenderResult(BaseModel):
    """The output of a full render.

    The rendered PDF is written to disk by the renderer and addressed via
    ``doc_url``. We don't carry the raw bytes on the model because the
    ``Run`` record is JSON-serialised into SQLite and PDF bytes aren't
    UTF-8.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(min_length=1)
    doc_url: str = Field(min_length=1)
    pdf_size_bytes: int = Field(default=0, ge=0)
    diffs: tuple[RenderDiff, ...] = ()


class Run[TailoredT: BaseModel](BaseModel):
    """Persisted run state. Generic over the consumer's tailored type.

    Each consumer binds ``TailoredT`` to its concrete output model
    (eg resumeai uses ``Run[TailoredResume]``, coverletterai uses
    ``Run[TailoredCoverLetter]``). Pydantic v2's generic-model support
    keeps validation honest at the parameterised type.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    request: TailorRequest
    status: RunStatus = RunStatus.PENDING
    created_at: datetime
    updated_at: datetime
    detail: str = ""
    error: str | None = None
    requirements: JobRequirements | None = None
    tailored: TailoredT | None = None
    result: RenderResult | None = None
    verification: VerificationResult | None = None
    # Vision-based QC on the rendered PDF. Optional because the vision
    # pass is best-effort -- missing OAuth token, SDK install issue, or
    # API failure all degrade silently to ``None``.
    vision_verification: VerificationResult | None = None
