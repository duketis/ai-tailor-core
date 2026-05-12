"""A minimal ``TailoredT`` stub used across the runs test suite.

We need a concrete pydantic ``BaseModel`` to bind ``Run[TailoredT]`` /
``BaseOrchestrator[TailoredT]`` against. The stub stands in for whatever
the consumer (resumeai's ``TailoredResume``, coverletterai's
``TailoredCoverLetter``) provides.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StubTailored(BaseModel):
    """Tiny pydantic model to stand in for a consumer's tailored type."""

    model_config = ConfigDict(frozen=True)

    label: str = "stub"
