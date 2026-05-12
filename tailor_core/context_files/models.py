"""Frozen Pydantic models for uploaded context files."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ContextFileKind(StrEnum):
    """How we extracted text from the file. Drives display + grouping."""

    PDF = "pdf"
    CSV = "csv"
    TEXT = "text"
    MARKDOWN = "markdown"


class ContextFile(BaseModel):
    """One uploaded file."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    kind: ContextFileKind
    extracted_text: str
    byte_size: int = Field(ge=0)
    tags: tuple[str, ...] = ()
    uploaded_at: datetime
    note: str = ""
