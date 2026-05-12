"""Pydantic models + enums for the JD pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Seniority(StrEnum):
    """Coarse seniority bucket inferred from title + body text."""

    UNKNOWN = "unknown"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"
    PRINCIPAL = "principal"


class RoleType(StrEnum):
    """Coarse role category. Driven by title + body keywords."""

    UNKNOWN = "unknown"
    ENGINEERING = "engineering"
    DATA = "data"
    DESIGN = "design"
    PRODUCT = "product"
    SALES = "sales"
    MARKETING = "marketing"
    OPERATIONS = "operations"
    SUPPORT = "support"
    HR = "hr"
    FINANCE = "finance"
    LEGAL = "legal"
    HEALTHCARE = "healthcare"
    OTHER = "other"


class EmploymentType(StrEnum):
    UNKNOWN = "unknown"
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERN = "intern"
    CASUAL = "casual"


class RemoteType(StrEnum):
    UNKNOWN = "unknown"
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"


class FetchedJD(BaseModel):
    """The output of the fetcher — raw HTML + cleaned plain text + metadata."""

    model_config = ConfigDict(frozen=True)

    source_url: str
    raw_html: str
    cleaned_text: str
    fetched_at: datetime
    ats: str  # "greenhouse", "lever", "ashby", "workable", "smartrecruiters", "unknown"


class JobRequirements(BaseModel):
    """The structured shape downstream phases consume.

    Field-by-field source of truth:

    - ``title``, ``company``, ``required_skills``, ``nice_to_have_skills``,
      ``must_haves``, ``employer_vocabulary``: extracted by the LLM.
    - ``location``, ``seniority``, ``role_type``, ``employment_type``,
      ``remote_type``: extracted deterministically (regex / keyword scan)
      with the LLM as a no-op fallback path for Phase 2.
    """

    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1)
    company: str | None = None
    location: str | None = None
    role_type: RoleType = RoleType.UNKNOWN
    seniority: Seniority = Seniority.UNKNOWN
    employment_type: EmploymentType = EmploymentType.UNKNOWN
    remote_type: RemoteType = RemoteType.UNKNOWN
    required_skills: tuple[str, ...] = ()
    nice_to_have_skills: tuple[str, ...] = ()
    must_haves: tuple[str, ...] = ()
    employer_vocabulary: tuple[str, ...] = ()
    raw_text: str = ""
    source_url: str | None = None
