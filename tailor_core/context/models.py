"""Frozen Pydantic models for the user-context tree."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Contact(BaseModel):
    """Contact block from ``resume.yaml``.

    Everything except ``email`` is optional — a tailored resume may include
    only what the role's location/seniority warrants (no phone for global
    remote roles, no postal address ever).
    """

    model_config = ConfigDict(frozen=True)

    email: str
    phone: str | None = None
    location: str | None = None
    linkedin: str | None = None
    github: str | None = None
    website: str | None = None


class Education(BaseModel):
    model_config = ConfigDict(frozen=True)

    institution: str
    degree: str
    field: str | None = None
    year_start: int | None = None
    year_end: int | None = None


class ResumeBase(BaseModel):
    """The ``resume.yaml`` shape — facts that don't change per-tailoring."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    headline: str | None = None
    contact: Contact
    skills: tuple[str, ...] = ()
    education: tuple[Education, ...] = ()
    certifications: tuple[str, ...] = ()


class WorkHistoryEntry(BaseModel):
    """One per-role narrative from ``work_history/<slug>.md``.

    The frontmatter carries the structured metadata; ``summary`` and
    ``bullets`` are extracted from the markdown body.
    """

    model_config = ConfigDict(frozen=True)

    slug: str = Field(min_length=1)
    title: str = Field(min_length=1)
    company: str = Field(min_length=1)
    start: str | None = None
    end: str | None = None
    location: str | None = None
    technologies: tuple[str, ...] = ()
    summary: str = ""
    bullets: tuple[str, ...] = ()
    raw_markdown: str = ""


class GitAuditEntry(BaseModel):
    """One per-repo audit from ``git_audit/<slug>.md``."""

    model_config = ConfigDict(frozen=True)

    slug: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    role: str | None = None
    period: str | None = None
    summary: str = ""
    raw_markdown: str = ""


class CoverLetterEntry(BaseModel):
    """One past cover letter from ``cover_letters/<slug>.md``."""

    model_config = ConfigDict(frozen=True)

    slug: str = Field(min_length=1)
    role: str | None = None
    company: str | None = None
    body: str = ""
    raw_markdown: str = ""


class ProjectEntry(BaseModel):
    """One personal project from ``projects/<slug>.md``.

    Projects live outside ``work_history/`` because they aren't paid
    engagements; they're the public (or private) side-projects a
    candidate uses to demonstrate range and engineering discipline.
    The ``body`` is preserved so the agent can read the candidate's
    positioning rules (e.g. "never link strategyminer.xyz") verbatim
    when deciding how to render the project's right-column.

    ``local_path`` (optional) points the loader at the project's local
    folder so a recursive scanner can pull README + structure + git log
    into ``scanned`` -- richer agent context than the .md alone.
    """

    model_config = ConfigDict(frozen=True)

    slug: str = Field(min_length=1)
    name: str = Field(min_length=1)
    url: str | None = None
    status: str | None = None
    stack: str | None = None
    summary: str = ""
    bullets: tuple[str, ...] = ()
    body: str = ""
    local_path: str | None = None
    scanned: str = ""


class UserContext(BaseModel):
    """The full materialised context the tailoring agent sees."""

    model_config = ConfigDict(frozen=True)

    resume: ResumeBase | None = None
    work_history: tuple[WorkHistoryEntry, ...] = ()
    git_audit: tuple[GitAuditEntry, ...] = ()
    cover_letters: tuple[CoverLetterEntry, ...] = ()
    projects: tuple[ProjectEntry, ...] = ()
    # Verbatim authoritative reference content. ``master_resume`` is the
    # candidate's plain-text master resume; ``reference_resumes`` are any
    # already-rendered tailored resumes the candidate considers known-good.
    # The agent reads these to ground its own output in the candidate's
    # exact phrasing rather than hallucinating.
    master_resume: str = ""
    reference_resumes: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        """True when nothing was loaded (no resume.yaml + no markdown files)."""
        return (
            self.resume is None
            and not self.work_history
            and not self.git_audit
            and not self.cover_letters
            and not self.projects
            and not self.master_resume
            and not self.reference_resumes
        )
