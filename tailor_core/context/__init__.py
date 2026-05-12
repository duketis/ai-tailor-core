"""User context — the "everything Jonathan has ever done" side of tailoring.

The Phase 4 tailoring agent matches a :class:`~tailor_core.jd.models.JobRequirements`
against a :class:`~tailor_core.context.models.UserContext` produced by the
:class:`~tailor_core.context.store.ContextStore`. This package is the loader +
storage layer for the latter.

Sources (all under a single root directory, by default ``UserContext/``):

- ``resume.yaml`` — structured base resume (name, contact, skills, education).
- ``work_history/*.md`` — per-role narratives. YAML frontmatter for metadata,
  markdown body with bullets for the day-to-day output.
- ``git_audit/*.md`` — per-repo audit notes (what shipped, what mattered).
- ``cover_letters/*.md`` — past cover letters keyed by role/company.

The store re-loads automatically when any source file is modified — the
agent always sees Jonathan's latest context, no restart needed.
"""

from __future__ import annotations
