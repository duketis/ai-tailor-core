"""Job-description ingestion + structured extraction.

Public surface:

- :func:`~tailor_core.jd.fetcher.fetch_jd` — pull HTML from a URL and return a
  :class:`~tailor_core.jd.models.FetchedJD` (raw HTML + cleaned plain text).
- :func:`~tailor_core.jd.parser.parse_jd_text` — turn cleaned plain text into a
  :class:`~tailor_core.jd.models.JobRequirements` via deterministic regex passes
  + a single LLM call for the open-ended fields.
- :func:`~tailor_core.jd.parser.parse_jd_url` — convenience that does fetch +
  parse in one go.
"""

from __future__ import annotations
