"""User-uploaded context files (PDFs, CSVs, text, markdown).

The Settings UI lets the user upload arbitrary files (a git audit CSV, a
project portfolio PDF, hand-written notes about a side project, etc).
On upload we extract the file's text content and persist both the
extracted text and the original bytes; the tailoring agent receives the
extracted text alongside the structured ``UserContext`` from the
filesystem ``UserContext/`` directory.

The user can tag each file (eg. ``project:resumeai``, ``role:engineering``)
and the agent decides which to include based on the JD.
"""

from __future__ import annotations
