"""Scan local project directories — README, structure, git log — into text.

The user registers an absolute path to a project they've worked on; the
scanner walks the directory and produces a single plain-text summary
(README content + top-level file listing + ``git log`` of their authored
commits, when ``.git/`` is present). The summary is then stored as a
``ContextFile`` so it surfaces to the tailoring agent alongside uploaded
files and the master template.

Privacy: the scanner deliberately skips secret-likely files / dirs
(``.env``, ``credentials*``, ``node_modules``, ``__pycache__``, etc.) and
caps each section's character budget so a giant monorepo can't dump
gigabytes into the agent's prompt.
"""

from __future__ import annotations
