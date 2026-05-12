---
project: sample-tool
url: https://github.com/alex/sample-tool
status: v0.4.x (open-source, MIT)
stack: Python, FastAPI, SQLite
---

A sample personal project used in the loader's integration fixture. It's
the **code-review showcase** for the fictitious candidate -- recruiters
can click through to GitHub and audit the code.

**Important framing for resumes / cover letters:**
- Always describe as "open-source", not "in production".
- Pair with an architecture-focused private project if available.

## What sample-tool does

- Ingests sample data from public APIs and stores normalised rows in SQLite.
- Exposes a FastAPI surface for queries plus a small CLI.

## Engineering discipline

- 120+ tests at 92% coverage
- mypy strict, ruff strict, conventional commits
