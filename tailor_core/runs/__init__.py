"""Run lifecycle scaffolding.

Public surface:

- :mod:`tailor_core.runs.models` -- ``Run`` (generic over the consumer's
  ``TailoredT``), ``RunStatus``, ``RunEvent``, ``TailorRequest``,
  ``RenderResult`` / ``RenderDiff`` / ``RenderStatus``.
- :mod:`tailor_core.runs.events` -- ``RunEventBus`` (in-memory per-run
  pub/sub for SSE streaming).
- :mod:`tailor_core.runs.store` -- ``RunsStore`` protocol + SQLite +
  in-memory backends, parameterised on ``TailoredT``.
- :mod:`tailor_core.runs.orchestrator` -- ``BaseOrchestrator`` template-
  method skeleton; subclasses implement ``_tailor`` / ``_render`` /
  ``_verify`` / ``_verify_visually`` hooks.
"""

from __future__ import annotations
