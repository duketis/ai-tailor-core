"""Verifier scaffolding.

Two passes built around the same data model:

- A text-mode pass: a second LLM call reviews the agent's structured output
  the way a careful human would. Each consumer supplies its own ``SYSTEM_PROMPT``
  + structured input; the lib provides the JSON-parse, schema-validate, result-
  merging, and page-count-overflow helpers.
- A vision pass: pages of the rendered PDF go to the Anthropic messages API
  for a visual review. Consumer supplies the ``SYSTEM_PROMPT`` describing what
  to look for; the lib rasterises and dispatches.

Public surface lives in ``tailor_core.verifier.models`` (the result models),
``tailor_core.verifier.scaffold`` (the text-mode helpers), and
``tailor_core.verifier.vision`` (the vision helpers).
"""

from __future__ import annotations
