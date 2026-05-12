"""LLM access layer.

Single ``LLMClient`` ``Protocol`` that every caller depends on. The production
implementation (:class:`~tailor_core.llm.client.ClaudeCliClient`) spawns the
``claude`` CLI as a subprocess so callers use Anthropic's Max-subscription OAuth
instead of the per-token API. Tests use
:class:`~tailor_core.llm.client.FakeLLMClient`.
"""

from __future__ import annotations
