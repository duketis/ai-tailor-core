"""Shared building blocks for AI-driven tailoring pipelines.

The package is consumed by ``resumeai`` (JD → tailored resume PDF) and
``coverletterai`` (JD + tailored resume → tailored cover letter PDF). It holds
everything those two apps would otherwise duplicate: the ``claude`` CLI
subprocess wrapper, the JD fetcher / parser / extractor, the run-orchestration
skeleton, the verifier scaffolding, the user-context loader, the local-project
scanner, the uploaded-context-file store, and the shared Docker bits.

App-specific things (the LaTeX template, the agent prompt, the FastAPI routes)
live in each consumer, not here.
"""

from __future__ import annotations

__version__ = "0.1.2"

__all__ = ["__version__"]
