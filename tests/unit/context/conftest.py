"""Shared fixtures for the context tests.

The committed sample tree under ``tests/fixtures/sample_context/`` doubles
as both an integration-style fixture for the loader and as a "what does a
real UserContext look like" reference for documentation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SAMPLE_CONTEXT_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "sample_context"


@pytest.fixture
def sample_context_root() -> Path:
    return SAMPLE_CONTEXT_ROOT
