"""Smoke test: the package imports and exposes ``__version__``."""

from __future__ import annotations

import tailor_core


def test_version_is_a_nonempty_semver_string() -> None:
    assert isinstance(tailor_core.__version__, str)
    assert tailor_core.__version__.count(".") == 2
    assert all(part.isdigit() for part in tailor_core.__version__.split("."))
