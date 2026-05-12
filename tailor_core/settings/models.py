"""Pydantic model for runtime settings.

Apps subclass ``BaseRuntimeSettings`` to add their own user-tunable keys.
For example, ``resumeai.settings.RuntimeSettings`` adds ``template_name``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BaseRuntimeSettings(BaseModel):
    """User-tunable runtime settings persisted in the local SQLite DB.

    Subclass to add app-specific keys; the lib's store round-trips the
    subclass intact via pydantic's ``model_validate_json`` / ``model_dump_json``.
    """

    model_config = ConfigDict(frozen=True)

    model: str | None = None
