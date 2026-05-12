"""Runtime settings -- pydantic model + persistent store.

The lib ships a ``BaseRuntimeSettings`` model carrying the keys every consumer
needs (just ``model`` -- the LLM model override). Apps subclass to add their
own fields. The ``SqliteSettingsStore`` / ``InMemorySettingsStore`` are
generic over the settings type so a consumer's subclass round-trips intact.
"""

from __future__ import annotations
