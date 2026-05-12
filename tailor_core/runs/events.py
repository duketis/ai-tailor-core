"""In-memory pub/sub for run events.

Each run gets zero-or-more subscribers; the orchestrator publishes events
as the pipeline progresses; subscribers (typically an SSE route) receive
them via async queues. When the run terminates the publisher pushes a
sentinel ``None`` to each queue so subscribers can close cleanly.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from tailor_core.runs.models import RunEvent


class RunEventBus:
    """Per-run fan-out. Lock-protected so multiple concurrent SSE consumers
    don't trample each other when the publisher fans out an event."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[RunEvent | None]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def publish(self, event: RunEvent) -> None:
        async with self._lock:
            queues = list(self._subscribers.get(event.run_id, ()))
        for queue in queues:
            await queue.put(event)

    async def close(self, run_id: str) -> None:
        """Signal end-of-stream to all subscribers of ``run_id``."""
        async with self._lock:
            queues = list(self._subscribers.pop(run_id, ()))
        for queue in queues:
            await queue.put(None)

    async def subscribe(self, run_id: str) -> AsyncIterator[RunEvent]:
        """Async generator that yields events until end-of-stream."""
        queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()
        async with self._lock:
            self._subscribers[run_id].append(queue)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
        finally:
            async with self._lock:
                if queue in self._subscribers.get(run_id, ()):
                    self._subscribers[run_id].remove(queue)
                    if not self._subscribers[run_id]:
                        self._subscribers.pop(run_id, None)
