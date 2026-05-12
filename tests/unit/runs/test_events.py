"""RunEventBus pub/sub tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from tailor_core.runs.events import RunEventBus
from tailor_core.runs.models import RunEvent, RunStatus


def _event(run_id: str = "run_x", *, status: RunStatus = RunStatus.TAILORING) -> RunEvent:
    return RunEvent(
        run_id=run_id, status=status, detail="working", at=datetime(2026, 5, 9, tzinfo=UTC)
    )


@pytest.mark.asyncio
async def test_subscriber_receives_published_events() -> None:
    bus = RunEventBus()
    received: list[RunEvent] = []

    async def consume() -> None:
        async for event in bus.subscribe("run_x"):
            received.append(event)

    consumer = asyncio.create_task(consume())
    # Yield to let the subscriber register.
    await asyncio.sleep(0)

    await bus.publish(_event())
    await bus.close("run_x")
    await consumer

    assert len(received) == 1
    assert received[0].status is RunStatus.TAILORING


@pytest.mark.asyncio
async def test_publish_to_run_with_no_subscribers_is_noop() -> None:
    bus = RunEventBus()
    # Must not raise.
    await bus.publish(_event(run_id="never-subscribed"))


@pytest.mark.asyncio
async def test_close_to_run_with_no_subscribers_is_noop() -> None:
    bus = RunEventBus()
    await bus.close("never-subscribed")  # must not raise


@pytest.mark.asyncio
async def test_multiple_subscribers_each_get_every_event() -> None:
    bus = RunEventBus()
    received: list[list[RunEvent]] = [[], []]

    async def consume(idx: int) -> None:
        async for event in bus.subscribe("run_x"):
            received[idx].append(event)

    consumers = [asyncio.create_task(consume(i)) for i in range(2)]
    await asyncio.sleep(0)

    await bus.publish(_event(status=RunStatus.PARSING_JD))
    await bus.publish(_event(status=RunStatus.TAILORING))
    await bus.close("run_x")
    await asyncio.gather(*consumers)

    assert [e.status for e in received[0]] == [RunStatus.PARSING_JD, RunStatus.TAILORING]
    assert [e.status for e in received[1]] == [RunStatus.PARSING_JD, RunStatus.TAILORING]


@pytest.mark.asyncio
async def test_cancelled_subscriber_cleans_up_its_queue() -> None:
    """Cancellation mid-iteration must not leak the queue back into the bus."""
    bus = RunEventBus()

    async def consume() -> None:
        async for _event in bus.subscribe("run_x"):
            return

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)

    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    # No further publishes should reach a stale queue. The internal map
    # should also be cleaned out.
    assert "run_x" not in bus._subscribers


@pytest.mark.asyncio
async def test_one_cancelled_subscriber_does_not_remove_others() -> None:
    """When several subscribers exist and one is cancelled, the survivors
    keep receiving events."""
    bus = RunEventBus()
    received: list[RunEvent] = []

    async def long_consumer() -> None:
        async for event in bus.subscribe("run_x"):
            received.append(event)

    async def short_consumer() -> None:
        async for _event in bus.subscribe("run_x"):
            return

    long = asyncio.create_task(long_consumer())
    short = asyncio.create_task(short_consumer())
    await asyncio.sleep(0)

    short.cancel()
    with pytest.raises(asyncio.CancelledError):
        await short

    await bus.publish(_event())
    await bus.close("run_x")
    await long

    assert len(received) == 1


@pytest.mark.asyncio
async def test_subscriber_only_sees_events_for_its_own_run() -> None:
    bus = RunEventBus()
    received: list[RunEvent] = []

    async def consume() -> None:
        async for event in bus.subscribe("run_a"):
            received.append(event)

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)

    await bus.publish(_event(run_id="run_b"))
    await bus.publish(_event(run_id="run_a"))
    await bus.close("run_a")
    await consumer

    assert len(received) == 1
    assert received[0].run_id == "run_a"
