"""Bounded-concurrency async helpers.

The cognitive loop fires many *independent* LLM/embedding calls (per-focal-point
insights, per-neighbor memory evolution, multi-agent turns) that were awaited
serially. These helpers run them concurrently with a cap so the wall-clock is the
slowest call, not the sum — without unbounded fan-out against a rate-limited API.
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


async def gather_bounded(coros: Iterable[Awaitable[R]], limit: int = 8) -> list[R]:
    """Run awaitables concurrently, at most ``limit`` at once; preserves input order."""
    coros = list(coros)
    if not coros:
        return []
    if limit <= 0:
        limit = len(coros)
    sem = asyncio.Semaphore(limit)

    async def _run(co: Awaitable[R]) -> R:
        async with sem:
            return await co

    return await asyncio.gather(*(_run(c) for c in coros))


async def amap(fn: Callable[[T], Awaitable[R]], items: Iterable[T], concurrency: int = 8) -> list[R]:
    """Async map ``fn`` over ``items`` with bounded concurrency; order preserved."""
    return await gather_bounded((fn(it) for it in items), limit=concurrency)
