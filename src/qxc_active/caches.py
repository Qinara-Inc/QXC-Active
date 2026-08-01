# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Qinara

"""Bounded FIFO insertion for the package's memoisation tables.

Every memoisation dict in the package admits new entries through
:func:`cache_put`, which evicts the oldest entry once the table is full.
Eviction is deliberately FIFO, not LRU: with FIFO the eviction order
depends only on the insertion order, which is itself a deterministic
function of the input, so the package's determinism guarantees are
unaffected. (LRU eviction order would depend on the *lookup* pattern,
which can differ between otherwise-identical runs — e.g. warm vs cold
caches across successive reductions in one process.)

Caches here are pure memoisation: evicting an entry can never change any
result, only the cost of recomputing it.
"""

from __future__ import annotations


def cache_put(table: dict, key, value, limit: int) -> None:
    """Insert ``key: value`` into ``table``, evicting oldest-first at ``limit``.

    Python dicts preserve insertion order, so the first key in iteration
    order is always the oldest surviving entry.
    """
    if key in table:
        table[key] = value
        return
    while len(table) >= limit:
        del table[next(iter(table))]
    table[key] = value
