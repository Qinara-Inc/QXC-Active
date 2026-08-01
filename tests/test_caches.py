"""FIFO-bounded memoisation table behaviour."""

from __future__ import annotations

from qxc_active.caches import cache_put


def test_evicts_oldest_first_at_limit():
    table: dict = {}
    for i in range(5):
        cache_put(table, i, i * 10, limit=3)
    # 0 and 1 (the oldest) were evicted, insertion order preserved
    assert list(table.items()) == [(2, 20), (3, 30), (4, 40)]


def test_updating_existing_key_keeps_position_and_size():
    table: dict = {}
    for i in range(3):
        cache_put(table, i, i, limit=3)
    cache_put(table, 0, 99, limit=3)  # update, not insert: nothing evicted
    assert list(table.items()) == [(0, 99), (1, 1), (2, 2)]
    cache_put(table, 3, 3, limit=3)  # now 0 is still the oldest and goes
    assert list(table.items()) == [(1, 1), (2, 2), (3, 3)]


def test_never_exceeds_limit_even_if_overfull():
    table = {i: i for i in range(10)}
    cache_put(table, "new", 1, limit=3)
    assert len(table) == 3
    assert table["new"] == 1
