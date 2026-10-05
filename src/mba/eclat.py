"""Eclat frequent-itemset mining via vertical tid-list intersections."""

from __future__ import annotations

from typing import Hashable

Item = Hashable
Itemset = frozenset[Item]
TidList = set[int]


def eclat(
    transactions: list[Itemset],
    min_support: float = 0.01,
    max_len: int | None = None,
) -> dict[Itemset, float]:
    """Mine frequent itemsets with Eclat (depth-first tid-list intersections).

    Returns the same ``dict[frozenset, support]`` contract as Apriori / FP-Growth.
    """
    if not transactions:
        return {}
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be in (0, 1]")

    n = len(transactions)
    min_count = max(1, int(np_ceil(min_support * n)))

    # Vertical database: item -> set of transaction ids
    tidlists: dict[Item, TidList] = {}
    for tid, basket in enumerate(transactions):
        for item in basket:
            tidlists.setdefault(item, set()).add(tid)

    # Prune infrequent singletons
    tidlists = {i: tids for i, tids in tidlists.items() if len(tids) >= min_count}

    frequent: dict[Itemset, float] = {
        frozenset([item]): len(tids) / n for item, tids in tidlists.items()
    }

    # Sort items by ascending support (classic Eclat heuristic)
    items_sorted = sorted(tidlists.keys(), key=lambda i: len(tidlists[i]))

    def _mine(prefix: list[Item], candidates: list[Item], prefix_tids: TidList | None) -> None:
        for i, item in enumerate(candidates):
            new_prefix = prefix + [item]
            if len(new_prefix) == 1:
                new_tids = tidlists[item]
            else:
                assert prefix_tids is not None
                new_tids = prefix_tids & tidlists[item]
            if len(new_tids) < min_count:
                continue
            frequent[frozenset(new_prefix)] = len(new_tids) / n
            if max_len is not None and len(new_prefix) >= max_len:
                continue
            # Only intersect with items after current (lexicographic depth-first)
            _mine(new_prefix, candidates[i + 1 :], new_tids)

    _mine([], items_sorted, None)
    return frequent


def np_ceil(x: float) -> int:
    """Ceiling without importing numpy (keep eclat lightweight)."""
    i = int(x)
    return i if i == x else i + 1
