"""Classical Apriori frequent-itemset mining.

Unlike the original notebook (which enumerated all k-combinations on orders of
*exactly* size k), this implementation:

1. Keeps baskets of size >= k
2. Builds candidates level-wise from frequent (k-1)-itemsets
3. Prunes by minimum support before counting
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Hashable

import pandas as pd


Item = Hashable
Itemset = frozenset[Item]


def _itemset_support(
    transactions: list[Itemset],
    candidates: set[Itemset],
) -> dict[Itemset, float]:
    """Count support for each candidate itemset."""
    if not candidates:
        return {}
    n = len(transactions)
    counts: Counter[Itemset] = Counter()
    # Index candidates by size for a cheap membership filter
    for basket in transactions:
        for cand in candidates:
            if cand.issubset(basket):
                counts[cand] += 1
    return {c: counts[c] / n for c in candidates}


def _join_candidates(prev_frequent: set[Itemset], k: int) -> set[Itemset]:
    """Generate k-itemset candidates from frequent (k-1)-itemsets."""
    prev = [sorted(s) for s in prev_frequent]
    candidates: set[Itemset] = set()
    for i, a in enumerate(prev):
        for b in prev[i + 1 :]:
            if a[: k - 2] == b[: k - 2]:
                merged = frozenset(a) | frozenset(b)
                if len(merged) == k:
                    # Apriori prune: all (k-1) subsets must be frequent
                    subsets = {frozenset(s) for s in combinations(merged, k - 1)}
                    if subsets.issubset(prev_frequent):
                        candidates.add(merged)
    return candidates


def apriori(
    transactions: list[Itemset],
    min_support: float = 0.01,
    max_len: int | None = None,
) -> dict[Itemset, float]:
    """Mine frequent itemsets with the Apriori algorithm.

    Parameters
    ----------
    transactions:
        List of frozenset baskets.
    min_support:
        Minimum support threshold in [0, 1].
    max_len:
        Optional maximum itemset size.

    Returns
    -------
    Mapping of frequent itemset -> support.
    """
    if not transactions:
        return {}
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be in (0, 1]")

    n = len(transactions)
    # L1
    item_counts: Counter[Item] = Counter()
    for basket in transactions:
        item_counts.update(basket)
    frequent: dict[Itemset, float] = {
        frozenset([item]): count / n
        for item, count in item_counts.items()
        if count / n >= min_support
    }
    prev_level = set(frequent.keys())
    k = 2

    while prev_level and (max_len is None or k <= max_len):
        candidates = _join_candidates(prev_level, k)
        if not candidates:
            break
        supports = _itemset_support(transactions, candidates)
        level = {c: s for c, s in supports.items() if s >= min_support}
        frequent.update(level)
        prev_level = set(level.keys())
        k += 1

    return frequent


def frequent_itemsets_to_frame(
    frequent: dict[Itemset, float],
    name_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Convert a frequent-itemset dict into a tidy DataFrame."""
    rows = []
    for itemset, support in frequent.items():
        items = sorted(itemset)
        names = (
            [name_map.get(i, str(i)) for i in items] if name_map else [str(i) for i in items]
        )
        rows.append(
            {
                "itemset": tuple(items),
                "itemset_names": tuple(names),
                "size": len(items),
                "support": support,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["itemset", "itemset_names", "size", "support"])
    return (
        pd.DataFrame(rows)
        .sort_values(["size", "support"], ascending=[True, False])
        .reset_index(drop=True)
    )
