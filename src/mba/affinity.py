"""Pairwise affinity metrics as a complementary view to itemset mining.

Pointwise Mutual Information (PMI) and Jaccard highlight item pairs that
co-occur more than chance — useful when you care about strong *associations*
rather than absolute frequency.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import log2
from typing import Hashable

import numpy as np
import pandas as pd

Item = Hashable
Itemset = frozenset[Item]


def cooccurrence_matrix(
    transactions: list[Itemset],
    top_n_items: int = 30,
) -> pd.DataFrame:
    """Build a co-occurrence count matrix for the most frequent items."""
    item_counts: Counter[Item] = Counter()
    for basket in transactions:
        item_counts.update(basket)
    top_items = [item for item, _ in item_counts.most_common(top_n_items)]
    index = {item: i for i, item in enumerate(top_items)}
    matrix = np.zeros((len(top_items), len(top_items)), dtype=float)

    for basket in transactions:
        present = [index[i] for i in basket if i in index]
        for a, b in combinations(sorted(present), 2):
            matrix[a, b] += 1
            matrix[b, a] += 1
        for i in present:
            matrix[i, i] += 1

    return pd.DataFrame(matrix, index=top_items, columns=top_items)


def pairwise_pmi(
    transactions: list[Itemset],
    min_count: int = 5,
    top_n: int = 40,
    name_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Compute PMI and Jaccard for item pairs.

    PMI(x,y) = log2( P(x,y) / (P(x)P(y)) )
    """
    n = len(transactions)
    if n == 0:
        return pd.DataFrame()

    singles: Counter[Item] = Counter()
    pairs: Counter[tuple[Item, Item]] = Counter()
    for basket in transactions:
        items = sorted(basket)
        singles.update(items)
        pairs.update(combinations(items, 2))

    rows = []
    for (a, b), count in pairs.items():
        if count < min_count:
            continue
        p_a = singles[a] / n
        p_b = singles[b] / n
        p_ab = count / n
        if p_a == 0 or p_b == 0 or p_ab == 0:
            continue
        pmi = log2(p_ab / (p_a * p_b))
        jaccard = count / (singles[a] + singles[b] - count)
        name_a = name_map.get(a, str(a)) if name_map else str(a)
        name_b = name_map.get(b, str(b)) if name_map else str(b)
        rows.append(
            {
                "item_a": a,
                "item_b": b,
                "name_a": name_a,
                "name_b": name_b,
                "count": count,
                "support": p_ab,
                "pmi": pmi,
                "jaccard": jaccard,
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["pmi", "support"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
