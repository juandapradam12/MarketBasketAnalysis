"""FP-Growth frequent-itemset mining via mlxtend.

Complementary to Apriori: builds a compressed FP-tree and mines patterns
without candidate generation, which is typically faster on dense baskets.
"""

from __future__ import annotations

from typing import Hashable

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

Item = Hashable
Itemset = frozenset[Item]


def fp_growth(
    transactions: list[Itemset],
    min_support: float = 0.01,
    max_len: int | None = None,
) -> dict[Itemset, float]:
    """Mine frequent itemsets with FP-Growth.

    Returns the same dict[frozenset, support] shape as :func:`mba.apriori.apriori`
    so results are easy to compare.
    """
    if not transactions:
        return {}

    baskets = [list(t) for t in transactions]
    encoder = TransactionEncoder()
    encoded = encoder.fit(baskets).transform(baskets)
    df = pd.DataFrame(encoded, columns=encoder.columns_)

    patterns = fpgrowth(
        df,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )
    return {frozenset(row["itemsets"]): float(row["support"]) for _, row in patterns.iterrows()}
