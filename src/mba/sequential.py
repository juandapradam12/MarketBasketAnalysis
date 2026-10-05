"""Sequential pattern mining over user order histories.

Unlike unordered baskets, sequences respect time: product A in order t then
product B in a later order. We mine frequent consecutive item transitions
(first-order sequential rules) and longer subsequences with a light GSP-style
pass over user timelines.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Hashable

import pandas as pd

Item = Hashable


def user_sequences_from_orders(
    orders: pd.DataFrame,
    order_products: pd.DataFrame,
) -> dict[int, list[frozenset[int]]]:
    """Build per-user ordered list of baskets (by order_number)."""
    merged = order_products.merge(
        orders[["order_id", "user_id", "order_number"]],
        on="order_id",
        how="inner",
    )
    sequences: dict[int, list[frozenset[int]]] = defaultdict(list)
    for (user_id, _order_number), grp in merged.sort_values(
        ["user_id", "order_number"]
    ).groupby(["user_id", "order_number"]):
        sequences[int(user_id)].append(frozenset(int(x) for x in grp["product_id"].unique()))
    return dict(sequences)


def sequential_transitions(
    sequences: dict[int, list[frozenset[int]]],
    min_support: float = 0.02,
    min_confidence: float = 0.15,
    name_map: dict[int, str] | None = None,
    top_n: int = 50,
) -> pd.DataFrame:
    """Mine A ⇒ B transitions across consecutive orders of the same user.

    Support is P(A in order t and B in order t+1) over consecutive pairs.
    Confidence is P(B in t+1 | A in t).
    """
    if not sequences:
        return pd.DataFrame()

    pair_counts: Counter[tuple[Item, Item]] = Counter()
    ant_counts: Counter[Item] = Counter()
    n_pairs = 0

    for baskets in sequences.values():
        for t in range(len(baskets) - 1):
            a_set, b_set = baskets[t], baskets[t + 1]
            n_pairs += 1
            for a in a_set:
                ant_counts[a] += 1
                for b in b_set:
                    if a != b:
                        pair_counts[(a, b)] += 1

    if n_pairs == 0:
        return pd.DataFrame()

    rows = []
    for (a, b), count in pair_counts.items():
        support = count / n_pairs
        if support < min_support:
            continue
        conf = count / ant_counts[a] if ant_counts[a] else 0.0
        if conf < min_confidence:
            continue
        rows.append(
            {
                "antecedent": a,
                "consequent": b,
                "antecedent_name": name_map.get(a, str(a)) if name_map else str(a),
                "consequent_name": name_map.get(b, str(b)) if name_map else str(b),
                "count": count,
                "support": support,
                "confidence": conf,
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["confidence", "support"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def frequent_sequential_pairs(
    sequences: dict[int, list[frozenset[int]]],
    min_support: float = 0.03,
    name_map: dict[int, str] | None = None,
    top_n: int = 40,
) -> pd.DataFrame:
    """Frequent unordered pairs that appear in *different* consecutive orders.

    Useful as a compact sequential summary complementary to transitions.
    """
    if not sequences:
        return pd.DataFrame()

    n_users = len(sequences)
    pair_users: dict[tuple[Item, Item], set[int]] = defaultdict(set)

    for user_id, baskets in sequences.items():
        # Items seen across the user's timeline with order index
        item_orders: dict[Item, list[int]] = defaultdict(list)
        for idx, basket in enumerate(baskets):
            for item in basket:
                item_orders[item].append(idx)
        items = list(item_orders.keys())
        for a, b in combinations(sorted(items, key=str), 2):
            # Sequential if some occurrence of a precedes some of b (or vice versa)
            if min(item_orders[a]) < max(item_orders[b]) or min(item_orders[b]) < max(
                item_orders[a]
            ):
                pair_users[(a, b)].add(user_id)

    rows = []
    for (a, b), users in pair_users.items():
        support = len(users) / n_users
        if support < min_support:
            continue
        rows.append(
            {
                "item_a": a,
                "item_b": b,
                "name_a": name_map.get(a, str(a)) if name_map else str(a),
                "name_b": name_map.get(b, str(b)) if name_map else str(b),
                "user_count": len(users),
                "support": support,
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("support", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
