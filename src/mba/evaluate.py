"""Holdout evaluation for association rules and recommenders."""

from __future__ import annotations

from typing import Hashable

import pandas as pd

Item = Hashable
Itemset = frozenset[Item]


def pair_hit_rate(
    rules: pd.DataFrame,
    holdout_transactions: list[Itemset],
    top_k: int = 20,
) -> dict[str, float]:
    """Evaluate top-k rules as unordered pairs against holdout baskets.

    A rule hits if both antecedent and consequent items appear together in a
    holdout basket (for single-item antecedents/consequents) or if the full
    union is a subset (multi-item).
    """
    if rules.empty or not holdout_transactions:
        return {"top_k": float(top_k), "pair_precision": 0.0, "coverage": 0.0, "hits": 0.0}

    top = rules.head(top_k)
    hits = 0
    for _, row in top.iterrows():
        itemset = frozenset(row["antecedents"]) | frozenset(row["consequents"])
        if any(itemset.issubset(basket) for basket in holdout_transactions):
            hits += 1

    # Coverage: fraction of holdout baskets that contain at least one top rule
    covered = 0
    rule_sets = [
        frozenset(r["antecedents"]) | frozenset(r["consequents"])
        for _, r in top.iterrows()
    ]
    for basket in holdout_transactions:
        if any(s.issubset(basket) for s in rule_sets):
            covered += 1

    return {
        "top_k": float(top_k),
        "pair_precision": hits / len(top),
        "coverage": covered / len(holdout_transactions),
        "hits": float(hits),
    }


def recommend_recall_at_k(
    transactions_train: list[Itemset],
    transactions_holdout: list[Itemset],
    recommend_fn,
    k: int = 5,
    n_queries: int = 100,
) -> dict[str, float]:
    """Leave-one-out style recall@k using holdout baskets as ground truth.

    For each holdout basket with size >= 2, take one seed item, recommend k
    items, and check overlap with the remaining basket items.
    """
    if not transactions_holdout:
        return {"recall_at_k": 0.0, "n_queries": 0.0}

    recalls = []
    for basket in transactions_holdout[:n_queries]:
        items = list(basket)
        if len(items) < 2:
            continue
        seed = items[0]
        truth = set(items[1:])
        recs = recommend_fn(transactions_train, seed, top_n=k)
        if recs is None or len(recs) == 0:
            recalls.append(0.0)
            continue
        pred = set(recs["product_id"].tolist())
        recalls.append(len(pred & truth) / len(truth))

    if not recalls:
        return {"recall_at_k": 0.0, "n_queries": 0.0}
    return {"recall_at_k": float(sum(recalls) / len(recalls)), "n_queries": float(len(recalls))}
