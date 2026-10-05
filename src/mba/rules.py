"""Association-rule generation from frequent itemsets."""

from __future__ import annotations

from itertools import combinations
from typing import Hashable

import pandas as pd

Item = Hashable
Itemset = frozenset[Item]


def association_rules(
    frequent: dict[Itemset, float],
    min_confidence: float = 0.2,
    min_lift: float = 1.0,
    name_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Derive association rules with confidence, lift, leverage, conviction.

    For a rule A -> B:
      confidence = support(A∪B) / support(A)
      lift       = confidence / support(B)
      leverage   = support(A∪B) - support(A)*support(B)
      conviction = (1 - support(B)) / (1 - confidence)
    """
    rows = []
    for itemset, support_ab in frequent.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for r in range(1, len(items)):
            for antecedent_t in combinations(items, r):
                antecedent = frozenset(antecedent_t)
                consequent = itemset - antecedent
                support_a = frequent.get(antecedent)
                support_b = frequent.get(consequent)
                if support_a is None or support_b is None or support_a == 0:
                    continue
                confidence = support_ab / support_a
                if confidence < min_confidence:
                    continue
                lift = confidence / support_b if support_b else float("inf")
                if lift < min_lift:
                    continue
                leverage = support_ab - support_a * support_b
                if confidence < 1:
                    conviction = (1 - support_b) / (1 - confidence)
                else:
                    conviction = float("inf")

                ant_ids = tuple(sorted(antecedent))
                cons_ids = tuple(sorted(consequent))
                if name_map:
                    ant_names = tuple(name_map.get(i, str(i)) for i in ant_ids)
                    cons_names = tuple(name_map.get(i, str(i)) for i in cons_ids)
                else:
                    ant_names = tuple(str(i) for i in ant_ids)
                    cons_names = tuple(str(i) for i in cons_ids)

                rows.append(
                    {
                        "antecedents": ant_ids,
                        "consequents": cons_ids,
                        "antecedent_names": ant_names,
                        "consequent_names": cons_names,
                        "support": support_ab,
                        "confidence": confidence,
                        "lift": lift,
                        "leverage": leverage,
                        "conviction": conviction,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "antecedents",
                "consequents",
                "antecedent_names",
                "consequent_names",
                "support",
                "confidence",
                "lift",
                "leverage",
                "conviction",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["lift", "confidence", "support"], ascending=False)
        .reset_index(drop=True)
    )
