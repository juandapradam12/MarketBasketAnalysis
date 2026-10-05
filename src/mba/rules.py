"""Association-rule generation from frequent itemsets."""

from __future__ import annotations

from itertools import combinations
from typing import Hashable

import pandas as pd

Item = Hashable
Itemset = frozenset[Item]


def _zhang_metric(support_ab: float, support_a: float, support_b: float) -> float:
    """Zhang's interestingness metric in [-1, 1].

    Positive values indicate positive dependence; negative indicate substitutes.
    """
    numerator = support_ab - support_a * support_b
    denom = max(support_ab * (1 - support_a), support_a * (support_b - support_ab))
    if denom == 0:
        return 0.0
    return numerator / denom


def association_rules(
    frequent: dict[Itemset, float],
    min_confidence: float = 0.2,
    min_lift: float = 1.0,
    name_map: dict[int, str] | None = None,
    margin_map: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Derive association rules with confidence, lift, leverage, conviction, Zhang.

    Optional ``margin_map`` adds ``utility = lift * sum(margins in rule)`` so
    rules can be ranked by approximate economic value.
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
                zhang = _zhang_metric(support_ab, support_a, support_b)

                ant_ids = tuple(sorted(antecedent))
                cons_ids = tuple(sorted(consequent))
                if name_map:
                    ant_names = tuple(name_map.get(i, str(i)) for i in ant_ids)
                    cons_names = tuple(name_map.get(i, str(i)) for i in cons_ids)
                else:
                    ant_names = tuple(str(i) for i in ant_ids)
                    cons_names = tuple(str(i) for i in cons_ids)

                margin_sum = 0.0
                if margin_map:
                    margin_sum = sum(margin_map.get(i, 0.0) for i in itemset)
                utility = lift * margin_sum if margin_map else None

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
                        "zhang": zhang,
                        "margin_sum": margin_sum if margin_map else None,
                        "utility": utility,
                    }
                )

    columns = [
        "antecedents",
        "consequents",
        "antecedent_names",
        "consequent_names",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
        "zhang",
        "margin_sum",
        "utility",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    sort_cols = ["lift", "confidence", "support"]
    if margin_map:
        sort_cols = ["utility", "lift", "confidence"]

    return (
        pd.DataFrame(rows)
        .sort_values(sort_cols, ascending=False)
        .reset_index(drop=True)
    )


def annotate_rule_departments(
    rules: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Add department/aisle scope labels: intra vs cross category."""
    if rules.empty:
        return rules.copy()

    dept = dict(zip(products["product_id"].astype(int), products["department_id"].astype(int)))
    aisle = dict(zip(products["product_id"].astype(int), products["aisle_id"].astype(int)))
    dept_name = {}
    aisle_name = {}
    if "department" in products.columns:
        dept_name = dict(zip(products["product_id"].astype(int), products["department"]))
    if "aisle" in products.columns:
        aisle_name = dict(zip(products["product_id"].astype(int), products["aisle"]))

    out = rules.copy()
    scopes = []
    dept_pairs = []
    aisle_pairs = []
    for _, row in out.iterrows():
        ids = list(row["antecedents"]) + list(row["consequents"])
        depts = {dept.get(i) for i in ids}
        aisles = {aisle.get(i) for i in ids}
        depts.discard(None)
        aisles.discard(None)
        if len(depts) <= 1 and len(aisles) <= 1:
            scope = "intra_aisle"
        elif len(depts) <= 1:
            scope = "intra_department"
        else:
            scope = "cross_department"
        scopes.append(scope)
        dept_pairs.append(tuple(sorted({dept_name.get(i, dept.get(i)) for i in ids})))
        aisle_pairs.append(tuple(sorted({aisle_name.get(i, aisle.get(i)) for i in ids})))

    out["scope"] = scopes
    out["departments"] = dept_pairs
    out["aisles"] = aisle_pairs
    return out


def filter_rules_by_scope(
    rules: pd.DataFrame,
    scope: str | None = None,
) -> pd.DataFrame:
    """Filter annotated rules by scope label."""
    if scope is None or rules.empty or "scope" not in rules.columns:
        return rules
    return rules[rules["scope"] == scope].reset_index(drop=True)
