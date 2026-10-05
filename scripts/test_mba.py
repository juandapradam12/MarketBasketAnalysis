"""Unit checks for the expanded MBA toolkit."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mba.affinity import pairwise_pmi
from mba.apriori import apriori
from mba.benchmark import benchmark_miners
from mba.eclat import eclat
from mba.evaluate import pair_hit_rate
from mba.fp_growth import fp_growth
from mba.recommend import item_item_recommendations, svd_recommendations
from mba.rules import annotate_rule_departments, association_rules
from mba.sequential import sequential_transitions, user_sequences_from_orders
import pandas as pd


class TestMBA(unittest.TestCase):
    def setUp(self) -> None:
        self.transactions = [
            frozenset(["milk", "bread", "butter"]),
            frozenset(["milk", "bread"]),
            frozenset(["milk", "diapers"]),
            frozenset(["bread", "butter"]),
            frozenset(["milk", "bread", "butter", "diapers"]),
            frozenset(["bread", "diapers"]),
            frozenset(["milk", "bread", "diapers"]),
            frozenset(["butter"]),
        ]

    def test_apriori_finds_milk_bread(self) -> None:
        freq = apriori(self.transactions, min_support=0.3, max_len=3)
        self.assertIn(frozenset(["milk", "bread"]), freq)

    def test_three_miners_agree(self) -> None:
        a = apriori(self.transactions, min_support=0.25, max_len=3)
        f = fp_growth(self.transactions, min_support=0.25, max_len=3)
        e = eclat(self.transactions, min_support=0.25, max_len=3)
        self.assertEqual(set(a.keys()), set(f.keys()))
        self.assertEqual(set(a.keys()), set(e.keys()))
        for k in a:
            self.assertAlmostEqual(a[k], f[k], places=6)
            self.assertAlmostEqual(a[k], e[k], places=6)

    def test_benchmark_frame(self) -> None:
        frame = benchmark_miners(self.transactions, min_support=0.25, max_len=3)
        self.assertEqual(set(frame["method"]), {"apriori", "fp_growth", "eclat"})
        self.assertTrue(frame["agrees_with_apriori"].all())

    def test_rules_zhang_and_utility(self) -> None:
        freq = apriori(self.transactions, min_support=0.25, max_len=3)
        margins = {"milk": 1.0, "bread": 0.5, "butter": 0.8, "diapers": 2.0}
        rules = association_rules(
            freq, min_confidence=0.5, min_lift=1.0, margin_map=margins
        )
        self.assertFalse(rules.empty)
        self.assertIn("zhang", rules.columns)
        self.assertIn("utility", rules.columns)
        self.assertTrue((rules["lift"] >= 1.0).all())

    def test_scope_annotation(self) -> None:
        products = pd.DataFrame(
            {
                "product_id": [1, 2, 3],
                "product_name": ["a", "b", "c"],
                "aisle_id": [10, 10, 20],
                "department_id": [1, 1, 2],
                "aisle": ["x", "x", "y"],
                "department": ["d1", "d1", "d2"],
            }
        )
        rules = pd.DataFrame(
            {
                "antecedents": [(1,), (1,)],
                "consequents": [(2,), (3,)],
                "antecedent_names": [("a",), ("a",)],
                "consequent_names": [("b",), ("c",)],
                "support": [0.1, 0.1],
                "confidence": [0.5, 0.5],
                "lift": [2.0, 2.0],
            }
        )
        out = annotate_rule_departments(rules, products)
        self.assertEqual(out.loc[0, "scope"], "intra_aisle")
        self.assertEqual(out.loc[1, "scope"], "cross_department")

    def test_pmi_positive(self) -> None:
        pmi = pairwise_pmi(self.transactions, min_count=2, top_n=20)
        pair = pmi[
            ((pmi["item_a"] == "milk") & (pmi["item_b"] == "bread"))
            | ((pmi["item_a"] == "bread") & (pmi["item_b"] == "milk"))
        ]
        self.assertFalse(pair.empty)
        self.assertGreater(float(pair.iloc[0]["pmi"]), 0)

    def test_recommenders(self) -> None:
        cf = item_item_recommendations(self.transactions, "milk", top_n=3)
        svd = svd_recommendations(self.transactions, ["milk"], top_n=3)
        self.assertFalse(cf.empty)
        self.assertFalse(svd.empty)

    def test_sequential_and_holdout(self) -> None:
        orders = pd.DataFrame(
            {
                "order_id": [1, 2, 3, 4],
                "user_id": [1, 1, 2, 2],
                "order_number": [1, 2, 1, 2],
            }
        )
        op = pd.DataFrame(
            {
                "order_id": [1, 1, 2, 2, 3, 4, 4],
                "product_id": [10, 11, 10, 12, 10, 10, 12],
            }
        )
        seqs = user_sequences_from_orders(orders, op)
        trans = sequential_transitions(seqs, min_support=0.1, min_confidence=0.1)
        self.assertFalse(trans.empty)

        freq = apriori(self.transactions, min_support=0.25, max_len=3)
        rules = association_rules(freq, min_confidence=0.5, min_lift=1.0)
        metrics = pair_hit_rate(rules, self.transactions, top_k=5)
        self.assertGreaterEqual(metrics["pair_precision"], 0.0)


if __name__ == "__main__":
    unittest.main()
