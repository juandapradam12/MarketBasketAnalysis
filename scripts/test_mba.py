"""Lightweight unit checks for the MBA toolkit."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mba.apriori import apriori
from mba.fp_growth import fp_growth
from mba.rules import association_rules
from mba.affinity import pairwise_pmi


class TestMBA(unittest.TestCase):
    def setUp(self) -> None:
        # Classic toy dataset
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
        self.assertGreaterEqual(freq[frozenset(["milk"])], 0.5)

    def test_apriori_fp_growth_agree(self) -> None:
        a = apriori(self.transactions, min_support=0.25, max_len=3)
        f = fp_growth(self.transactions, min_support=0.25, max_len=3)
        self.assertEqual(set(a.keys()), set(f.keys()))
        for k in a:
            self.assertAlmostEqual(a[k], f[k], places=6)

    def test_rules_have_lift(self) -> None:
        freq = apriori(self.transactions, min_support=0.25, max_len=3)
        rules = association_rules(freq, min_confidence=0.5, min_lift=1.0)
        self.assertFalse(rules.empty)
        self.assertTrue((rules["lift"] >= 1.0).all())

    def test_pmi_positive_for_associated_pair(self) -> None:
        pmi = pairwise_pmi(self.transactions, min_count=2, top_n=20)
        pair = pmi[
            ((pmi["item_a"] == "milk") & (pmi["item_b"] == "bread"))
            | ((pmi["item_a"] == "bread") & (pmi["item_b"] == "milk"))
        ]
        self.assertFalse(pair.empty)
        self.assertGreater(float(pair.iloc[0]["pmi"]), 0)


if __name__ == "__main__":
    unittest.main()
