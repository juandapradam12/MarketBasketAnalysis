"""Runtime benchmarks across frequent-itemset miners."""

from __future__ import annotations

import time
from typing import Callable

import pandas as pd

from .apriori import apriori
from .eclat import eclat
from .fp_growth import fp_growth


def benchmark_miners(
    transactions: list,
    min_support: float = 0.01,
    max_len: int | None = 4,
    miners: dict[str, Callable] | None = None,
) -> pd.DataFrame:
    """Time Apriori / FP-Growth / Eclat and check itemset-set agreement."""
    if miners is None:
        miners = {
            "apriori": apriori,
            "fp_growth": fp_growth,
            "eclat": eclat,
        }

    rows = []
    results: dict[str, set] = {}
    for name, fn in miners.items():
        t0 = time.perf_counter()
        freq = fn(transactions, min_support=min_support, max_len=max_len)
        elapsed = time.perf_counter() - t0
        results[name] = set(freq.keys())
        rows.append(
            {
                "method": name,
                "seconds": elapsed,
                "itemsets": len(freq),
                "min_support": min_support,
                "max_len": max_len,
            }
        )

    # Agreement vs Apriori reference when present
    ref = results.get("apriori")
    frame = pd.DataFrame(rows)
    if ref is not None:
        frame["agrees_with_apriori"] = [
            results[m] == ref for m in frame["method"]
        ]
    return frame.sort_values("seconds").reset_index(drop=True)
