#!/usr/bin/env python3
"""Generate an Instacart-like sample market-basket dataset.

The original notebook used the public Instacart MBA files, which are no longer
hosted at the old S3 URL and typically require a Kaggle account. This script
builds a compact, realistic substitute with planted associations that mirror
the notebook's findings (sparkling waters, berry mixes, produce staples) so the
enhanced pipeline runs end-to-end without external credentials.

Place real Instacart CSVs in ``data/raw/`` if you have them; ``run_analysis.py``
prefers those automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Catalog inspired by Instacart product ids / names from the original notebook
PRODUCTS = [
    (24852, "Banana", 24, 4),
    (13176, "Bag of Organic Bananas", 24, 4),
    (21137, "Organic Strawberries", 24, 4),
    (21903, "Organic Baby Spinach", 123, 4),
    (47626, "Large Lemon", 24, 4),
    (47766, "Organic Avocado", 24, 4),
    (47209, "Organic Hass Avocado", 24, 4),
    (16797, "Strawberries", 24, 4),
    (26209, "Limes", 24, 4),
    (27966, "Organic Raspberries", 123, 4),
    (22935, "Organic Yellow Onion", 83, 4),
    (24964, "Organic Garlic", 83, 4),
    (45007, "Organic Zucchini", 83, 4),
    (39275, "Organic Blueberries", 123, 4),
    (27845, "Organic Whole Milk", 84, 16),
    (49683, "Cucumber Kirby", 83, 4),
    (28204, "Organic Fuji Apple", 24, 4),
    (40706, "Organic Grape Tomatoes", 123, 4),
    (5876, "Organic Lemon", 24, 4),
    (49235, "Organic Half & Half", 53, 16),
    (14233, "Natural Spring Water", 115, 7),
    (12341, "Hass Avocados", 24, 4),
    # Sparkling-water cluster (top 4-set in the original notebook)
    (14947, "Pure Sparkling Water", 115, 7),
    (21709, "Sparkling Lemon Water", 115, 7),
    (35221, "Lime Sparkling Water", 115, 7),
    (44632, "Sparkling Water Grapefruit", 115, 7),
    (26620, "Peach Pear Flavored Sparkling Water", 115, 7),
    # Berry cluster (second notable 4-set)
    (21288, "Blackberries", 24, 4),
    (43352, "Raspberries", 24, 4),
    # Other grocery staples for noise / secondary patterns
    (196, "Soda", 77, 7),
    (6184, "Sparkling Water Berry", 115, 7),
    (37710, "Trail Mix", 125, 19),
    (43154, "Organic Red Potatoes", 83, 4),
    (30489, "Organic Raw Kombucha Gingerade", 31, 7),
    (27086, "Half & Half", 53, 16),
    (5077, "100% Whole Wheat Bread", 112, 3),
    (35951, "Organic Large Extra Fancy Fuji Apple", 24, 4),
    (17794, "Carrots", 83, 4),
    (4605, "Yellow Onions", 83, 4),
    (8518, "Organic Red Onion", 83, 4),
    (31717, "Organic Cilantro", 16, 4),
    (39928, "Organic Baby Carrots", 123, 4),
    (46979, "Asparagus", 83, 4),
    (34126, "Organic Italian Parsley Bunch", 16, 4),
    (28842, "Organic Sweet Peas", 116, 1),
    (22825, "Organic Broccoli", 83, 4),
    (24184, "Organic Baby Broccoli", 83, 4),
    (10749, "Organic Cilantro Bunch", 16, 4),
    (18523, "Organic Baby Arugula", 123, 4),
    (27104, "Fresh Basil", 16, 4),
]


# Planted theme baskets: (product_ids, relative weight)
THEMES = [
    ([14947, 21709, 35221, 44632], 0.045),  # sparkling water quartet
    ([14947, 21709, 35221, 26620], 0.020),
    ([16797, 21288, 39275, 43352], 0.030),  # berry quartet
    ([16797, 21137, 27966, 39275], 0.025),
    ([24852, 47766, 21903], 0.040),  # banana + avocado + spinach
    ([13176, 21137, 47209], 0.035),
    ([24852, 47626, 26209], 0.025),
    ([27845, 5077, 27086], 0.030),  # breakfast
    ([22935, 24964, 45007, 49683], 0.020),  # savory veg
    ([14233, 14947, 21709], 0.015),
]


def _sample_theme(rng: np.random.Generator) -> list[int]:
    weights = np.array([w for _, w in THEMES], dtype=float)
    weights /= weights.sum()
    idx = int(rng.choice(len(THEMES), p=weights))
    return list(THEMES[idx][0])


def _sample_noise(rng: np.random.Generator, k: int, exclude: set[int]) -> list[int]:
    pool = [pid for pid, *_ in PRODUCTS if pid not in exclude]
    # Mild popularity skew toward early catalog entries
    prefs = np.linspace(1.5, 0.4, num=len(pool))
    prefs /= prefs.sum()
    chosen = rng.choice(pool, size=min(k, len(pool)), replace=False, p=prefs)
    return list(map(int, chosen))


def generate(
    n_orders: int = 8000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    products = pd.DataFrame(
        PRODUCTS,
        columns=["product_id", "product_name", "aisle_id", "department_id"],
    )

    rows = []
    for order_id in range(1, n_orders + 1):
        # Mix of theme-heavy and random baskets
        if rng.random() < 0.55:
            core = _sample_theme(rng)
        else:
            core = _sample_noise(rng, k=int(rng.integers(2, 5)), exclude=set())

        extra_n = int(rng.integers(0, 5))
        extras = _sample_noise(rng, k=extra_n, exclude=set(core))
        basket = list(dict.fromkeys(core + extras))  # preserve order, unique
        if len(basket) < 2:
            basket = _sample_noise(rng, k=3, exclude=set())

        for pos, pid in enumerate(basket, start=1):
            rows.append(
                {
                    "order_id": order_id,
                    "product_id": pid,
                    "add_to_cart_order": pos,
                    "reordered": int(rng.random() < 0.55),
                }
            )

    order_products = pd.DataFrame(rows)
    return order_products, products


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-orders", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/sample"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    order_products, products = generate(n_orders=args.n_orders, seed=args.seed)
    op_path = args.out_dir / "order_products__train.csv"
    pr_path = args.out_dir / "products.csv"
    order_products.to_csv(op_path, index=False)
    products.to_csv(pr_path, index=False)
    print(f"Wrote {len(order_products):,} rows -> {op_path}")
    print(f"Wrote {len(products):,} products -> {pr_path}")
    print(f"Orders: {order_products['order_id'].nunique():,}")


if __name__ == "__main__":
    main()
