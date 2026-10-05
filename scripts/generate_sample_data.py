#!/usr/bin/env python3
"""Generate an Instacart-like sample market-basket dataset.

Produces:
  - products.csv, aisles.csv, departments.csv
  - orders.csv (user sequences + eval_set prior/train)
  - order_products__prior.csv, order_products__train.csv

Planted associations mirror the original notebook findings (sparkling waters,
berry mixes). Also plants sequential habits (e.g. bananas then milk) and
assigns synthetic margins for utility-aware rule ranking.

Place real Instacart CSVs in ``data/raw/`` if you have them; analysis scripts
prefer those automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

AISLES = {
    16: "fresh herbs",
    24: "fresh fruits",
    31: "refrigerated",
    53: "cream",
    77: "soft drinks",
    83: "fresh vegetables",
    84: "milk",
    112: "bread",
    115: "water seltzer sparkling water",
    116: "frozen produce",
    123: "packaged vegetables fruits",
    125: "trail mix snack mix",
}

DEPARTMENTS = {
    1: "frozen",
    3: "bakery",
    4: "produce",
    7: "beverages",
    16: "dairy eggs",
    19: "snacks",
}

# (product_id, name, aisle_id, department_id, unit_margin)
PRODUCTS = [
    (24852, "Banana", 24, 4, 0.25),
    (13176, "Bag of Organic Bananas", 24, 4, 0.40),
    (21137, "Organic Strawberries", 24, 4, 1.10),
    (21903, "Organic Baby Spinach", 123, 4, 0.90),
    (47626, "Large Lemon", 24, 4, 0.35),
    (47766, "Organic Avocado", 24, 4, 0.80),
    (47209, "Organic Hass Avocado", 24, 4, 0.85),
    (16797, "Strawberries", 24, 4, 0.95),
    (26209, "Limes", 24, 4, 0.30),
    (27966, "Organic Raspberries", 123, 4, 1.40),
    (22935, "Organic Yellow Onion", 83, 4, 0.45),
    (24964, "Organic Garlic", 83, 4, 0.50),
    (45007, "Organic Zucchini", 83, 4, 0.70),
    (39275, "Organic Blueberries", 123, 4, 1.30),
    (27845, "Organic Whole Milk", 84, 16, 1.20),
    (49683, "Cucumber Kirby", 83, 4, 0.55),
    (28204, "Organic Fuji Apple", 24, 4, 0.75),
    (40706, "Organic Grape Tomatoes", 123, 4, 0.85),
    (5876, "Organic Lemon", 24, 4, 0.40),
    (49235, "Organic Half & Half", 53, 16, 1.00),
    (14233, "Natural Spring Water", 115, 7, 0.60),
    (12341, "Hass Avocados", 24, 4, 0.70),
    (14947, "Pure Sparkling Water", 115, 7, 0.90),
    (21709, "Sparkling Lemon Water", 115, 7, 0.95),
    (35221, "Lime Sparkling Water", 115, 7, 0.95),
    (44632, "Sparkling Water Grapefruit", 115, 7, 0.95),
    (26620, "Peach Pear Flavored Sparkling Water", 115, 7, 1.00),
    (21288, "Blackberries", 24, 4, 1.20),
    (43352, "Raspberries", 24, 4, 1.25),
    (196, "Soda", 77, 7, 0.80),
    (6184, "Sparkling Water Berry", 115, 7, 0.95),
    (37710, "Trail Mix", 125, 19, 1.50),
    (43154, "Organic Red Potatoes", 83, 4, 0.65),
    (30489, "Organic Raw Kombucha Gingerade", 31, 7, 1.80),
    (27086, "Half & Half", 53, 16, 0.90),
    (5077, "100% Whole Wheat Bread", 112, 3, 1.10),
    (35951, "Organic Large Extra Fancy Fuji Apple", 24, 4, 0.80),
    (17794, "Carrots", 83, 4, 0.50),
    (4605, "Yellow Onions", 83, 4, 0.35),
    (8518, "Organic Red Onion", 83, 4, 0.45),
    (31717, "Organic Cilantro", 16, 4, 0.55),
    (39928, "Organic Baby Carrots", 123, 4, 0.70),
    (46979, "Asparagus", 83, 4, 1.00),
    (34126, "Organic Italian Parsley Bunch", 16, 4, 0.55),
    (28842, "Organic Sweet Peas", 116, 1, 0.85),
    (22825, "Organic Broccoli", 83, 4, 0.75),
    (24184, "Organic Baby Broccoli", 83, 4, 0.80),
    (10749, "Organic Cilantro Bunch", 16, 4, 0.55),
    (18523, "Organic Baby Arugula", 123, 4, 0.90),
    (27104, "Fresh Basil", 16, 4, 0.60),
]

THEMES = [
    ([14947, 21709, 35221, 44632], 0.045),
    ([14947, 21709, 35221, 26620], 0.020),
    ([16797, 21288, 39275, 43352], 0.030),
    ([16797, 21137, 27966, 39275], 0.025),
    ([24852, 47766, 21903], 0.040),
    ([13176, 21137, 47209], 0.035),
    ([24852, 47626, 26209], 0.025),
    ([27845, 5077, 27086], 0.030),
    ([22935, 24964, 45007, 49683], 0.020),
    ([14233, 14947, 21709], 0.015),
]

# Sequential habits: if user bought antecedent recently, boost consequent next order
SEQ_HABITS = [
    (24852, 27845, 0.45),  # banana → milk
    (13176, 21137, 0.40),  # organic bananas → organic strawberries
    (14947, 21709, 0.50),  # pure sparkling → lemon sparkling
    (16797, 21288, 0.35),  # strawberries → blackberries
    (5077, 27086, 0.40),   # bread → half & half
]


def _sample_theme(rng: np.random.Generator) -> list[int]:
    weights = np.array([w for _, w in THEMES], dtype=float)
    weights /= weights.sum()
    idx = int(rng.choice(len(THEMES), p=weights))
    return list(THEMES[idx][0])


def _sample_noise(rng: np.random.Generator, k: int, exclude: set[int]) -> list[int]:
    pool = [pid for pid, *_ in PRODUCTS if pid not in exclude]
    prefs = np.linspace(1.5, 0.4, num=len(pool))
    prefs /= prefs.sum()
    chosen = rng.choice(pool, size=min(k, len(pool)), replace=False, p=prefs)
    return list(map(int, chosen))


def _make_basket(rng: np.random.Generator, boost: list[int] | None = None) -> list[int]:
    if rng.random() < 0.55:
        core = _sample_theme(rng)
    else:
        core = _sample_noise(rng, k=int(rng.integers(2, 5)), exclude=set())
    if boost:
        core = list(dict.fromkeys(boost + core))
    extras = _sample_noise(rng, k=int(rng.integers(0, 5)), exclude=set(core))
    basket = list(dict.fromkeys(core + extras))
    if len(basket) < 2:
        basket = _sample_noise(rng, k=3, exclude=set())
    return basket


def generate(
    n_users: int = 1200,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    products = pd.DataFrame(
        PRODUCTS,
        columns=["product_id", "product_name", "aisle_id", "department_id", "unit_margin"],
    )
    aisles = pd.DataFrame(
        [{"aisle_id": k, "aisle": v} for k, v in sorted(AISLES.items())]
    )
    departments = pd.DataFrame(
        [{"department_id": k, "department": v} for k, v in sorted(DEPARTMENTS.items())]
    )

    order_rows: list[dict] = []
    op_rows: list[dict] = []
    order_id = 1

    for user_id in range(1, n_users + 1):
        n_orders = int(rng.integers(4, 11))
        prev_items: set[int] = set()
        days_since = None
        for order_number in range(1, n_orders + 1):
            eval_set = "train" if order_number == n_orders else "prior"
            # Sequential boost from previous basket
            boost: list[int] = []
            for ant, cons, p in SEQ_HABITS:
                if ant in prev_items and rng.random() < p:
                    boost.append(cons)
            basket = _make_basket(rng, boost=boost or None)
            prev_items = set(basket)

            if order_number == 1:
                days_since = np.nan
            else:
                days_since = float(rng.integers(1, 30))

            order_rows.append(
                {
                    "order_id": order_id,
                    "user_id": user_id,
                    "eval_set": eval_set,
                    "order_number": order_number,
                    "order_dow": int(rng.integers(0, 7)),
                    "order_hour_of_day": int(rng.integers(7, 22)),
                    "days_since_prior_order": days_since,
                }
            )
            for pos, pid in enumerate(basket, start=1):
                # Higher reorder chance if item was in previous basket for this user
                reordered = int(pid in prev_items and order_number > 1 and rng.random() < 0.7)
                # fix: prev_items already overwritten — compute reorder from history differently
                op_rows.append(
                    {
                        "order_id": order_id,
                        "product_id": pid,
                        "add_to_cart_order": pos,
                        "reordered": reordered,
                        "eval_set": eval_set,
                    }
                )
            order_id += 1

    orders = pd.DataFrame(order_rows)
    order_products = pd.DataFrame(op_rows)

    # Recompute reordered properly using user history
    order_products = _recompute_reordered(orders, order_products)

    prior = order_products[order_products["eval_set"] == "prior"].drop(columns=["eval_set"])
    train = order_products[order_products["eval_set"] == "train"].drop(columns=["eval_set"])

    return {
        "products": products,
        "aisles": aisles,
        "departments": departments,
        "orders": orders,
        "order_products__prior": prior.reset_index(drop=True),
        "order_products__train": train.reset_index(drop=True),
    }


def _recompute_reordered(orders: pd.DataFrame, op: pd.DataFrame) -> pd.DataFrame:
    """Mark reordered=1 if the user bought the product in any earlier order."""
    merged = op.merge(
        orders[["order_id", "user_id", "order_number"]],
        on="order_id",
        how="left",
    )
    merged = merged.sort_values(["user_id", "product_id", "order_number"])
    seen_before = merged.groupby(["user_id", "product_id"]).cumcount() > 0
    merged["reordered"] = seen_before.astype(int)
    return merged.drop(columns=["user_id", "order_number"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-users", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("data/sample"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables = generate(n_users=args.n_users, seed=args.seed)
    for name, df in tables.items():
        path = args.out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"Wrote {len(df):,} rows -> {path}")

    n_orders = tables["orders"]["order_id"].nunique()
    n_prior = tables["order_products__prior"]["order_id"].nunique()
    n_train = tables["order_products__train"]["order_id"].nunique()
    print(f"Users: {args.n_users:,} | Orders: {n_orders:,} (prior={n_prior:,}, train={n_train:,})")


if __name__ == "__main__":
    main()
