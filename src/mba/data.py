"""Transaction loading and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_order_products(path: str | Path) -> pd.DataFrame:
    """Load order-product edge list with columns order_id, product_id."""
    df = pd.read_csv(path)
    required = {"order_id", "product_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def load_products(path: str | Path) -> pd.DataFrame:
    """Load product catalog with product_id and product_name."""
    df = pd.read_csv(path)
    required = {"product_id", "product_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def filter_frequent_products(
    order_products: pd.DataFrame,
    min_count: int = 50,
) -> tuple[pd.DataFrame, pd.Index]:
    """Keep products that appear at least ``min_count`` times."""
    counts = order_products["product_id"].value_counts()
    keep = counts[counts >= min_count].index
    filtered = order_products[order_products["product_id"].isin(keep)].copy()
    return filtered, keep


def filter_orders_by_size(
    order_products: pd.DataFrame,
    min_size: int = 2,
) -> pd.DataFrame:
    """Keep orders with at least ``min_size`` products (Apriori-compatible)."""
    sizes = order_products.groupby("order_id")["product_id"].nunique()
    keep_orders = sizes[sizes >= min_size].index
    return order_products[order_products["order_id"].isin(keep_orders)].copy()


def transactions_from_orders(order_products: pd.DataFrame) -> list[frozenset[int]]:
    """Convert order-product edges into a list of frozenset baskets."""
    grouped = order_products.groupby("order_id")["product_id"].agg(
        lambda s: frozenset(int(x) for x in s.unique())
    )
    return list(grouped.values)


def product_name_map(products: pd.DataFrame) -> dict[int, str]:
    """Map product_id -> product_name."""
    return dict(zip(products["product_id"].astype(int), products["product_name"]))


def lookup_names(
    item_ids: Iterable[int],
    name_map: dict[int, str],
) -> list[str]:
    """Resolve product ids to names; fall back to the id string."""
    return [name_map.get(int(i), str(i)) for i in item_ids]
