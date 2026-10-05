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


def load_orders(path: str | Path) -> pd.DataFrame:
    """Load orders table (order_id, user_id, order_number, ...)."""
    df = pd.read_csv(path)
    required = {"order_id", "user_id", "order_number"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def enrich_products(
    products: pd.DataFrame,
    aisles: pd.DataFrame | None = None,
    departments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join aisle / department names onto the product catalog."""
    out = products.copy()
    if aisles is not None and "aisle_id" in out.columns:
        out = out.merge(aisles, on="aisle_id", how="left")
    if departments is not None and "department_id" in out.columns:
        out = out.merge(departments, on="department_id", how="left")
    return out


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


def filter_reordered(
    order_products: pd.DataFrame,
    reordered_only: bool | None = None,
) -> pd.DataFrame:
    """Optionally keep only reordered or only first-time line items."""
    if reordered_only is None or "reordered" not in order_products.columns:
        return order_products
    flag = 1 if reordered_only else 0
    return order_products[order_products["reordered"] == flag].copy()


def filter_by_department(
    order_products: pd.DataFrame,
    products: pd.DataFrame,
    department_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Keep only products belonging to the given department ids."""
    if not department_ids:
        return order_products
    keep = set(int(x) for x in department_ids)
    valid = products[products["department_id"].isin(keep)]["product_id"]
    return order_products[order_products["product_id"].isin(valid)].copy()


def transactions_from_orders(order_products: pd.DataFrame) -> list[frozenset[int]]:
    """Convert order-product edges into a list of frozenset baskets."""
    grouped = order_products.groupby("order_id")["product_id"].agg(
        lambda s: frozenset(int(x) for x in s.unique())
    )
    return list(grouped.values)


def product_name_map(products: pd.DataFrame) -> dict[int, str]:
    """Map product_id -> product_name."""
    return dict(zip(products["product_id"].astype(int), products["product_name"]))


def product_margin_map(products: pd.DataFrame) -> dict[int, float]:
    """Map product_id -> unit_margin when available."""
    if "unit_margin" not in products.columns:
        return {}
    return dict(
        zip(
            products["product_id"].astype(int),
            products["unit_margin"].astype(float),
        )
    )


def lookup_names(
    item_ids: Iterable[int],
    name_map: dict[int, str],
) -> list[str]:
    """Resolve product ids to names; fall back to the id string."""
    return [name_map.get(int(i), str(i)) for i in item_ids]


def resolve_dataset_dir(root: Path, data_dir: Path | None = None) -> Path:
    """Prefer data/raw when complete, else data/sample."""
    candidates = []
    if data_dir is not None:
        candidates.append(data_dir)
    candidates.extend([root / "data" / "raw", root / "data" / "sample", root])
    for base in candidates:
        if (base / "order_products__train.csv").exists() and (base / "products.csv").exists():
            return base
    raise FileNotFoundError(
        "Could not find order_products__train.csv and products.csv. "
        "Run: python scripts/generate_sample_data.py"
    )
