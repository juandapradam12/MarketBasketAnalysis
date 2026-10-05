"""Item-item collaborative filtering and SVD recommenders from baskets."""

from __future__ import annotations

from collections import defaultdict
from typing import Hashable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

Item = Hashable
Itemset = frozenset[Item]


def build_item_user_matrix(
    transactions: list[Itemset],
) -> tuple[sparse.csr_matrix, list[Item]]:
    """Binary basket × item matrix (rows = baskets, cols = items)."""
    item_to_col: dict[Item, int] = {}
    rows, cols, data = [], [], []
    for r, basket in enumerate(transactions):
        for item in basket:
            if item not in item_to_col:
                item_to_col[item] = len(item_to_col)
            rows.append(r)
            cols.append(item_to_col[item])
            data.append(1.0)
    matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(transactions), len(item_to_col)),
    )
    items = [None] * len(item_to_col)
    for item, col in item_to_col.items():
        items[col] = item
    return matrix, list(items)


def item_item_recommendations(
    transactions: list[Itemset],
    product_id: int | Item,
    top_n: int = 10,
    name_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Cosine item-item CF: recommend items similar to ``product_id``."""
    matrix, items = build_item_user_matrix(transactions)
    if product_id not in items:
        return pd.DataFrame(columns=["product_id", "product_name", "score"])

    # Item similarity from item vectors (columns)
    item_vectors = normalize(matrix.T.tocsr(), norm="l2", axis=1)
    idx = items.index(product_id)
    sims = cosine_similarity(item_vectors[idx], item_vectors).ravel()
    order = np.argsort(-sims)
    rows = []
    for j in order:
        if items[j] == product_id:
            continue
        pid = items[j]
        rows.append(
            {
                "product_id": pid,
                "product_name": name_map.get(pid, str(pid)) if name_map else str(pid),
                "score": float(sims[j]),
            }
        )
        if len(rows) >= top_n:
            break
    return pd.DataFrame(rows)


def svd_recommendations(
    transactions: list[Itemset],
    seed_items: list[Item],
    top_n: int = 10,
    n_components: int = 16,
    name_map: dict[int, str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Recommend via TruncatedSVD latent factors given a seed basket."""
    matrix, items = build_item_user_matrix(transactions)
    if matrix.shape[1] < 2:
        return pd.DataFrame(columns=["product_id", "product_name", "score"])

    n_comp = max(1, min(n_components, matrix.shape[1] - 1, matrix.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    # Factorize items in latent space via (items × factors) from X ≈ U S Vt
    # Use item embeddings from V
    svd.fit(matrix)
    item_factors = svd.components_.T  # (n_items, n_comp)

    seed_idx = [items.index(i) for i in seed_items if i in items]
    if not seed_idx:
        return pd.DataFrame(columns=["product_id", "product_name", "score"])

    query = item_factors[seed_idx].mean(axis=0)
    scores = item_factors @ query
    seed_set = set(seed_items)
    order = np.argsort(-scores)
    rows = []
    for j in order:
        pid = items[j]
        if pid in seed_set:
            continue
        rows.append(
            {
                "product_id": pid,
                "product_name": name_map.get(pid, str(pid)) if name_map else str(pid),
                "score": float(scores[j]),
            }
        )
        if len(rows) >= top_n:
            break
    return pd.DataFrame(rows)


def basket_cooccurrence_scores(
    transactions: list[Itemset],
    product_id: Item,
    top_n: int = 10,
    name_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Simple conditional P(other | product) ranking (baseline recommender)."""
    joint: dict[Item, int] = defaultdict(int)
    alone = 0
    for basket in transactions:
        if product_id not in basket:
            continue
        alone += 1
        for other in basket:
            if other != product_id:
                joint[other] += 1
    if alone == 0:
        return pd.DataFrame(columns=["product_id", "product_name", "score"])

    rows = [
        {
            "product_id": oid,
            "product_name": name_map.get(oid, str(oid)) if name_map else str(oid),
            "score": count / alone,
        }
        for oid, count in joint.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
