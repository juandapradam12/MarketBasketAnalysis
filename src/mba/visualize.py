"""Visualization helpers for market-basket analysis results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

# Consistent visual language for the project figures
PALETTE = {
    "bg": "#f7f4ef",
    "ink": "#1c2a2e",
    "accent": "#0f6e56",
    "accent2": "#c45c26",
    "muted": "#6b7c80",
    "grid": "#d9d2c5",
}


def _style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor(PALETTE["bg"])
    ax.set_title(title, color=PALETTE["ink"], fontsize=13, pad=12, fontweight="bold")
    ax.tick_params(colors=PALETTE["ink"])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6, alpha=0.8)


def plot_top_products(
    order_products: pd.DataFrame,
    name_map: dict[int, str],
    top_n: int = 15,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Horizontal bar chart of the most frequent products."""
    counts = order_products["product_id"].value_counts().head(top_n)
    labels = [name_map.get(i, str(i)) for i in counts.index]
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(9, 6), facecolor=PALETTE["bg"])
    else:
        fig = ax.figure

    ax.barh(range(len(counts)), counts.values[::-1], color=PALETTE["accent"])
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel("Order appearances", color=PALETTE["ink"])
    _style_axes(ax, f"Top {top_n} products by frequency")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=PALETTE["bg"])
    return ax


def plot_itemset_support(
    itemsets: pd.DataFrame,
    size: int = 2,
    top_n: int = 12,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Bar chart of top frequent itemsets of a given size."""
    subset = itemsets[itemsets["size"] == size].head(top_n).copy()
    if subset.empty:
        raise ValueError(f"No itemsets of size {size}")
    labels = [" + ".join(names) for names in subset["itemset_names"]]
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=PALETTE["bg"])
    else:
        fig = ax.figure

    ax.barh(range(len(subset)), subset["support"].values[::-1], color=PALETTE["accent2"])
    ax.set_yticks(range(len(subset)))
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel("Support", color=PALETTE["ink"])
    _style_axes(ax, f"Top {size}-itemsets by support")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=PALETTE["bg"])
    return ax


def plot_rules_scatter(
    rules: pd.DataFrame,
    top_n: int = 40,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Scatter of confidence vs support, sized/colored by lift."""
    data = rules.head(top_n).copy()
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=PALETTE["bg"])
    else:
        fig = ax.figure

    sc = ax.scatter(
        data["support"],
        data["confidence"],
        s=np.clip(data["lift"] * 40, 40, 400),
        c=data["lift"],
        cmap="YlOrRd",
        alpha=0.85,
        edgecolors=PALETTE["ink"],
        linewidths=0.4,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Lift", color=PALETTE["ink"])
    ax.set_xlabel("Support", color=PALETTE["ink"])
    ax.set_ylabel("Confidence", color=PALETTE["ink"])
    _style_axes(ax, "Association rules: support vs confidence (size/color = lift)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=PALETTE["bg"])
    return ax


def plot_cooccurrence_network(
    pmi_df: pd.DataFrame,
    top_edges: int = 25,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Network graph of high-PMI product pairs."""
    edges = pmi_df.head(top_edges)
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(row["name_a"], row["name_b"], weight=row["pmi"], support=row["support"])

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(11, 8), facecolor=PALETTE["bg"])
    else:
        fig = ax.figure

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No edges to plot", ha="center", va="center")
        return ax

    pos = nx.spring_layout(G, k=1.4 / np.sqrt(max(G.number_of_nodes(), 1)), seed=42)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    widths = [0.8 + 1.8 * (w / max(weights)) for w in weights]

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=PALETTE["accent"], node_size=700, alpha=0.9
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax, width=widths, edge_color=PALETTE["accent2"], alpha=0.7
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color=PALETTE["ink"])
    ax.set_axis_off()
    ax.set_facecolor(PALETTE["bg"])
    ax.set_title(
        "Product affinity network (edge weight ∝ PMI)",
        color=PALETTE["ink"],
        fontsize=13,
        pad=12,
        fontweight="bold",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=PALETTE["bg"])
    return ax


def plot_metric_comparison(
    apriori_df: pd.DataFrame,
    fpgrowth_df: pd.DataFrame,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Compare itemset counts by size for Apriori vs FP-Growth."""
    a_counts = apriori_df.groupby("size").size()
    f_counts = fpgrowth_df.groupby("size").size()
    sizes = sorted(set(a_counts.index) | set(f_counts.index))
    a_vals = [int(a_counts.get(s, 0)) for s in sizes]
    f_vals = [int(f_counts.get(s, 0)) for s in sizes]

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor=PALETTE["bg"])
    else:
        fig = ax.figure

    x = np.arange(len(sizes))
    width = 0.35
    ax.bar(x - width / 2, a_vals, width, label="Apriori", color=PALETTE["accent"])
    ax.bar(x + width / 2, f_vals, width, label="FP-Growth", color=PALETTE["accent2"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Itemset size", color=PALETTE["ink"])
    ax.set_ylabel("Frequent itemsets", color=PALETTE["ink"])
    ax.legend(frameon=False)
    _style_axes(ax, "Apriori vs FP-Growth: frequent itemsets by size")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=PALETTE["bg"])
    return ax


def plot_pmi_heatmap_pairs(
    pmi_df: pd.DataFrame,
    top_n: int = 12,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Simple ranked PMI bars for top pairs."""
    data = pmi_df.head(top_n).copy()
    labels = [f"{a} ↔ {b}" for a, b in zip(data["name_a"], data["name_b"])]
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=PALETTE["bg"])
    else:
        fig = ax.figure

    ax.barh(range(len(data)), data["pmi"].values[::-1], color=PALETTE["accent"])
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel("PMI", color=PALETTE["ink"])
    _style_axes(ax, f"Top {top_n} product pairs by PMI")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=PALETTE["bg"])
    return ax
