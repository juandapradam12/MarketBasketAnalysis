"""Interactive Plotly visualizations for notebooks and Streamlit."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx


def interactive_rules_scatter(rules: pd.DataFrame, top_n: int = 60) -> go.Figure:
    """Support vs confidence scatter with lift coloring and hover labels."""
    data = rules.head(top_n).copy()
    if data.empty:
        return go.Figure()
    data["rule"] = [
        f"{' + '.join(map(str, a))} → {' + '.join(map(str, c))}"
        for a, c in zip(data["antecedent_names"], data["consequent_names"])
    ]
    fig = px.scatter(
        data,
        x="support",
        y="confidence",
        size="lift",
        color="lift",
        hover_name="rule",
        hover_data=["zhang", "utility", "scope"]
        if "scope" in data.columns
        else ["zhang", "utility"],
        color_continuous_scale="YlOrRd",
        title="Association rules (interactive)",
    )
    fig.update_layout(template="plotly_white", height=520)
    return fig


def interactive_pmi_network(pmi_df: pd.DataFrame, top_edges: int = 25) -> go.Figure:
    """Interactive affinity network from PMI pairs."""
    edges = pmi_df.head(top_edges)
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(row["name_a"], row["name_b"], weight=float(row["pmi"]))

    if G.number_of_nodes() == 0:
        return go.Figure()

    pos = nx.spring_layout(G, seed=42, k=1.3)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_text = list(G.nodes())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1.2, color="#c45c26"),
            hoverinfo="none",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker=dict(size=18, color="#0f6e56"),
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Product affinity network (PMI)",
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=560,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def interactive_benchmark(timing: pd.DataFrame) -> go.Figure:
    """Bar chart comparing miner runtimes."""
    fig = px.bar(
        timing,
        x="method",
        y="seconds",
        color="method",
        text="itemsets",
        title="Frequent-itemset miner runtime",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(template="plotly_white", showlegend=False, height=420)
    return fig


def interactive_top_itemsets(itemsets: pd.DataFrame, size: int = 4, top_n: int = 10) -> go.Figure:
    subset = itemsets[itemsets["size"] == size].head(top_n).copy()
    if subset.empty:
        return go.Figure()
    subset["label"] = [" + ".join(map(str, n)) for n in subset["itemset_names"]]
    fig = px.bar(
        subset.sort_values("support"),
        x="support",
        y="label",
        orientation="h",
        title=f"Top {size}-itemsets by support",
        color="support",
        color_continuous_scale="Teal",
    )
    fig.update_layout(template="plotly_white", height=480, yaxis_title="")
    return fig
