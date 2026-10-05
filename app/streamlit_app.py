"""Streamlit demo for the market-basket analysis toolkit."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mba.affinity import pairwise_pmi
from mba.apriori import apriori, frequent_itemsets_to_frame
from mba.benchmark import benchmark_miners
from mba.data import (
    enrich_products,
    filter_frequent_products,
    filter_orders_by_size,
    filter_reordered,
    load_order_products,
    load_orders,
    load_products,
    product_margin_map,
    product_name_map,
    resolve_dataset_dir,
    transactions_from_orders,
)
from mba.interactive import (
    interactive_benchmark,
    interactive_pmi_network,
    interactive_rules_scatter,
    interactive_top_itemsets,
)
from mba.recommend import item_item_recommendations, svd_recommendations
from mba.rules import annotate_rule_departments, association_rules, filter_rules_by_scope
from mba.sequential import sequential_transitions, user_sequences_from_orders


st.set_page_config(page_title="Market Basket Lab", layout="wide")
st.title("Market Basket Lab")
st.caption("Apriori · FP-Growth · Eclat · rules · PMI · sequential · recommenders")


@st.cache_data(show_spinner=False)
def load_bundle():
    base = resolve_dataset_dir(ROOT)
    products = load_products(base / "products.csv")
    aisles = pd.read_csv(base / "aisles.csv") if (base / "aisles.csv").exists() else None
    departments = (
        pd.read_csv(base / "departments.csv") if (base / "departments.csv").exists() else None
    )
    products = enrich_products(products, aisles, departments)
    train = load_order_products(base / "order_products__train.csv")
    prior_path = base / "order_products__prior.csv"
    prior = load_order_products(prior_path) if prior_path.exists() else None
    orders_path = base / "orders.csv"
    orders = load_orders(orders_path) if orders_path.exists() else None
    return base, products, train, prior, orders


base, products, train, prior, orders = load_bundle()
name_map = product_name_map(products)
margin_map = product_margin_map(products)

with st.sidebar:
    st.header("Controls")
    st.write(f"Data: `{base}`")
    min_support = st.slider("Min support", 0.005, 0.05, 0.01, 0.001)
    min_confidence = st.slider("Min confidence", 0.1, 0.8, 0.25, 0.05)
    min_lift = st.slider("Min lift", 1.0, 5.0, 1.1, 0.1)
    max_len = st.selectbox("Max itemset size", [2, 3, 4], index=2)
    reorder_mode = st.selectbox(
        "Line items",
        ["all", "reordered only", "first-time only"],
    )
    scope = st.selectbox(
        "Rule scope",
        ["all", "intra_aisle", "intra_department", "cross_department"],
    )
    run = st.button("Run mining", type="primary")


def prepare(op: pd.DataFrame) -> list:
    df = op.copy()
    if reorder_mode == "reordered only":
        df = filter_reordered(df, True)
    elif reorder_mode == "first-time only":
        df = filter_reordered(df, False)
    df, _ = filter_frequent_products(df, min_count=20)
    df = filter_orders_by_size(df, min_size=2)
    return transactions_from_orders(df)


if run or "freq" not in st.session_state:
    with st.spinner("Mining frequent itemsets and rules..."):
        tx = prepare(train)
        freq = apriori(tx, min_support=min_support, max_len=max_len)
        itemsets = frequent_itemsets_to_frame(freq, name_map=name_map)
        rules = association_rules(
            freq,
            min_confidence=min_confidence,
            min_lift=min_lift,
            name_map=name_map,
            margin_map=margin_map or None,
        )
        rules = annotate_rule_departments(rules, products)
        if scope != "all":
            rules = filter_rules_by_scope(rules, scope)
        pmi = pairwise_pmi(tx, min_count=5, top_n=40, name_map=name_map)
        timing = benchmark_miners(tx, min_support=min_support, max_len=max_len)
        st.session_state.update(
            {
                "tx": tx,
                "freq": freq,
                "itemsets": itemsets,
                "rules": rules,
                "pmi": pmi,
                "timing": timing,
            }
        )

itemsets = st.session_state["itemsets"]
rules = st.session_state["rules"]
pmi = st.session_state["pmi"]
timing = st.session_state["timing"]
tx = st.session_state["tx"]

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Itemsets & rules", "Affinity network", "Benchmark", "Sequential", "Recommend"]
)

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Baskets", f"{len(tx):,}")
    c2.metric("Frequent itemsets", f"{len(itemsets):,}")
    c3.metric("Rules", f"{len(rules):,}")
    st.plotly_chart(interactive_top_itemsets(itemsets, size=min(max_len, 4)), use_container_width=True)
    st.plotly_chart(interactive_rules_scatter(rules), use_container_width=True)
    st.dataframe(rules.head(30), use_container_width=True)

with tab2:
    st.plotly_chart(interactive_pmi_network(pmi), use_container_width=True)
    st.dataframe(pmi.head(25), use_container_width=True)

with tab3:
    st.plotly_chart(interactive_benchmark(timing), use_container_width=True)
    st.dataframe(timing, use_container_width=True)

with tab4:
    if orders is None:
        st.info("No orders.csv found — generate sample data to enable sequential mining.")
    else:
        op_all = train if prior is None else pd.concat([prior, train], ignore_index=True)
        seqs = user_sequences_from_orders(orders, op_all)
        transitions = sequential_transitions(
            seqs, min_support=0.01, min_confidence=0.15, name_map=name_map, top_n=40
        )
        st.subheader("Next-order transitions (A in order t ⇒ B in t+1)")
        st.dataframe(transitions, use_container_width=True)

with tab5:
    product_options = {
        f"{name_map[i]} ({i})": i
        for i in sorted(name_map.keys(), key=lambda x: name_map[x])
    }
    choice = st.selectbox("Seed product", list(product_options.keys()))
    seed = product_options[choice]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Item-item CF (cosine)**")
        st.dataframe(
            item_item_recommendations(tx, seed, top_n=10, name_map=name_map),
            use_container_width=True,
        )
    with c2:
        st.markdown("**SVD latent factors**")
        st.dataframe(
            svd_recommendations(tx, [seed], top_n=10, name_map=name_map),
            use_container_width=True,
        )
