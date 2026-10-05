#!/usr/bin/env python3
"""End-to-end market-basket analysis pipeline.

Runs Apriori / FP-Growth / Eclat, association rules (with Zhang + utility),
PMI affinity, sequential transitions, recommenders, holdout evaluation,
benchmarks, and static + interactive figure exports.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

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
from mba.eclat import eclat
from mba.evaluate import pair_hit_rate, recommend_recall_at_k
from mba.fp_growth import fp_growth
from mba.interactive import (
    interactive_benchmark,
    interactive_pmi_network,
    interactive_rules_scatter,
    interactive_top_itemsets,
)
from mba.recommend import item_item_recommendations, svd_recommendations
from mba.rules import annotate_rule_departments, association_rules
from mba.sequential import sequential_transitions, user_sequences_from_orders
from mba.visualize import (
    plot_cooccurrence_network,
    plot_itemset_support,
    plot_metric_comparison,
    plot_pmi_heatmap_pairs,
    plot_rules_scatter,
    plot_top_products,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--min-product-count", type=int, default=30)
    parser.add_argument("--min-support", type=float, default=0.01)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--min-lift", type=float, default=1.1)
    parser.add_argument("--max-len", type=int, default=4)
    parser.add_argument(
        "--reordered-only",
        choices=["all", "reordered", "first"],
        default="all",
    )
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    base = resolve_dataset_dir(ROOT, args.data_dir)
    print(f"Dataset dir: {base}")

    products = load_products(base / "products.csv")
    aisles = pd.read_csv(base / "aisles.csv") if (base / "aisles.csv").exists() else None
    departments = (
        pd.read_csv(base / "departments.csv") if (base / "departments.csv").exists() else None
    )
    products = enrich_products(products, aisles, departments)
    name_map = product_name_map(products)
    margin_map = product_margin_map(products)

    train = load_order_products(base / "order_products__train.csv")
    prior = (
        load_order_products(base / "order_products__prior.csv")
        if (base / "order_products__prior.csv").exists()
        else None
    )
    orders = load_orders(base / "orders.csv") if (base / "orders.csv").exists() else None

    def prep(op: pd.DataFrame) -> pd.DataFrame:
        df = op.copy()
        if args.reordered_only == "reordered":
            df = filter_reordered(df, True)
        elif args.reordered_only == "first":
            df = filter_reordered(df, False)
        df, _ = filter_frequent_products(df, min_count=args.min_product_count)
        return filter_orders_by_size(df, min_size=2)

    # Train mining on prior when available (holdout = train eval_set)
    mine_src = prior if prior is not None else train
    holdout_src = train if prior is not None else None

    compacted = prep(mine_src)
    transactions = transactions_from_orders(compacted)
    print(f"Train baskets: {len(transactions):,} | products: {compacted['product_id'].nunique():,}")

    # --- Benchmark all three miners ---
    timing = benchmark_miners(
        transactions, min_support=args.min_support, max_len=args.max_len
    )
    print("\nBenchmark:")
    print(timing.to_string(index=False))

    # --- Primary itemsets via Apriori + mirrors ---
    t0 = time.perf_counter()
    freq_apriori = apriori(transactions, min_support=args.min_support, max_len=args.max_len)
    t_apriori = time.perf_counter() - t0
    apriori_df = frequent_itemsets_to_frame(freq_apriori, name_map=name_map)

    freq_fp = fp_growth(transactions, min_support=args.min_support, max_len=args.max_len)
    fp_df = frequent_itemsets_to_frame(freq_fp, name_map=name_map)

    freq_eclat = eclat(transactions, min_support=args.min_support, max_len=args.max_len)
    eclat_df = frequent_itemsets_to_frame(freq_eclat, name_map=name_map)

    print(
        f"\nItemsets — Apriori: {len(apriori_df)} | FP-Growth: {len(fp_df)} | Eclat: {len(eclat_df)}"
    )

    rules = association_rules(
        freq_apriori,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        name_map=name_map,
        margin_map=margin_map or None,
    )
    rules = annotate_rule_departments(rules, products)
    print(f"Rules: {len(rules):,}")
    if not rules.empty and "scope" in rules.columns:
        print(rules["scope"].value_counts().to_string())

    pmi = pairwise_pmi(transactions, min_count=8, top_n=50, name_map=name_map)

    # Sequential
    transitions = pd.DataFrame()
    if orders is not None:
        op_all = mine_src.copy()
        seqs = user_sequences_from_orders(orders, op_all)
        transitions = sequential_transitions(
            seqs, min_support=0.008, min_confidence=0.12, name_map=name_map, top_n=40
        )
        print(f"Sequential transitions: {len(transitions):,}")

    # Recommenders demo seed = top product
    top_pid = int(compacted["product_id"].value_counts().index[0])
    cf_recs = item_item_recommendations(transactions, top_pid, top_n=10, name_map=name_map)
    svd_recs = svd_recommendations(transactions, [top_pid], top_n=10, name_map=name_map)

    # Holdout evaluation
    eval_metrics = {}
    if holdout_src is not None:
        holdout = prep(holdout_src)
        holdout_tx = transactions_from_orders(holdout)
        eval_metrics["rules"] = pair_hit_rate(rules, holdout_tx, top_k=20)

        def _rec(train_tx, seed, top_n=5):
            return item_item_recommendations(train_tx, seed, top_n=top_n, name_map=name_map)

        eval_metrics["item_item_recall"] = recommend_recall_at_k(
            transactions, holdout_tx, _rec, k=5, n_queries=80
        )
        print("\nHoldout evaluation:")
        print(json.dumps(eval_metrics, indent=2))

    fours = apriori_df[apriori_df["size"] == 4].head(5)
    print("\nTop 4-itemsets:")
    for _, row in fours.iterrows():
        print(f"  support={row['support']:.4f}  {list(row['itemset_names'])}")

    # Persist
    args.results_dir.mkdir(parents=True, exist_ok=True)
    apriori_df.to_csv(args.results_dir / "frequent_itemsets_apriori.csv", index=False)
    fp_df.to_csv(args.results_dir / "frequent_itemsets_fpgrowth.csv", index=False)
    eclat_df.to_csv(args.results_dir / "frequent_itemsets_eclat.csv", index=False)
    rules.to_csv(args.results_dir / "association_rules.csv", index=False)
    pmi.to_csv(args.results_dir / "pairwise_pmi.csv", index=False)
    timing.to_csv(args.results_dir / "timing.csv", index=False)
    cf_recs.to_csv(args.results_dir / "recs_item_item.csv", index=False)
    svd_recs.to_csv(args.results_dir / "recs_svd.csv", index=False)
    if not transitions.empty:
        transitions.to_csv(args.results_dir / "sequential_transitions.csv", index=False)
    if eval_metrics:
        (args.results_dir / "holdout_metrics.json").write_text(json.dumps(eval_metrics, indent=2))

    # Static figures
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_top_products(compacted, name_map, save_path=args.figures_dir / "01_top_products.png")
    plt.close("all")
    if not apriori_df[apriori_df["size"] == 2].empty:
        plot_itemset_support(apriori_df, size=2, save_path=args.figures_dir / "02_top_pairs.png")
        plt.close("all")
    if not apriori_df[apriori_df["size"] == 3].empty:
        plot_itemset_support(apriori_df, size=3, save_path=args.figures_dir / "03_top_triples.png")
        plt.close("all")
    if not apriori_df[apriori_df["size"] == 4].empty:
        plot_itemset_support(
            apriori_df, size=4, save_path=args.figures_dir / "04_top_quadruples.png"
        )
        plt.close("all")
    if not rules.empty:
        plot_rules_scatter(rules, save_path=args.figures_dir / "05_rules_scatter.png")
        plt.close("all")
    if not pmi.empty:
        plot_pmi_heatmap_pairs(pmi, save_path=args.figures_dir / "06_top_pmi_pairs.png")
        plt.close("all")
        plot_cooccurrence_network(pmi, save_path=args.figures_dir / "07_affinity_network.png")
        plt.close("all")
    plot_metric_comparison(apriori_df, fp_df, save_path=args.figures_dir / "08_apriori_vs_fpgrowth.png")
    plt.close("all")

    # Extra: 3-way timing bar via matplotlib
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(timing["method"], timing["seconds"], color=["#0f6e56", "#c45c26", "#3d5a80"])
    ax.set_ylabel("Seconds")
    ax.set_title("Apriori vs FP-Growth vs Eclat runtime")
    fig.tight_layout()
    fig.savefig(args.figures_dir / "09_miner_benchmark.png", dpi=160, bbox_inches="tight")
    plt.close("all")

    # Interactive HTML exports
    interactive_dir = args.figures_dir / "interactive"
    interactive_dir.mkdir(parents=True, exist_ok=True)
    interactive_rules_scatter(rules).write_html(interactive_dir / "rules_scatter.html")
    interactive_pmi_network(pmi).write_html(interactive_dir / "affinity_network.html")
    interactive_benchmark(timing).write_html(interactive_dir / "benchmark.html")
    if not apriori_df[apriori_df["size"] == 4].empty:
        interactive_top_itemsets(apriori_df, size=4).write_html(
            interactive_dir / "top_quadruples.html"
        )

    print(f"\nFigures -> {args.figures_dir}")
    print(f"Results  -> {args.results_dir}")
    print(f"Apriori wall time (primary): {t_apriori:.2f}s")


if __name__ == "__main__":
    main()
