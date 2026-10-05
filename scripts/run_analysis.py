#!/usr/bin/env python3
"""End-to-end market-basket analysis: Apriori, FP-Growth, rules, PMI, figures."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mba.affinity import pairwise_pmi
from mba.apriori import apriori, frequent_itemsets_to_frame
from mba.data import (
    filter_frequent_products,
    filter_orders_by_size,
    load_order_products,
    load_products,
    product_name_map,
    transactions_from_orders,
)
from mba.fp_growth import fp_growth
from mba.rules import association_rules
from mba.visualize import (
    plot_cooccurrence_network,
    plot_itemset_support,
    plot_metric_comparison,
    plot_pmi_heatmap_pairs,
    plot_rules_scatter,
    plot_top_products,
)


def resolve_data_paths(data_dir: Path | None) -> tuple[Path, Path]:
    """Prefer real Instacart CSVs in data/raw, else fall back to sample."""
    candidates = []
    if data_dir is not None:
        candidates.append(data_dir)
    candidates.extend([ROOT / "data" / "raw", ROOT / "data" / "sample", ROOT])

    for base in candidates:
        op = base / "order_products__train.csv"
        pr = base / "products.csv"
        if op.exists() and pr.exists():
            return op, pr
    raise FileNotFoundError(
        "Could not find order_products__train.csv and products.csv. "
        "Run: python scripts/generate_sample_data.py"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--min-product-count", type=int, default=40)
    parser.add_argument("--min-support", type=float, default=0.01)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--min-lift", type=float, default=1.1)
    parser.add_argument("--max-len", type=int, default=4)
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    op_path, pr_path = resolve_data_paths(args.data_dir)
    print(f"Using orders:   {op_path}")
    print(f"Using products: {pr_path}")

    order_products = load_order_products(op_path)
    products = load_products(pr_path)
    name_map = product_name_map(products)

    # Preprocess: frequent items + baskets with enough items for size-k mining
    compacted, _ = filter_frequent_products(order_products, min_count=args.min_product_count)
    compacted = filter_orders_by_size(compacted, min_size=2)
    transactions = transactions_from_orders(compacted)
    n_orders = len(transactions)
    print(f"Baskets after filtering: {n_orders:,}")
    print(f"Unique products kept:    {compacted['product_id'].nunique():,}")

    # --- Apriori ---
    t0 = time.perf_counter()
    freq_apriori = apriori(transactions, min_support=args.min_support, max_len=args.max_len)
    t_apriori = time.perf_counter() - t0
    apriori_df = frequent_itemsets_to_frame(freq_apriori, name_map=name_map)
    print(f"Apriori:   {len(apriori_df):,} itemsets in {t_apriori:.2f}s")

    # --- FP-Growth ---
    t0 = time.perf_counter()
    freq_fp = fp_growth(transactions, min_support=args.min_support, max_len=args.max_len)
    t_fp = time.perf_counter() - t0
    fp_df = frequent_itemsets_to_frame(freq_fp, name_map=name_map)
    print(f"FP-Growth: {len(fp_df):,} itemsets in {t_fp:.2f}s")

    # --- Rules (from Apriori itemsets) ---
    rules = association_rules(
        freq_apriori,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        name_map=name_map,
    )
    print(f"Rules:     {len(rules):,} (min_conf={args.min_confidence}, min_lift={args.min_lift})")

    # --- Pairwise PMI ---
    pmi = pairwise_pmi(transactions, min_count=8, top_n=50, name_map=name_map)
    print(f"PMI pairs: {len(pmi):,}")

    # Highlight original notebook goal: top 4-itemsets
    fours = apriori_df[apriori_df["size"] == 4].head(5)
    print("\nTop 4-itemsets (Apriori):")
    if fours.empty:
        print("  (none above min_support — try lowering --min-support)")
    else:
        for _, row in fours.iterrows():
            print(f"  support={row['support']:.4f}  {list(row['itemset_names'])}")

    # Persist tables
    args.results_dir.mkdir(parents=True, exist_ok=True)
    apriori_df.to_csv(args.results_dir / "frequent_itemsets_apriori.csv", index=False)
    fp_df.to_csv(args.results_dir / "frequent_itemsets_fpgrowth.csv", index=False)
    rules.to_csv(args.results_dir / "association_rules.csv", index=False)
    pmi.to_csv(args.results_dir / "pairwise_pmi.csv", index=False)
    timing = (
        f"method,seconds,itemsets\n"
        f"apriori,{t_apriori:.4f},{len(apriori_df)}\n"
        f"fp_growth,{t_fp:.4f},{len(fp_df)}\n"
    )
    (args.results_dir / "timing.csv").write_text(timing)

    # Figures
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_top_products(
        compacted, name_map, top_n=15, save_path=args.figures_dir / "01_top_products.png"
    )
    plt.close("all")

    if not apriori_df[apriori_df["size"] == 2].empty:
        plot_itemset_support(
            apriori_df,
            size=2,
            top_n=12,
            save_path=args.figures_dir / "02_top_pairs.png",
        )
        plt.close("all")

    if not apriori_df[apriori_df["size"] == 3].empty:
        plot_itemset_support(
            apriori_df,
            size=3,
            top_n=10,
            save_path=args.figures_dir / "03_top_triples.png",
        )
        plt.close("all")

    if not apriori_df[apriori_df["size"] == 4].empty:
        plot_itemset_support(
            apriori_df,
            size=4,
            top_n=8,
            save_path=args.figures_dir / "04_top_quadruples.png",
        )
        plt.close("all")

    if not rules.empty:
        plot_rules_scatter(rules, top_n=50, save_path=args.figures_dir / "05_rules_scatter.png")
        plt.close("all")

    if not pmi.empty:
        plot_pmi_heatmap_pairs(pmi, top_n=15, save_path=args.figures_dir / "06_top_pmi_pairs.png")
        plt.close("all")
        plot_cooccurrence_network(
            pmi, top_edges=22, save_path=args.figures_dir / "07_affinity_network.png"
        )
        plt.close("all")

    plot_metric_comparison(
        apriori_df, fp_df, save_path=args.figures_dir / "08_apriori_vs_fpgrowth.png"
    )
    plt.close("all")

    print(f"\nFigures -> {args.figures_dir}")
    print(f"Results  -> {args.results_dir}")


if __name__ == "__main__":
    main()
