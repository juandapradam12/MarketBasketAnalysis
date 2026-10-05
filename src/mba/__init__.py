"""Market basket analysis toolkit: Apriori, FP-Growth, rules, and visuals."""

from .apriori import apriori, frequent_itemsets_to_frame
from .rules import association_rules
from .fp_growth import fp_growth
from .affinity import pairwise_pmi, cooccurrence_matrix
from .visualize import (
    plot_top_products,
    plot_itemset_support,
    plot_rules_scatter,
    plot_cooccurrence_network,
    plot_metric_comparison,
)

__all__ = [
    "apriori",
    "frequent_itemsets_to_frame",
    "association_rules",
    "fp_growth",
    "pairwise_pmi",
    "cooccurrence_matrix",
    "plot_top_products",
    "plot_itemset_support",
    "plot_rules_scatter",
    "plot_cooccurrence_network",
    "plot_metric_comparison",
]
