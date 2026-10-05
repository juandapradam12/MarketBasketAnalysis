"""Market basket analysis toolkit: miners, rules, recommenders, and visuals."""

from .apriori import apriori, frequent_itemsets_to_frame
from .eclat import eclat
from .fp_growth import fp_growth
from .rules import association_rules, annotate_rule_departments, filter_rules_by_scope
from .affinity import pairwise_pmi, cooccurrence_matrix
from .sequential import sequential_transitions, user_sequences_from_orders
from .recommend import item_item_recommendations, svd_recommendations
from .benchmark import benchmark_miners
from .evaluate import pair_hit_rate
from .visualize import (
    plot_top_products,
    plot_itemset_support,
    plot_rules_scatter,
    plot_cooccurrence_network,
    plot_metric_comparison,
)

__all__ = [
    "apriori",
    "eclat",
    "fp_growth",
    "frequent_itemsets_to_frame",
    "association_rules",
    "annotate_rule_departments",
    "filter_rules_by_scope",
    "pairwise_pmi",
    "cooccurrence_matrix",
    "sequential_transitions",
    "user_sequences_from_orders",
    "item_item_recommendations",
    "svd_recommendations",
    "benchmark_miners",
    "pair_hit_rate",
    "plot_top_products",
    "plot_itemset_support",
    "plot_rules_scatter",
    "plot_cooccurrence_network",
    "plot_metric_comparison",
]
