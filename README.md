# Market Basket Analysis

Association mining on grocery baskets — a reusable **MBA toolkit** with Apriori, FP-Growth, Eclat, association rules, PMI networks, sequential patterns, recommenders, holdout evaluation, and a Streamlit demo.

Originally built around the [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis) dataset. The classic goal still holds: find products that appear together. This rewrite fixes the algorithm, adds complementary techniques, and ships visualizations plus a reproducible sample dataset.

<p align="center">
  <img src="figures/04_top_quadruples.png" alt="Top 4-itemsets by support" width="720"/>
</p>

---

## What's in this repo

| Path | Role |
|---|---|
| `Association Analysis.ipynb` | Original exploratory notebook (history) |
| `notebooks/enhanced_association_analysis.ipynb` | Walkthrough of the enhanced pipeline |
| `src/mba/` | Reusable Python package |
| `app/streamlit_app.py` | Interactive Streamlit lab |
| `scripts/generate_sample_data.py` | Instacart-like sample with planted associations |
| `scripts/run_analysis.py` | End-to-end CLI → CSV + PNG + Plotly HTML |
| `scripts/test_mba.py` | Unit tests |
| `data/sample/` | Generated sample CSVs (ready to run) |
| `data/raw/` | Drop real Instacart files here (gitignored) |
| `figures/` | Static + interactive visualizations |
| `results/` | Itemsets, rules, PMI, timing, holdout metrics |

---

## Problem with the original approach

The original notebook *mentioned* Apriori but actually:

1. Filtered to orders of **exactly** size 4 (should be **≥ 4**)
2. Enumerated all `itertools.combinations` (no support pruning)
3. Reported raw counts only — no confidence / lift / rules
4. Had almost no visualization

---

## Techniques included

### Frequent-itemset miners
| Method | Idea | Typical role |
|---|---|---|
| **Apriori** | Level-wise candidates + anti-monotonic prune | Teaching / reference |
| **FP-Growth** | Compressed FP-tree, no candidates | Fast production mining |
| **Eclat** | Vertical tid-list intersections | Sparse / depth-first baseline |

On the sample prior set (~7.3k baskets, `minsup=0.01`, `max_len=4`):

| Method | Itemsets | Runtime | Agrees with Apriori |
|---|---:|---:|:---:|
| Eclat | 590 | ~0.03 s | yes |
| FP-Growth | 590 | ~0.11 s | yes |
| Apriori | 590 | ~1.5 s | — |

### Association rules
Support, confidence, lift, leverage, conviction, plus:
- **Zhang** interestingness (signed dependence)
- **Utility** ≈ `lift × Σ margins` (synthetic `unit_margin` in sample data)
- **Scope** labels: `intra_aisle` / `intra_department` / `cross_department`

### Complementary views
- **PMI / Jaccard** affinity pairs + network graph
- **Sequential transitions** `A in order t ⇒ B in order t+1` (needs `orders.csv`)
- **Recommenders**: item-item cosine CF and TruncatedSVD
- **Holdout eval**: rule pair precision + recommender recall@k (prior → train)

### Filters
- Reordered vs first-time line items
- Department / aisle constrained mining (via product joins)
- Prefer `data/raw/` when real Instacart CSVs are present

---

## Quick start

```bash
pip install -r requirements.txt

# Sample data (already committed; regenerate if needed)
python scripts/generate_sample_data.py

# Full pipeline
python scripts/run_analysis.py \
  --min-support 0.01 \
  --min-confidence 0.2 \
  --min-lift 1.05

# Tests
python scripts/test_mba.py

# Streamlit demo
streamlit run app/streamlit_app.py
```

Optional: place real Instacart files in `data/raw/`:

```
order_products__train.csv
order_products__prior.csv
products.csv
orders.csv
aisles.csv
departments.csv
```

`run_analysis.py` / the Streamlit app prefer `data/raw/` automatically.

---

## Sample results

Top 4-itemsets recover the original notebook themes:

1. **Sparkling water flight** — Pure / Lemon / Lime / Grapefruit (`support ≈ 0.09`)
2. **Berry mix** — Strawberries / Blackberries / Organic Blueberries / Raspberries (`support ≈ 0.055`)

Holdout (rules trained on prior, scored on train): top-20 rule pair precision = **1.0**, basket coverage ≈ **13%**; item-item recall@5 ≈ **0.41** on the sample.

---

## Visualizations

### Top products & itemsets
<img src="figures/01_top_products.png" alt="Top products" width="640"/>
<img src="figures/04_top_quadruples.png" alt="Top quadruples" width="640"/>

### Rules & PMI
<img src="figures/05_rules_scatter.png" alt="Rules scatter" width="640"/>
<img src="figures/07_affinity_network.png" alt="Affinity network" width="720"/>

### Miner benchmark
<img src="figures/09_miner_benchmark.png" alt="Miner benchmark" width="560"/>

Interactive Plotly HTML (open in a browser):

- `figures/interactive/rules_scatter.html`
- `figures/interactive/affinity_network.html`
- `figures/interactive/benchmark.html`
- `figures/interactive/top_quadruples.html`

---

## Package API (short)

```python
from mba.apriori import apriori
from mba.eclat import eclat
from mba.fp_growth import fp_growth
from mba.rules import association_rules, annotate_rule_departments
from mba.affinity import pairwise_pmi
from mba.sequential import sequential_transitions
from mba.recommend import item_item_recommendations, svd_recommendations
from mba.benchmark import benchmark_miners
from mba.evaluate import pair_hit_rate

freq = apriori(transactions, min_support=0.01, max_len=4)
rules = association_rules(freq, min_confidence=0.3, min_lift=1.2, margin_map=margins)
rules = annotate_rule_departments(rules, products)
```

Add `src/` to `PYTHONPATH` (CLI scripts / Streamlit do this for you).

---

## Architecture

```text
orders + order_products + products
        │
        ├─► filter (freq / size / reordered / department)
        │
        ├─► Apriori / FP-Growth / Eclat ─► rules (+ Zhang, utility, scope)
        ├─► PMI affinity network
        ├─► sequential transitions (user timelines)
        ├─► item-item CF + SVD recommenders
        └─► holdout metrics (prior → train)
```

---

## License / data

Code is for personal / educational use. If you use the official Instacart 2017 dataset, follow Instacart’s non-commercial terms and cite:

> “The Instacart Online Grocery Shopping Dataset 2017”
