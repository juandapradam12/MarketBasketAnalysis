# Market Basket Analysis

Association mining on grocery baskets — upgraded from a brute-force notebook into a small reusable toolkit with **Apriori**, **FP-Growth**, **association rules**, and **PMI affinity networks**.

Originally built around the [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis) dataset. The classic homework goal still holds: find the product collections that appear together most often. This rewrite fixes the algorithm, adds complementary techniques, and ships visualizations plus a reproducible sample dataset.

<p align="center">
  <img src="figures/04_top_quadruples.png" alt="Top 4-itemsets by support" width="720"/>
</p>

---

## What's in this repo

| Path | Role |
|---|---|
| `Association Analysis.ipynb` | Original exploratory notebook (kept for history) |
| `notebooks/enhanced_association_analysis.ipynb` | Walkthrough of the enhanced pipeline |
| `src/mba/` | Reusable Python package (algorithms + plots) |
| `scripts/generate_sample_data.py` | Instacart-like sample with planted associations |
| `scripts/run_analysis.py` | End-to-end CLI → CSV results + PNG figures |
| `scripts/test_mba.py` | Unit checks (Apriori ≡ FP-Growth, rules, PMI) |
| `data/sample/` | Generated sample CSVs (ready to run) |
| `data/raw/` | Drop real Instacart files here (gitignored) |
| `figures/` | Generated visualizations |
| `results/` | Frequent itemsets, rules, PMI tables |

---

## Problem with the original approach

The original notebook *mentioned* Apriori but actually:

1. Filtered to orders of **exactly** size 4 (should be **≥ 4** for 4-itemsets)
2. Enumerated all `itertools.combinations` inside each basket (no support pruning)
3. Reported raw appearance counts only — no **confidence**, **lift**, or rule form
4. Had no visualizations beyond tabular output

That works on a heavily cut-down slice, but it is neither Apriori nor scalable.

### What “real” Apriori does

```
L1 = frequent 1-itemsets (support ≥ minsup)
for k = 2, 3, ...:
    Ck = join(Lk-1)          # candidate generation
    Ck = prune(Ck, Lk-1)     # drop candidates with infrequent subsets
    Lk = { c ∈ Ck | support(c) ≥ minsup }
```

Anti-monotonicity: **if an itemset is infrequent, every superset is infrequent.** That prune is what makes association mining tractable.

---

## Techniques included

### 1. Apriori (`src/mba/apriori.py`)
Level-wise frequent-itemset mining with candidate join + subset prune. Good for teaching and for verifying thresholds.

### 2. FP-Growth (`src/mba/fp_growth.py`)
Builds a compressed FP-tree and mines patterns without candidate generation (via `mlxtend`). Same output contract as Apriori → easy A/B comparison.

On the bundled sample (8,000 baskets, `minsup=0.008`, `max_len=4`):

| Method | Itemsets | Runtime |
|---|---:|---:|
| Apriori | 678 | ~2.2 s |
| FP-Growth | 678 | ~0.2 s |

### 3. Association rules (`src/mba/rules.py`)
From frequent itemsets, derive `A → B` with:

| Metric | Meaning |
|---|---|
| **Support** | `P(A ∪ B)` — how common the whole set is |
| **Confidence** | `P(B \| A)` — reliability of the implication |
| **Lift** | `conf / P(B)` — >1 means positive dependence |
| **Leverage** | `P(A∪B) − P(A)P(B)` — absolute surprise |
| **Conviction** | how strongly B depends on A |

### 4. Pairwise PMI / Jaccard (`src/mba/affinity.py`)
Complementary to support-based mining. Popular items dominate raw counts; **PMI** surfaces pairs that co-occur *more than chance*, which is useful for niche but strong affinities (flavored waters, berry mixes, savory veg).

---

## Quick start

```bash
# 1. Dependencies
pip install -r requirements.txt

# 2. Sample data (already committed under data/sample/, regenerate if needed)
python scripts/generate_sample_data.py

# 3. Run the full pipeline
python scripts/run_analysis.py \
  --min-support 0.008 \
  --min-confidence 0.2 \
  --min-lift 1.05

# 4. Tests
python scripts/test_mba.py
```

Optional: place real Instacart files at

```
data/raw/order_products__train.csv
data/raw/products.csv
```

`run_analysis.py` prefers `data/raw/` automatically when those files exist. (The old public S3 mirror is gone; Kaggle is the usual source.)

Interactive exploration:

```bash
jupyter notebook notebooks/enhanced_association_analysis.ipynb
```

---

## Sample results

With the planted sample, the top 4-itemsets recover the same themes the original notebook found on Instacart train data:

1. **Sparkling water flight** — Pure / Lemon / Lime / Grapefruit (`support ≈ 0.093`)
2. **Berry mix** — Strawberries / Blackberries / Organic Blueberries / Raspberries (`support ≈ 0.054`)

High-lift rules tend to sit inside those thematic clusters (veg prep kits, berry pairs, water flavors), which is exactly what you want for cross-sell / bundle design.

---

## Visualizations

### Top products
<img src="figures/01_top_products.png" alt="Top products" width="640"/>

### Frequent pairs & triples
<img src="figures/02_top_pairs.png" alt="Top pairs" width="640"/>
<img src="figures/03_top_triples.png" alt="Top triples" width="640"/>

### Top 4-itemsets (original project goal)
<img src="figures/04_top_quadruples.png" alt="Top quadruples" width="640"/>

### Association rules — support vs confidence (size/color = lift)
<img src="figures/05_rules_scatter.png" alt="Rules scatter" width="640"/>

### PMI ranking & affinity network
<img src="figures/06_top_pmi_pairs.png" alt="Top PMI pairs" width="640"/>
<img src="figures/07_affinity_network.png" alt="Affinity network" width="720"/>

### Apriori vs FP-Growth coverage
<img src="figures/08_apriori_vs_fpgrowth.png" alt="Apriori vs FP-Growth" width="560"/>

---

## Algorithm cheat-sheet

```text
Baskets ──► filter frequent SKUs ──► transactions (frozensets)
                 │
                 ├─► Apriori  ──┐
                 │              ├─► frequent itemsets ──► association rules
                 └─► FP-Growth ─┘
                 │
                 └─► pairwise PMI / Jaccard ──► affinity network
```

**When to use what**

| Goal | Prefer |
|---|---|
| Teaching / verifying support prune | Apriori |
| Faster mining on denser baskets | FP-Growth |
| “Customers who bought A also bought B” | Association rules (lift + confidence) |
| Surprising niche affinities | PMI / Jaccard network |
| Shelf / bundle themes | 3–4 itemsets by support *and* high-PMI clusters |

---

## Package API (short)

```python
from mba.apriori import apriori, frequent_itemsets_to_frame
from mba.fp_growth import fp_growth
from mba.rules import association_rules
from mba.affinity import pairwise_pmi

freq = apriori(transactions, min_support=0.01, max_len=4)
rules = association_rules(freq, min_confidence=0.3, min_lift=1.2)
pmi = pairwise_pmi(transactions, min_count=5, top_n=30)
```

Add `src/` to `PYTHONPATH` (the CLI scripts and notebook do this for you).

---

## Design notes / future ideas

- **Eclat / vertical tid-lists** — another frequent-itemset baseline that shines when baskets are sparse.
- **Sequence / next-item models** — Instacart also has order timestamps; Markov or transformer recommenders go beyond unordered baskets.
- **Department-aware mining** — constrain candidates within / across aisles to reduce noise.
- **Interactive viz** — Plotly/PyVis hover tooltips on the affinity network.
- **Calibration on full Instacart** — retune `min_support` (often ≪ 0.01 on 130k+ train orders).

---

## License / data

Code in this repository is for personal / educational use. If you use the official Instacart 2017 dataset, follow Instacart’s non-commercial terms and cite:

> “The Instacart Online Grocery Shopping Dataset 2017”, Accessed from https://www.instacart.com/datasets/grocery-shopping-2017
