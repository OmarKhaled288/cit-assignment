# Customer Segmentation of an Online Retailer — Technical Report

**Objective:** Use unsupervised learning (K-Means) to group customers into actionable segments, so the business can treat loyal high-value customers differently from one-off / lapsed buyers.
**Tooling:** `polars` (data wrangling), `scikit-learn` (scaling, K-Means, validation metrics, PCA), `matplotlib` / `seaborn` / `plotly` (visualisation).

---

## 1. Executive summary

We cleaned the raw transaction log down to a reliable UK product-sales dataset, aggregated it to **one row per customer** with six RFM-style features, transformed and scaled those features, and clustered them with K-Means. Three independent internal validation metrics **unanimously selected k = 2**, producing two clearly interpretable segments:

| Segment | Size | Orders | Lifetime spend | Recency | Tenure | Marketing posture |
|---|---|---|---|---|---|---|
| **Loyal / high-value** | ~1,794 | ~7.4 | ~£3,350 | ~36 days (recent) | ~244 days (long) | **Retain & reward** |
| **One-off / lapsed** | ~2,114 | ~1.5 | ~£388 | ~139 days (cold) | ~34 days (short) | **Reactivate / win-back** |

The split is driven primarily by **engagement** (frequency, recency, tenure) rather than basket size, and is visually separable in PCA space (79.1% of variance captured in two dimensions).

---

## 2. Methodology overview

The pipeline runs in six stages, each justified by what the previous stage revealed:

1. **Exploratory Data Analysis (EDA)** — understand structure, coverage and quality.
2. **Cleaning & scoping** — remove unusable rows and restrict to a coherent population.
3. **Feature engineering** — aggregate transactions into customer-level RFM features.
4. **Preprocessing** — log-transform skew, then standardise.
5. **Model selection & fitting** — choose *k*, fit K-Means.
6. **Profiling & validation** — describe and validate the segments.

---

## 3. Exploratory Data Analysis

EDA here is not decorative — every chart answers a question that directly drives a later decision.

### 3.1 Transaction volume over time

![Invoices over time](<graphs/Invoices over time.png>)

Line-item volume is broadly stable (~30k–40k/month) through most of the year, then ramps **sharply from September to November 2011** — the pre-Christmas gift season — before collapsing in December. The December drop is an **artefact, not a trend**: the data ends on 2011-12-09, so that month is only one-third observed. *Takeaway:* the dataset spans a full annual cycle with strong Q4 seasonality, and the final month must not be read as a decline.

### 3.2 Missing customer IDs

![CustomerID nulls](<graphs/CustomerID nulls.png>)

About **135,000 of 542,000 rows (~25%) have no `CustomerID`**. Because the entire analysis is *customer*-centric, these rows cannot be attributed to anyone and are unusable for segmentation. *Decision:* drop them — the single largest cleaning step.

### 3.3 Non-positive quantities and prices

![Non-positive values](<graphs/Non-positive values.png>)

There are ~10,600 rows with `Quantity ≤ 0` and ~2,500 with `UnitPrice ≤ 0`. These are **returns, cancellations, freebies and manual adjustments**, not genuine sales. The raw data even contains a quantity of −80,995 and a unit price of −11,062. *Decision:* these must be handled before computing spend, or they will distort every monetary feature.

### 3.4 Geographic concentration

![Top 12 countries — UK dominates](<graphs/Top 12 countries by transaction share — the UK dominates (~89%).png>)

The **United Kingdom alone accounts for ~89%** of all transactions. The next largest markets — Germany (2.3%), France (2.1%), EIRE (1.8%) — are rounding errors by comparison.

![Top 12 countries excluding the UK](<graphs/Top 12 countries by transaction share minus the UK.png>)

Even after removing the UK, no single foreign market exceeds ~21% *of the small remainder*. These volumes are too thin to segment reliably and likely follow different purchasing and shipping behaviour. *Decision:* **restrict the analysis to the UK** and model it as one coherent population, rather than letting a long tail of tiny markets inject noise.

---

## 4. Data cleaning & scoping

Driven by the EDA findings, cleaning proceeds in steps, re-inspecting after each:

1. **Drop rows with null `CustomerID`** (§3.2) — required for customer aggregation.
2. **Restrict to the United Kingdom** (§3.4) — drop the `Country` column afterwards.
3. **Remove non-product line items** — `StockCode`s that are not the usual 5-digit product codes (postage `POST`, samples, the discount code `D`, manual adjustments, etc.). These are not products a customer "buys" in a segmentation sense.
4. **Remove returns and their matching original sale** — invoices starting with `C` are cancellations. We negate their quantity and use an **anti-join** on `(CustomerID, StockCode, Quantity, UnitPrice)` to remove both the return *and* the original purchase it reverses, so a bought-then-returned item nets to zero rather than inflating apparent spend.
5. **Add `total_amount = Quantity × UnitPrice`** — line-level revenue, the basis for the monetary features.
6. **Remove duplicate rows** — remove any rows that appear more than once in the dataset after all the cleaning steps.

![Data cleaning results](<graphs/Data cleaning results.png>)

The waterfall of row counts: **541,909 raw → 344,751 retained** (~197k rows / ~36% dropped). The bulk of the loss is the missing-customer rows; the rest is non-UK, non-product and return-related. What remains is a clean ledger of genuine UK product purchases tied to identifiable customers.

### 4.1 Post-cleaning sanity checks

![UK monthly revenue and order volume](<graphs/UK monthly revenue and order volume.png>)

Monthly **revenue (bars)** and **distinct orders (line)** move together and show the same healthy Q4 ramp, peaking around November 2011 (~£0.95M, ~2,300 orders). Coverage is continuous with no gaps, and the thin December tail is again the truncation artefact. This confirms the cleaned data is temporally sound.

![Transaction-level distributions](<graphs/Transaction-level distributions.png>)

`Quantity`, `UnitPrice` and `total_amount` are plotted on a **log y-axis**. All three are extremely **right-skewed**: the overwhelming mass sits at small values, with a long thin tail of very large orders (a few wholesale-scale purchases). *Takeaway:* any feature built from these will inherit the skew — which is exactly why a log transform is applied before modelling (§6).

---

## 5. Feature engineering — RFM-style customer profiles

Clustering must operate on **customers**, not individual transactions, so the cleaned line items are aggregated to one row per `CustomerID`. The six features follow the classic **RFM** (Recency, Frequency, Monetary) marketing framework, plus engagement breadth:

| Feature | RFM dimension | Definition | What it captures |
|---|---|---|---|
| `num_orders` | **Frequency** | distinct invoices | how often they buy |
| `num_products` | breadth | distinct stock codes | how broadly they shop |
| `total_spent` | **Monetary** | Σ `total_amount` | lifetime value |
| `avg_order_value` | Monetary | `total_spent / num_orders` | typical basket size |
| `tenure_days` | engagement | last − first order | how long they've been active |
| `recency_days` | **Recency** | snapshot date − last order | how cold/warm they are now |

The **snapshot date** (`ref_date`) is the most recent activity in the dataset, so recency is measured relative to "now" in the data's own frame of reference. This yields **~3,908 customers**.

![Customer-level feature distributions](<graphs/Customer-level feature distributions.png>)

On a log count axis, the four count/money features (`num_orders`, `num_products`, `total_spent`, `avg_order_value`) are heavily **right-skewed** — a handful of "whale" customers dwarf everyone else. `tenure_days` is bimodal (a spike at ~0 for single-purchase customers, plus a spread up to ~365), and `recency_days` is spread across the whole year. *Takeaway:* the monetary/count features need a log transform; recency and tenure, being naturally bounded by the ~1-year window, can stay linear.

---

## 6. Preprocessing — transform & scale

Two steps prepare the features for a distance-based algorithm:

- **`log1p` on the skewed columns** (`num_orders`, `num_products`, `total_spent`, `avg_order_value`). `log1p(x) = log(1 + x)` compresses the long right tail so a few extreme spenders don't dominate the distance metric, and it handles zeros safely. Recency and tenure are left linear.
- **`StandardScaler`** (z-score: subtract mean, divide by standard deviation) on all six features. K-Means uses **Euclidean distance**, so without standardisation `total_spent` (in the thousands) would completely swamp `num_orders` (single digits) — every cluster would effectively be a spend cut. Scaling lets each feature contribute comparably.

---

## 7. Model selection — choosing the number of clusters *k*

K-Means requires *k* to be fixed in advance, so we sweep **k = 2…10**, fitting each with `n_init=10` (ten random initialisations, best kept) and `random_state=42` (reproducibility). Because clustering is unsupervised — there are no ground-truth labels — we judge each *k* with **four internal diagnostics** rather than trusting any single one.

### 7.1 The metrics — what each means and why it is used

**Inertia (within-cluster sum of squares).** The sum of squared distances from every point to its assigned centroid. It measures **compactness** — lower is tighter. It *always* decreases as *k* grows (more centroids ⇒ shorter distances), so it can't be maximised directly; instead we use the **elbow method**, looking for the *k* where the marginal gain flattens. *Why:* it's the quantity K-Means actually minimises and reveals the point of diminishing returns.

**Silhouette score** *(range −1 to 1, higher better).* For each point, let `a` = mean distance to other points in its own cluster and `b` = mean distance to the nearest *other* cluster; the silhouette is `(b − a) / max(a, b)`. It simultaneously rewards **cohesion** (small `a`) and **separation** (large `b`). The score reported is the mean over all customers. *Why:* it's the most widely used internal index, needs no ground truth, and also gives a per-point diagnostic (see §8).

**Davies–Bouldin index** *(≥ 0, lower better).* For every pair of clusters it computes (spread\_i + spread\_j) / (distance between their centroids), then averages each cluster's worst-case ratio. Low values mean clusters are **compact and far apart**. *Why:* it is computed from cluster centroids and spreads directly (cheap) and provides an independent cross-check on silhouette, with different sensitivities.

**Calinski–Harabasz index** *(higher better).* Also called the *variance-ratio criterion*: the ratio of **between-cluster dispersion** to **within-cluster dispersion**, scaled by (N − k)/(k − 1). High values mean clusters are dense and well separated relative to their internal scatter. *Why:* a third independent lens — it tends to reward well-separated, dense solutions and complements the other two.

> Using three separation metrics plus the elbow guards against any single metric's bias. When they **agree**, confidence is high; when they **disagree**, the choice of *k* becomes an explicit trade-off between statistical separation and business interpretability.

### 7.2 Results

![Choosing k: elbow + silhouette](<graphs/Choosing k.png>)

The inertia curve (blue) bends early and then declines smoothly — a modest elbow around k = 2–3 with no dramatic later kink. The silhouette (orange) is **highest at k = 2 (0.357)** and falls away as *k* increases.

![Internal cluster-validation metrics vs. k](<graphs/Internal cluster-validation metrics vs. k.png>)

All three separation metrics point to the **same answer, k = 2**:

| k | Inertia | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ |
|---|---|---|---|---|
| **2** | 13,628 | **0.357** | **1.074** | **2,814.5** |
| 3 | 11,097 | 0.263 | 1.331 | 2,173.3 |
| 4 | 9,613 | 0.252 | 1.273 | 1,873.0 |
| 5 | 8,291 | 0.244 | 1.261 | 1,783.7 |
| … | … | … | … | … |

Silhouette is **maximised**, Davies–Bouldin is **minimised**, and Calinski–Harabasz is **maximised**, all at **k = 2**. This unanimous agreement is a strong, defensible signal that the cleanest structure in the data is a **two-way split**.

---

## 8. Final model & validation

We fit the final K-Means with **k = 2**.

![Silhouette plot for k=2](<graphs/Silhouette plot for k.png>)

The per-customer silhouette plot shows both clusters sitting **almost entirely above zero** (very few negative/misassigned points), with the mean at **0.357** (red dashed line). One cluster (~2,100 customers) is slightly larger than the other (~1,800). The widths and shapes are reasonably balanced — neither cluster is a sliver, and neither is dominated by negative silhouettes. This confirms k = 2 yields genuinely separated, sensibly sized groups rather than one giant blob plus an outlier pocket.

---

## 9. Cluster profiles & interpretation

Clusters are profiled on the **original (un-logged) scale** so the averages are human-readable:

| Metric | One-off / lapsed | Loyal / high-value |
|---|---|---|
| Customers | ~2,114 | ~1,794 |
| Avg orders | 1.5 | 7.4 |
| Avg lifetime spend | £388 | £3,350 |
| Avg basket value | £283 | £429 |
| Avg recency | 139 days (cold) | 36 days (recent) |
| Avg tenure | 34 days (short) | 244 days (long) |

![Feature distributions by cluster](<graphs/Feature distributions by cluster.png>)

The boxplots show clear separation on **frequency, monetary value, breadth, recency and tenure**: the loyal group orders far more often, spends far more, buys a wider range of products, has purchased recently, and has been active for most of the year. The two groups differ *least* on **average basket value** — i.e. the segmentation is driven by **how engaged a customer is, not how big each individual order is**. A casual buyer's single order can be as large as a loyal customer's typical order; what distinguishes them is repeat behaviour.

> **Note on cluster labels.** The numeric cluster index (0 / 1) that K-Means assigns is arbitrary and can swap between runs. In this report the segments are named by behaviour, not by index. After any re-run, re-read the profile table to confirm which index maps to which segment before quoting numbers.

![Customer segments in PCA space](<graphs/Customer segments in PCA space.png>)

To visualise the 6-dimensional clustering, **PCA** projects the scaled features onto their two highest-variance directions. The two principal components together retain **79.1% of the total variance** (PC1 60.9%, PC2 18.2%), so this 2-D picture is a faithful summary. The two clusters occupy **distinct regions along PC1** with only a modest overlap zone in the middle — consistent with the moderate (not extreme) silhouette of 0.357. PC1 effectively acts as an "engagement axis," separating the loyal cohort from the one-off cohort.

### 9.1 Revenue concentration — "few whales, many minnows" quantified

Rather than merely assert that a small loyal core drives most of the business, we measure it. Splitting total revenue by segment:

| Segment | Customers | Share of customers | Share of revenue |
|---|---|---|---|
| **Loyal / high-value** | 1,793 | 45.9% | **88.0%** |
| **One-off / lapsed** | 2,115 | 54.1% | 12.0% |

The loyal segment is under half the customer base yet generates **~88% of all revenue** (≈£5.99M of ≈£6.81M total). A customer-level **Lorenz curve** (cumulative revenue against cumulative customers, ranked by spend) sharpens the picture:

- Top **10%** of customers → **57.7%** of revenue
- Top **20%** of customers → **72.0%** of revenue
- Top **50%** of customers → **91.3%** of revenue

This is a textbook **Pareto concentration**: revenue rides overwhelmingly on a small, high-value core. It raises the stakes on **retention** — losing a handful of loyal customers costs far more than losing many casual ones — and confirms that the engagement-based split is also the revenue-relevant one.

---

## 10. Insights gained

1. **The business is overwhelmingly a UK operation (~89%).** Geographic scoping is justified and any segmentation strategy should be UK-first.
2. **Customers fall cleanly into two behavioural types.** Statistically the data supports two groups, unanimously, across four diagnostics — a loyal/high-value core and a much larger one-off/lapsed tail.
3. **Engagement, not basket size, is the dividing line.** The clusters differ most on frequency, recency and tenure and least on average order value. Retention levers (repeat purchasing) matter more than upselling a single basket.
4. **The customer base is "few whales, many minnows" — now quantified (§9.1).** Revenue is heavily concentrated: the loyal segment (45.9% of customers) generates **88.0% of revenue**, and the top 20% of customers alone account for ~72%. This Pareto concentration makes **retention** the dominant lever — the business's revenue rides on a small, high-value core.
5. **Strong Q4 seasonality.** Revenue and order volume ramp sharply into the pre-Christmas period — relevant for campaign timing.
6. **Clear action mapping:** *retain & reward* the loyal segment (loyalty perks, early access); *reactivate* the one-off/lapsed segment (win-back offers, re-engagement email) given its cold recency and short tenure.

---

## 11. Limitations & future work

Most of the limitations below are a function of **time** rather than method; given more of it, the following would be the priorities.

**Modelling**
- **Only K-Means was tried.** K-Means assumes roughly spherical, similarly-sized clusters and is sensitive to scaling and outliers. *Given more time:* validate the structure with **hierarchical/agglomerative clustering**, **DBSCAN/HDBSCAN** (density-based, finds non-spherical clusters and flags outliers), and a **Gaussian Mixture Model** (soft assignments), and compare whether they recover the same two groups.
- **k = 2 is statistically cleanest but coarse for marketing.** *Given more time:* deliberately profile a **k = 3–4** solution to see whether a useful "mid-tier" or "at-risk" persona emerges, accepting a small drop in cohesion for richer actionability.
- **No alternative to clustering.** *Given more time:* compare against a transparent **RFM quantile-scoring** scheme (e.g. score each of R/F/M 1–5), which is often more directly interpretable to a marketing team.

**Data & features**
- **Cancellation matching is heuristic.** The anti-join on `(customer, product, quantity, price)` is an approximation; it can miss partial returns or returns with differing prices. *Given more time:* reconcile cancellations to their original invoice via the invoice reference where available.
- **Outliers / wholesale buyers were not separately treated.** Extreme orders (e.g. quantity 80,995) may be B2B rather than retail. *Given more time:* detect and either segment or cap them explicitly.
- **Limited feature set.** *Given more time:* add inter-purchase interval, purchase regularity, product-category mix, return rate, and seasonality of each customer's activity.

**Validation & robustness**
- **Single 12-month snapshot.** Recency and tenure are sensitive to the snapshot date, and there is no out-of-time validation. *Given more time:* test **stability** by re-clustering on rolling windows or bootstrap resamples, and check that segment membership is consistent.
- **Preprocessing choices were not tuned.** *Given more time:* compare `StandardScaler` against `RobustScaler` (more outlier-tolerant), and assess sensitivity of the final segments.

---

*Reproducibility note:* all results use `random_state=42`. K-Means cluster **indices are arbitrary** and may differ between runs even when the segments themselves are identical; quote segment statistics from the profile table of the specific run that produced the figures.
