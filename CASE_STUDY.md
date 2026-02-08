# Case Study — Mall Customer Segmentation (Decision-Ready Clustering)

## Overview
This project converts a small “mall customers” dataset into **actionable customer personas** using unsupervised learning.  
The goal is not to showcase clustering plots — it is to produce **segment definitions** that can be used by growth/CRM teams.

## Problem
Teams often have limited customer signals (age, income proxy, and an engagement proxy such as “spending score”).  
The key questions are:
- Which customers behave like **high-value VIPs**?
- Which customers are **price-sensitive** but engaged?
- Which segments need **retention** vs **win-back** strategies?

## Data
Typical columns:
- `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1–100)`

A lightweight engineered feature is added:
- `Spend_Income_Ratio = Spending Score / (Annual Income + ε)`  
to isolate “high spend relative to income” behavior.

## Approach
1) **Preprocessing**
- Normalize column names, encode `Gender` when needed, and engineer `Spend_Income_Ratio`.
- Scale features using `StandardScaler`.

2) **Clustering**
- **KMeans** is used as the primary algorithm for stable, interpretable clusters on scaled numeric features.
- Cluster count is typically selected via multiple signals (Elbow + Silhouette + Calinski–Harabasz), then validated visually.

3) **Explainability**
- Build a cluster profile table (means + counts).
- Generate personas from profile rules based on income/spend levels to avoid label mix-ups when cluster IDs change.

4) **Outlier detection (optional)**
- Use **DBSCAN** with a k-distance percentile heuristic to flag unusual shoppers (possible VIP anomalies or noisy points).

## Deliverables
This repo supports exportable outputs under `./artifacts/`:
- `cluster_profiles.csv` — per-cluster aggregates
- `personas.csv` — persona name, traits, and recommended actions
- `outliers.csv` — DBSCAN outliers (if detected)
- `metrics.json` — clustering quality metrics
- `metadata.json` — run configuration and thresholds

You can generate these deliverables via:
- Notebook execution, or
- `python scripts/make_deliverables.py`

## Limitations
- Small dataset (200 rows) and limited behavioral features (no RFM, purchase history, channel mix).
- KMeans sensitivity to scaling/initialization; DBSCAN sensitivity to `eps` and `min_samples`.
- “Spending Score” is a proxy; real systems should validate segments against outcomes (conversion, AOV, retention).

## Next Steps
- Enrich with RFM features and purchase history.
- Add stability checks (repeated runs / bootstrapped ARI).
- Tie segments to KPIs and validate actions via experiments (A/B testing).
