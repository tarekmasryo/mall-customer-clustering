# 🛍️ Mall Customer Segmentation — KMeans, PCA & DBSCAN

Unsupervised segmentation with **cluster profiles** + **auto-generated personas** that translate directly into marketing/CRM actions.

**Full case study:** `CASE_STUDY.md`

---

## ✅ What you get
- Leak-safe, reproducible clustering workflow (scaling → k-selection → clustering)
- Multiple algorithms for comparison (**KMeans**, Agglomerative, **GMM**)
- PCA visualizations for separation and explainability
- DBSCAN-based outlier flagging (optional)
- Exportable deliverables under `./artifacts/`:
  - `cluster_profiles.csv`
  - `personas.csv`
  - `outliers.csv` (if detected)
  - `metrics.json`
  - `metadata.json`

---

## 📂 Dataset
- **Size:** 200 customers
- **Typical columns:** `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1–100)`

### Local (recommended)
1) Download the dataset CSV (commonly named `Mall_Customers.csv`).
2) Place it here:

`./data/raw/Mall_Customers.csv`

Dataset files are not included in this repository.

### Kaggle
The notebook also supports the common Kaggle path:
`/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv`

---

## ⚡ Quick Start

```bash
git clone https://github.com/tarekmasryo/mall-customer-clustering
cd mall-customer-clustering
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook mall-customer-segmentation-eda-clustering.ipynb
```

---

## 📦 Export deliverables (no notebook UI needed)

After placing the dataset CSV under `data/raw/`:

```bash
python scripts/make_deliverables.py
```

Artifacts will be saved under `./artifacts/`.

---

## 🧠 Method (high level)
1) **Cleaning & prep** (rename columns, encode gender when needed)
2) **Feature engineering**: `Spend_Income_Ratio = Spending Score / (Annual Income + ε)`
3) **Scaling**: `StandardScaler`
4) **k-selection**: Elbow + Silhouette + Calinski–Harabasz, then visual validation via PCA
5) **Clustering**: KMeans (primary) + comparisons (Agglomerative, GMM)
6) **Personas**: generated from cluster profiles (income/spend rules) to avoid label mix-ups
7) **Outliers**: optional DBSCAN flagging via k-distance heuristic

---

## ⚠️ Limitations
- Small dataset and limited behavioral features (no RFM, purchase history, channels).
- KMeans is sensitive to scaling/initialization; DBSCAN depends on `eps` and `min_samples`.
- Treat this as a segmentation template — validate on richer data before production rollout.

---

## 📜 License
MIT (code). Dataset remains under its original terms.
