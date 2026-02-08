# Artifacts

This folder is created by the notebook / scripts and contains exportable deliverables such as:

- `cluster_profiles.csv` — per-cluster aggregates (means + counts)
- `personas.csv` — persona name, traits, and recommended actions per cluster
- `outliers.csv` — DBSCAN outliers (if detected)
- `metrics.json` — clustering quality metrics (Silhouette / DB / CH)
- `metadata.json` — run configuration (features, k, random_state, etc.)

Artifacts are not committed by default.
