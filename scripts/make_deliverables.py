#!/usr/bin/env python3
"""
Export decision-ready segmentation deliverables from the Mall Customers dataset.

Outputs (./artifacts):
- cluster_profiles.csv
- personas.csv
- outliers.csv (optional)
- metrics.json
- metadata.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RunConfig:
    k: int = 6
    random_state: int = 42
    eps_percentile: int = 90
    min_samples: int = 4
    features: tuple = ("Age", "Annual Income", "Spending Score", "Spend_Income_Ratio")


def _find_dataset(path: Optional[str]) -> Path:
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path))
    candidates.extend(
        [
            Path("data/raw/Mall_Customers.csv"),
            Path("Mall_Customers.csv"),
            Path("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv"),
        ]
    )
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Dataset not found. Place `Mall_Customers.csv` under `data/raw/` "
        "or pass --data PATH."
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Common Kaggle columns
    rename_map = {
        "Annual Income (k$)": "Annual Income",
        "Spending Score (1-100)": "Spending Score",
        "Spending Score (1–100)": "Spending Score",
    }
    df = df.rename(columns=rename_map)

    # Minimal validation
    required = {"Gender", "Age", "Annual Income", "Spending Score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Encode gender if present as strings
    if df["Gender"].dtype == object:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).fillna(df["Gender"])

    # Feature engineering
    eps = 1e-9
    df["Spend_Income_Ratio"] = df["Spending Score"] / (df["Annual Income"] + eps)
    return df


def _persona_rules(inc: float, spend: float) -> tuple[str, str]:
    if inc > 70 and spend > 60:
        return "VIP Big Spenders", "Exclusive offers, early access, VIP events"
    if inc > 70 and spend < 40:
        return "Affluent Savers", "Premium upsell nudges, concierge-style guidance"
    if inc < 40 and spend > 60:
        return "Budget Enthusiasts", "Value bundles, student/entry-level deals"
    if inc < 40 and spend < 30:
        return "Disengaged Adults", "Win-back campaigns, limited-time incentives"
    return "Balanced Shoppers", "Loyalty perks, bundles, cross-sell recommendations"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to Mall_Customers.csv")
    parser.add_argument("--k", type=int, default=6, help="Number of clusters for KMeans (default: 6)")
    parser.add_argument("--out", default="artifacts", help="Output directory (default: artifacts)")
    args = parser.parse_args()

    cfg = RunConfig(k=args.k)

    data_path = _find_dataset(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df = _normalize_columns(df)

    X = df[list(cfg.features)].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=cfg.k, random_state=cfg.random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df["Cluster"] = labels

    # Cluster profiles
    profile = (
        df.groupby("Cluster")[list(cfg.features)]
        .mean()
        .join(df.groupby("Cluster").size().rename("Count"))
        .round(3)
        .reset_index()
    )
    profile.to_csv(out_dir / "cluster_profiles.csv", index=False)

    # Personas
    personas = []
    for _, row in profile.iterrows():
        c = int(row["Cluster"])
        inc = float(row["Annual Income"])
        spend = float(row["Spending Score"])
        name, strat = _persona_rules(inc, spend)
        traits = f"Age~{row['Age']:.0f} | Income~{inc:.0f} | Spend~{spend:.0f}"
        personas.append({"Cluster": c, "Persona": name, "Traits": traits, "Strategy": strat, "Count": int(row["Count"])})
    persona_df = pd.DataFrame(personas).sort_values(["Cluster"])
    persona_df.to_csv(out_dir / "personas.csv", index=False)

    # Metrics
    metrics = {
        "silhouette": float(silhouette_score(X_scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
        "k": int(cfg.k),
        "n_rows": int(df.shape[0]),
        "n_features": int(X.shape[1]),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Optional DBSCAN outliers (k-distance heuristic)
    nn = NearestNeighbors(n_neighbors=cfg.min_samples).fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    distances = np.sort(distances[:, -1])
    eps = float(np.percentile(distances, cfg.eps_percentile))

    dbscan = DBSCAN(eps=eps, min_samples=cfg.min_samples)
    db_labels = dbscan.fit_predict(X_scaled)
    n_outliers = int(np.sum(db_labels == -1))
    if n_outliers > 0:
        outliers = df.loc[db_labels == -1, ["Gender", "Age", "Annual Income", "Spending Score", "Spend_Income_Ratio"]].copy()
        outliers.to_csv(out_dir / "outliers.csv", index=False)

    metadata = asdict(cfg) | {
        "dataset_path": str(data_path),
        "dbscan_eps": eps,
        "dbscan_outliers": n_outliers,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved artifacts to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
