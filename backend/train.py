"""Train FloodGuard's experimental 24/48/72-hour flood classifiers.

Input: backend/data/all_stations.parquet (preferred) or all_stations.csv.
The file must contain station_id, date, numeric predictors, and the target
columns flood_within_24h / 48h / 72h. No synthetic labels are generated.
"""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, brier_score_loss,
)
from app.config import DATA_PATHS, MODEL_DIR, MODEL_PATH, TARGETS

IDENTIFIERS = {"station_id", "station_name", "date", "basin", "river", "name"}
FORBIDDEN_PARTS = ("flood_within_", "label_regime", "target_", "future_", "lead_")


def load_data() -> pd.DataFrame:
    for path in DATA_PATHS:
        if path.exists():
            print(f"Loading {path} ...")
            return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    raise FileNotFoundError(
        "Training data missing. Put all_stations.parquet or all_stations.csv "
        "inside backend/data/. See README.md for the expected source and schema."
    )


def main() -> None:
    df = load_data()
    required = {"station_id", "date", *TARGETS.values()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    feature_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        low = col.lower()
        if col in IDENTIFIERS or any(part in low for part in FORBIDDEN_PARTS):
            continue
        feature_cols.append(col)
    if not feature_cols:
        raise ValueError("No numeric predictor columns were found in the dataset.")

    unique_dates = np.array(sorted(df["date"].dropna().unique()))
    if len(unique_dates) < 10:
        raise ValueError("At least 10 distinct dates are required for a time-based evaluation.")
    cutoff = unique_dates[max(1, int(len(unique_dates) * 0.8))]
    # 72-hour target horizon: leave a 3-day embargo before the test window
    # so training labels cannot look forward into the held-out period.
    train_mask = df["date"] < (pd.Timestamp(cutoff) - pd.Timedelta(days=3))
    test_mask = df["date"] >= cutoff

    bundle = {
        "models": {}, "feature_columns": feature_cols,
        "cutoff_date": str(pd.Timestamp(cutoff).date()),
        "dataset_rows": int(len(df)),
        "dataset_first_date": str(df.date.min().date()),
        "dataset_last_date": str(df.date.max().date()),
        "metrics": {},
        "target_definition": "Source dataset forward-looking flood labels; experimental, not an official warning.",
    }

    for horizon, target in TARGETS.items():
        y = pd.to_numeric(df[target], errors="coerce")
        valid_train = train_mask & y.notna()
        valid_test = test_mask & y.notna()
        y_train = y.loc[valid_train].astype(int)
        y_test = y.loc[valid_test].astype(int)
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            print(f"Skipping {horizon}: train/test labels need both classes.")
            continue
        xtr = df.loc[valid_train, feature_cols].replace([np.inf, -np.inf], np.nan)
        xte = df.loc[valid_test, feature_cols].replace([np.inf, -np.inf], np.nan)
        counts = y_train.value_counts().to_dict()
        weights = y_train.map({c: len(y_train) / (2 * n) for c, n in counts.items()}).to_numpy()
        model = HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.08, max_leaf_nodes=23,
            l2_regularization=1.0, random_state=42, early_stopping=True,
        )
        model.fit(xtr, y_train, sample_weight=weights)
        prob = model.predict_proba(xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics = {
            "train_rows": int(len(y_train)), "test_rows": int(len(y_test)),
            "positive_rate_test": float(y_test.mean()),
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "pr_auc": float(average_precision_score(y_test, prob)),
            "precision_at_0_5": float(precision_score(y_test, pred, zero_division=0)),
            "recall_at_0_5": float(recall_score(y_test, pred, zero_division=0)),
            "brier_score": float(brier_score_loss(y_test, prob)),
        }
        bundle["models"][horizon] = model
        bundle["metrics"][horizon] = metrics
        print(f"{horizon} metrics: {json.dumps(metrics, indent=2)}")

    if not bundle["models"]:
        raise RuntimeError("No horizon model could be trained. Check label coverage in the dataset.")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    (MODEL_DIR / "evaluation_metrics.json").write_text(
        json.dumps(bundle["metrics"], indent=2), encoding="utf-8"
    )
    print(f"Saved model bundle: {MODEL_PATH}")
    print(f"Saved evaluation metrics: {MODEL_DIR / 'evaluation_metrics.json'}")
    print("IMPORTANT: Review data provenance and metrics before presenting this as a forecasting system.")


if __name__ == "__main__":
    main()
