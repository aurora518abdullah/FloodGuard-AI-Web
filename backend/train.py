
"""Train FloodGuard's experimental 24/48/72-hour flood classifiers.

Input:
    backend/data/all_stations.parquet (preferred) or all_stations.csv

Uses source-provided binary labels only.
This is an experimental model, not an official flood warning system.
"""

from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    brier_score_loss,
)

from app.config import DATA_PATHS, MODEL_DIR, MODEL_PATH, TARGETS


IDENTIFIERS = {
    "station_id",
    "station_name",
    "date",
    "basin",
    "river",
    "name",
}

FORBIDDEN_PARTS = (
    "flood_within_",
    "flood_bystor",
    "label_regime",
    "target_",
    "future_",
    "lead_",
)


def load_data() -> pd.DataFrame:
    """Load the available dataset."""

    for path in DATA_PATHS:
        if path.exists():
            print(f"Loading dataset: {path}")

            if path.suffix == ".parquet":
                return pd.read_parquet(path)

            return pd.read_csv(path)

    raise FileNotFoundError(
        "Training dataset not found. "
        "Put all_stations.parquet or all_stations.csv "
        "inside backend/data/."
    )


def select_features(df: pd.DataFrame) -> list[str]:
    """Select numeric predictors while excluding identifiers and labels."""

    feature_cols = []

    for col in df.select_dtypes(include=[np.number]).columns:
        low = col.lower()

        if col.lower() in IDENTIFIERS:
            continue

        if any(part in low for part in FORBIDDEN_PARTS):
            continue

        feature_cols.append(col)

    if not feature_cols:
        raise ValueError("No valid numeric predictor columns found.")

    print(f"\nSelected {len(feature_cols)} predictor features.")

    return feature_cols


def main() -> None:

    df = load_data()

    required = {"station_id", "date", *TARGETS.values()}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = (
        df.dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    feature_cols = select_features(df)

    bundle = {
        "models": {},
        "feature_columns": feature_cols,
        "cutoff_dates": {},
        "dataset_rows": int(len(df)),
        "dataset_first_date": str(df["date"].min().date()),
        "dataset_last_date": str(df["date"].max().date()),
        "metrics": {},
        "target_definition": (
            "Source dataset forward-looking flood labels. "
            "Experimental model, not an official warning."
        ),
    }

    print("\nDataset information:")
    print(f"Total rows: {len(df)}")
    print(f"First date: {df['date'].min()}")
    print(f"Last date: {df['date'].max()}")

    for horizon, target in TARGETS.items():

        print("\n" + "=" * 55)
        print(f"Training {horizon} model")
        print("=" * 55)

        y = pd.to_numeric(df[target], errors="coerce")

        # Only genuine binary labels are accepted.
        valid_label = y.isin([0, 1])

        valid_dates = np.array(
            sorted(df.loc[valid_label, "date"].unique())
        )

        if len(valid_dates) < 10:
            print(
                f"Skipping {horizon}: "
                "fewer than 10 valid label dates."
            )
            continue

        # Chronological 80/20 split based on valid label dates.
        cutoff_index = int(len(valid_dates) * 0.8)
        cutoff = pd.Timestamp(valid_dates[cutoff_index])

        # Three-day embargo for the 72-hour forecast horizon.
        train_mask = (
            valid_label
            & (df["date"] < cutoff - pd.Timedelta(days=3))
        )

        test_mask = (
            valid_label
            & (df["date"] >= cutoff)
        )

        y_train = y.loc[train_mask].astype(int)
        y_test = y.loc[test_mask].astype(int)

        print(f"Cutoff date: {cutoff.date()}")
        print(f"Training rows: {len(y_train)}")
        print(f"Testing rows: {len(y_test)}")

        print(
            "Training label counts:",
            y_train.value_counts().to_dict()
        )

        print(
            "Testing label counts:",
            y_test.value_counts().to_dict()
        )

        # Both classes must exist in both partitions.
        if y_train.nunique() < 2:
            print(
                f"Skipping {horizon}: "
                "training data does not contain both classes."
            )
            continue

        if y_test.nunique() < 2:
            print(
                f"Skipping {horizon}: "
                "testing data does not contain both classes."
            )
            continue

        x_train = (
            df.loc[train_mask, feature_cols]
            .replace([np.inf, -np.inf], np.nan)
        )

        x_test = (
            df.loc[test_mask, feature_cols]
            .replace([np.inf, -np.inf], np.nan)
        )

        # Balanced sample weights.
        counts = y_train.value_counts().to_dict()

        weights = y_train.map(
            {
                cls: len(y_train) / (2 * count)
                for cls, count in counts.items()
            }
        ).to_numpy()

        model = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=23,
            l2_regularization=1.0,
            random_state=42,
            early_stopping=True,
        )

        print(f"\nFitting {horizon} classifier...")

        model.fit(
            x_train,
            y_train,
            sample_weight=weights,
        )

        probabilities = model.predict_proba(x_test)[:, 1]

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        metrics = {
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "roc_auc": float(
                roc_auc_score(y_test, probabilities)
            ),
            "pr_auc": float(
                average_precision_score(y_test, probabilities)
            ),
            "precision_at_0_5": float(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            "recall_at_0_5": float(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            "brier_score": float(
                brier_score_loss(y_test, probabilities)
            ),
        }

        bundle["models"][horizon] = model

        bundle["cutoff_dates"][horizon] = str(
            cutoff.date()
        )

        bundle["metrics"][horizon] = metrics

        print(f"\n{horizon} evaluation metrics:")

        print(
            json.dumps(
                metrics,
                indent=2
            )
        )

    if not bundle["models"]:
        raise RuntimeError(
            "No horizon model could be trained. "
            "Inspect label coverage and chronological split."
        )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        bundle,
        MODEL_PATH
    )

    metrics_path = MODEL_DIR / "evaluation_metrics.json"

    metrics_path.write_text(
        json.dumps(
            bundle["metrics"],
            indent=2
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 55)
    print("TRAINING COMPLETED")
    print("=" * 55)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {metrics_path}")

    print(
        "\nIMPORTANT: These are experimental results. "
        "Review dataset provenance, temporal validity, "
        "and independent evaluation before real-world use."
    )


if __name__ == "__main__":
    main()
