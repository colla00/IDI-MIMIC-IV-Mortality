"""
model.py
--------
Trains baseline and IDI-enhanced logistic regression models with temporal
validation (training: 2008-2018; test: 2019).

Models:
  Baseline:     age, sex (binary), icu_los_days
  IDI-Enhanced: baseline features + 9 IDI features

Both models use L2 regularization (C=1.0) via scikit-learn LogisticRegression.
All features are z-score standardized on the training set.

Reference:
  Collier AM, Shalhout SZ. JAMIA, 2026.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import os

DATA_PATH   = "data/idi_features.csv"
RESULTS_DIR = "results"

BASELINE_FEATURES = ["age", "sex_binary", "icu_los_days"]

IDI_FEATURES = [
    "idi_events_24h",
    "idi_events_per_hour",
    "idi_cv_interevent",
    "idi_std_interevent_min",
    "idi_mean_interevent_min",
    "idi_max_gap_min",
    "idi_gap_count_60m",
    "idi_gap_count_120m",
    "idi_burstiness",
]


def prepare_data(df):
    """Encode sex and compute icu_los_days."""
    df = df.copy()
    df["sex_binary"]   = (df["gender"].str.upper() == "M").astype(int)
    df["icu_los_days"] = df["icu_los_hours"] / 24.0
    df["admit_year"]   = pd.to_datetime(df["intime"]).dt.year
    return df


def temporal_split(df, test_year=2019):
    """
    Temporal validation split:
      Train: admissions up to and including (test_year - 1)
      Test:  admissions in test_year
    """
    train = df[df["admit_year"] < test_year].copy()
    test  = df[df["admit_year"] == test_year].copy()
    print(f"Train: {len(train):,} (years {train['admit_year'].min()}–"
          f"{train['admit_year'].max()}) | "
          f"mortality {train['mortality'].mean()*100:.1f}%")
    print(f"Test:  {len(test):,} (year {test_year}) | "
          f"mortality {test['mortality'].mean()*100:.1f}%")
    return train, test


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=1.0, max_iter=1000,
                                      solver="lbfgs", random_state=42))
    ])


def train_and_save(train, test, features, model_name):
    X_train = train[features].values
    y_train = train["mortality"].values
    X_test  = test[features].values
    y_test  = test["mortality"].values

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # Save predicted probabilities
    train_proba = pipe.predict_proba(X_train)[:, 1]
    test_proba  = pipe.predict_proba(X_test)[:, 1]

    out_train = train[["stay_id", "mortality"]].copy()
    out_train["pred_prob"] = train_proba
    out_train["split"] = "train"

    out_test = test[["stay_id", "mortality"]].copy()
    out_test["pred_prob"] = test_proba
    out_test["split"] = "test"

    out = pd.concat([out_train, out_test], ignore_index=True)
    out_path = os.path.join(RESULTS_DIR, f"predictions_{model_name}.csv")
    out.to_csv(out_path, index=False)
    print(f"  Predictions saved → {out_path}")

    # Save model
    model_path = os.path.join(RESULTS_DIR, f"model_{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"  Model saved      → {model_path}")

    return pipe, test_proba, y_test


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["intime"])
    df = prepare_data(df)
    print(f"Dataset: {len(df):,} stays | "
          f"mortality {df['mortality'].mean()*100:.2f}%\n")

    train, test = temporal_split(df, test_year=2019)

    print("\n--- Training Baseline Model ---")
    baseline_pipe, baseline_proba, y_test = train_and_save(
        train, test, BASELINE_FEATURES, "baseline")

    print("\n--- Training IDI-Enhanced Model ---")
    idi_features_all = BASELINE_FEATURES + IDI_FEATURES
    idi_pipe, idi_proba, _ = train_and_save(
        train, test, idi_features_all, "idi_enhanced")

    # Save combined test predictions for metrics.py
    test_out = test[["stay_id", "mortality"]].copy()
    test_out["baseline_prob"] = baseline_proba
    test_out["idi_prob"]      = idi_proba
    test_out.to_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)
    print(f"\nCombined test predictions saved → "
          f"{os.path.join(RESULTS_DIR, 'test_predictions.csv')}")
    print("\nDone. Run metrics.py to compute AUROC, calibration, and Brier score.")


if __name__ == "__main__":
    main()
