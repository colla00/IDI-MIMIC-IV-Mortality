"""
model.py
--------
Trains baseline and IDI-enhanced logistic regression models with temporal
validation (training: 2008-2018; test: 2019).

Models:
  Baseline:     age, sex_binary, icu_los_days
  IDI-Enhanced: baseline features + IDI features surviving leakage filter

Both models use L2 regularisation (C=1.0) via scikit-learn LogisticRegression.
Features are z-score standardised on the training set after median imputation.
SimpleImputer(strategy='median') is applied as the first pipeline step to
handle NaN IDI values that can arise from stays with sparse chartevent data.

Leakage filter: IDI features with |Pearson r| > 0.30 with ICU LOS are removed
before training to prevent reverse-causal leakage. Non-survivors have longer
ICU stays (median 71.3 h vs 53.1 h), so features correlated with LOS are
post-outcome proxies rather than true predictors.

Reference:
  Collier AM, Shalhout SZ. Development and Validation of the Intensive
  Documentation Index for ICU Mortality Prediction. JAMIA, 2026.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os

DATA_PATH   = "data/idi_features.csv"
RESULTS_DIR = "results"

BASELINE_FEATURES = ["age", "sex_binary", "icu_los_days"]

IDI_FEATURES_ALL = [
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

OUTCOME = "hospital_mortality"


# ── Leakage filter ────────────────────────────────────────────────────────────
def apply_leakage_filter(df: pd.DataFrame,
                          feature_cols: list,
                          los_col: str = "icu_los_days",
                          threshold: float = 0.30) -> list:
    """
    Remove IDI features with |Pearson r| > threshold with ICU LOS.
    Filter is computed on training data only to prevent data leakage.
    Returns the list of feature names that pass the filter.
    """
    kept, dropped = [], []
    for col in feature_cols:
        valid = df[[col, los_col]].dropna()
        if len(valid) < 10:
            kept.append(col)
            continue
        r = valid.corr().iloc[0, 1]
        if abs(r) <= threshold:
            kept.append(col)
        else:
            dropped.append((col, round(r, 4)))

    print(f"\nLeakage filter (|r| > {threshold} with {los_col}):")
    print(f"  Input features : {len(feature_cols)}")
    print(f"  Kept           : {len(kept)}")
    if dropped:
        print(f"  Dropped        : {len(dropped)}")
        for col, r in sorted(dropped, key=lambda x: abs(x[1]), reverse=True):
            print(f"    {col}: r = {r}")
    return kept


# ── Data preparation ──────────────────────────────────────────────────────────
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Encode sex, compute icu_los_days, extract admit_year."""
    df = df.copy()
    df["sex_binary"]   = (df["gender"].str.upper() == "M").astype(int)
    df["icu_los_days"] = df["icu_los_hours"] / 24.0
    df["admit_year"]   = pd.to_datetime(df["intime"]).dt.year
    return df


def temporal_split(df: pd.DataFrame, test_year: int = 2019):
    """
    Temporal validation split:
      Train: admissions strictly before test_year
      Test:  admissions in test_year
    """
    train = df[df["admit_year"] <  test_year].copy()
    test  = df[df["admit_year"] == test_year].copy()
    print(f"Train: {len(train):,}  "
          f"(years {train['admit_year'].min()}-{train['admit_year'].max()})  "
          f"mortality {train[OUTCOME].mean() * 100:.1f}%")
    print(f"Test:  {len(test):,}  "
          f"(year {test_year})  "
          f"mortality {test[OUTCOME].mean() * 100:.1f}%")
    return train, test


# ── Model pipeline ────────────────────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    """
    Three-step pipeline: median imputation -> z-score scaling -> logistic regression.
    Imputation uses training-set medians to prevent test-set leakage.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(C=1.0, max_iter=1000,
                                       solver="lbfgs", random_state=42)),
    ])


def train_and_save(train: pd.DataFrame,
                   test:  pd.DataFrame,
                   features: list,
                   model_name: str):
    X_train = train[features].values
    y_train = train[OUTCOME].values
    X_test  = test[features].values
    y_test  = test[OUTCOME].values

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    train_proba = pipe.predict_proba(X_train)[:, 1]
    test_proba  = pipe.predict_proba(X_test)[:, 1]

    out_train              = train[["stay_id", OUTCOME]].copy()
    out_train["pred_prob"] = train_proba
    out_train["split"]     = "train"

    out_test              = test[["stay_id", OUTCOME]].copy()
    out_test["pred_prob"] = test_proba
    out_test["split"]     = "test"

    out      = pd.concat([out_train, out_test], ignore_index=True)
    out_path = os.path.join(RESULTS_DIR, f"predictions_{model_name}.csv")
    out.to_csv(out_path, index=False)
    print(f"  Predictions saved -> {out_path}")

    model_path = os.path.join(RESULTS_DIR, f"model_{model_name}.pkl")
    joblib.dump(pipe, model_path)
    print(f"  Model saved      -> {model_path}")

    return pipe, test_proba, y_test


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "tables"),  exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["intime"])
    df = prepare_data(df)
    print(f"Dataset: {len(df):,} stays | "
          f"mortality {df[OUTCOME].mean() * 100:.2f}%\n")

    train, test = temporal_split(df, test_year=2019)

    # Apply leakage filter on training data only
    idi_features_kept = apply_leakage_filter(
        train, IDI_FEATURES_ALL, los_col="icu_los_days", threshold=0.30
    )
    print(f"\nIDI features after leakage filter: {idi_features_kept}")

    print("\n--- Training Baseline Model ---")
    baseline_pipe, baseline_proba, y_test = train_and_save(
        train, test, BASELINE_FEATURES, "baseline"
    )

    print("\n--- Training IDI-Enhanced Model ---")
    idi_features_full = BASELINE_FEATURES + idi_features_kept
    idi_pipe, idi_proba, _ = train_and_save(
        train, test, idi_features_full, "idi_enhanced"
    )

    test_out                  = test[["stay_id", OUTCOME]].copy()
    test_out["baseline_prob"] = baseline_proba
    test_out["idi_prob"]      = idi_proba
    combined_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    test_out.to_csv(combined_path, index=False)
    print(f"\nCombined test predictions -> {combined_path}")
    print("Done. Run metrics.py to compute AUROC, calibration, and Brier score.")


if __name__ == "__main__":
    main()
