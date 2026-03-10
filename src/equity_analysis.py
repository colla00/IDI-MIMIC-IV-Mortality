"""
equity_analysis.py  [FIXED v2]
------------------
Computes IDI-enhanced model performance across racial/ethnic subgroups.

FIXES APPLIED:
  BUG-2  (upstream): cohort_selection.py v2 now saves a 'race' column —
         this script will no longer silently return early.
  MINOR-17: Replaced hardcoded ax.set_xlim(0.55, 0.80) with dynamic
            limits derived from the actual data, preventing data clipping.
  BUG-7: Updated outcome column reference to 'hospital_mortality'
         (was 'mortality') to match cohort_selection.py v2.

Reproduces Table S1 and Figure 3 (forest plot) from:
  Collier AM, Shalhout SZ. JAMIA, 2026.

Subgroups: Asian, Black, Hispanic, White, Other  (alphabetical per JAMA style)
Metric: AUROC (95% CI via bootstrap, n=2000), ΔAUROC vs baseline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import os

RESULTS_DIR = "results"
PREDS_PATH  = os.path.join(RESULTS_DIR, "test_predictions.csv")
COHORT_PATH = "data/idi_features.csv"

# Alphabetical order per JAMA inclusive-language guidelines
SUBGROUPS = {
    "Asian":    "Asian",
    "Black":    "Black",
    "Hispanic": "Hispanic",
    "White":    "White",
    "Other":    "Other",
}

N_BOOT = 2000
SEED   = 42

# FIX BUG-7: standardised outcome column name
OUTCOME = "hospital_mortality"


def bootstrap_auroc_ci(y, pred, n=N_BOOT, seed=SEED):
    """Bootstrap 95% CI for AUROC."""
    rng  = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            continue
        aucs.append(roc_auc_score(y[idx], pred[idx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def main():
    os.makedirs(os.path.join(RESULTS_DIR, "tables"),  exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)

    # Load predictions and cohort
    preds  = pd.read_csv(PREDS_PATH)
    cohort = pd.read_csv(COHORT_PATH)

    # FIX BUG-2 (upstream): cohort_selection.py v2 now saves 'race'.
    # Keep this guard as a safety net.
    if "race" not in cohort.columns:
        raise RuntimeError(
            "'race' column missing from cohort data.\n"
            "Ensure you ran cohort_selection.py v2 which saves the 'race' column.\n"
            "See Bug-2 fix notes in cohort_selection.py."
        )

    df = preds.merge(cohort[["stay_id", "race"]], on="stay_id", how="left")

    records = []
    for label, race_val in SUBGROUPS.items():
        sub = df[df["race"].str.contains(race_val, case=False, na=False)]
        n   = len(sub)
        if n < 50:
            print(f"  {label}: n={n} — too small (< 50), skipping")
            continue
        y  = sub[OUTCOME].values
        p0 = sub["baseline_prob"].values
        p1 = sub["idi_prob"].values

        if y.sum() == 0 or y.sum() == n:
            print(f"  {label}: all same outcome label, skipping")
            continue

        auc0 = roc_auc_score(y, p0)
        auc1 = roc_auc_score(y, p1)
        lo1, hi1 = bootstrap_auroc_ci(y, p1)

        records.append({
            "Subgroup":       label,
            "N":              n,
            "Mortality (%)":  f"{y.mean()*100:.1f}",
            "Baseline AUROC": round(auc0, 3),
            "IDI AUROC":      round(auc1, 3),
            "95% CI Lower":   round(lo1, 3),
            "95% CI Upper":   round(hi1, 3),
            "ΔAUROC":         round(auc1 - auc0, 3),
        })
        print(f"  {label:10s}  n={n:5,}  "
              f"Baseline={auc0:.3f}  IDI={auc1:.3f}  "
              f"Δ={auc1-auc0:+.3f}  95% CI [{lo1:.3f}–{hi1:.3f}]")

    if not records:
        print("No subgroups met the minimum size threshold (n >= 50).")
        return

    res = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_DIR, "tables", "equity_subgroup_auc.csv")
    res.to_csv(csv_path, index=False)
    print(f"\nSubgroup table saved → {csv_path}")

    # ── Forest plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, max(3, len(res) * 0.9)))
    y_pos = range(len(res))

    ax.errorbar(
        res["IDI AUROC"], y_pos,
        xerr=[res["IDI AUROC"] - res["95% CI Lower"],
              res["95% CI Upper"] - res["IDI AUROC"]],
        fmt="o", color="firebrick", capsize=4, markersize=7,
        label="IDI-Enhanced AUROC (95% CI)",
    )
    ax.scatter(
        res["Baseline AUROC"], y_pos,
        marker="D", color="steelblue", zorder=5, label="Baseline AUROC",
    )
    ax.axvline(res["IDI AUROC"].mean(), color="gray", ls="--", lw=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(res["Subgroup"])
    ax.set_xlabel("AUROC")
    ax.set_title("IDI Model Performance by Racial/Ethnic Subgroup")

    # FIX MINOR-17: dynamic x-axis limits — no longer hardcoded to (0.55, 0.80)
    all_aucs = pd.concat([res["Baseline AUROC"], res["95% CI Lower"],
                           res["95% CI Upper"]])
    margin = 0.04
    ax.set_xlim(all_aucs.min() - margin, all_aucs.max() + margin)

    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "figures", "equity_forest.png")
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Forest plot saved → {fig_path}")


if __name__ == "__main__":
    main()
