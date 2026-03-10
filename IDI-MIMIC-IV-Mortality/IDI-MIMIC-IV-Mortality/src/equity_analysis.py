"""
equity_analysis.py
------------------
Computes IDI-enhanced model performance across racial/ethnic subgroups.

Reproduces Table S1 and Figure 3 (forest plot) from:
  Collier AM, Shalhout SZ. JAMIA, 2026.

Subgroups: White, Black, Hispanic, Asian, Other
Metric: AUROC (95% CI via bootstrap), ΔAUROC vs baseline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score
import os

RESULTS_DIR = "results"
PREDS_PATH  = os.path.join(RESULTS_DIR, "test_predictions.csv")
# Requires the original cohort joined with predictions
COHORT_PATH = "data/idi_features.csv"

SUBGROUPS = {
    "White":    "White",
    "Black":    "Black",
    "Hispanic": "Hispanic",
    "Asian":    "Asian",
    "Other":    "Other",
}

N_BOOT = 2000
SEED   = 42


def bootstrap_auroc_ci(y, pred, n=N_BOOT, seed=SEED):
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

    # Load predictions + cohort race data
    preds  = pd.read_csv(PREDS_PATH)
    cohort = pd.read_csv(COHORT_PATH)

    # Cohort needs a 'race' column — derived from MIMIC-IV admissions table
    # (add to cohort_selection.py if needed)
    if "race" not in cohort.columns:
        print("WARNING: 'race' column not found in cohort data.\n"
              "  Add race from MIMIC-IV admissions table in cohort_selection.py")
        return

    df = preds.merge(cohort[["stay_id", "race"]], on="stay_id", how="left")

    records = []
    for label, race_val in SUBGROUPS.items():
        sub = df[df["race"].str.contains(race_val, case=False, na=False)]
        if len(sub) < 50:
            print(f"  {label}: n={len(sub)} — too small, skipping")
            continue
        y  = sub["mortality"].values
        p0 = sub["baseline_prob"].values
        p1 = sub["idi_prob"].values
        auc0 = roc_auc_score(y, p0)
        auc1 = roc_auc_score(y, p1)
        lo1, hi1 = bootstrap_auroc_ci(y, p1)
        records.append({
            "Subgroup":        label,
            "N":               len(sub),
            "Mortality (%)":   f"{y.mean()*100:.1f}",
            "Baseline AUROC":  round(auc0, 3),
            "IDI AUROC":       round(auc1, 3),
            "95% CI Lower":    round(lo1, 3),
            "95% CI Upper":    round(hi1, 3),
            "ΔAUROC":          round(auc1 - auc0, 3),
        })
        print(f"  {label:10s}  n={len(sub):5,}  "
              f"Baseline={auc0:.3f}  IDI={auc1:.3f}  "
              f"Δ={auc1-auc0:+.3f}  95% CI [{lo1:.3f}–{hi1:.3f}]")

    res = pd.DataFrame(records)
    res.to_csv(os.path.join(RESULTS_DIR, "tables", "equity_subgroup_auc.csv"),
               index=False)
    print(f"\nSubgroup table saved → "
          f"{os.path.join(RESULTS_DIR, 'tables', 'equity_subgroup_auc.csv')}")

    # ── Forest plot ──
    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = range(len(res))
    ax.errorbar(res["IDI AUROC"], y_pos,
                xerr=[res["IDI AUROC"] - res["95% CI Lower"],
                      res["95% CI Upper"] - res["IDI AUROC"]],
                fmt="o", color="firebrick", capsize=4, markersize=7,
                label="IDI-Enhanced AUROC (95% CI)")
    ax.scatter(res["Baseline AUROC"], y_pos, marker="D",
               color="steelblue", zorder=5, label="Baseline AUROC")
    ax.axvline(res["IDI AUROC"].mean(), color="gray", ls="--", lw=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(res["Subgroup"])
    ax.set_xlabel("AUROC")
    ax.set_title("IDI Model Performance by Racial/Ethnic Subgroup")
    ax.set_xlim(0.55, 0.80)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "figures", "equity_forest.png"), dpi=150)
    plt.close()
    print("Forest plot saved → results/figures/equity_forest.png")


if __name__ == "__main__":
    main()
