"""
equity_analysis.py  [FIXED v3]
------------------
Computes IDI-enhanced model performance across racial/ethnic subgroups.

FIXES APPLIED:
  BUG-2  (upstream): cohort_selection.py v2 now saves a 'race' column —
         this script will no longer silently return early.
  MINOR-17: Replaced hardcoded ax.set_xlim(0.55, 0.80) with dynamic
            limits derived from the actual data, preventing data clipping.
  BUG-7: Updated outcome column reference to 'hospital_mortality'
         (was 'mortality') to match cohort_selection.py v2.
  BUG-8 (v3 NEW): Removed n < 50 hard skip that silently excluded Hispanic
         (n = 26) and Asian (n = 24) subgroups from the output, making
         Supplementary Table 1 non-reproducible from v2 code.
         Replacement: a WARNING is printed for n < 50 subgroups but results
         are still computed and saved with a 'small_sample_warning' flag
         column. This matches the manuscript which reports these subgroups
         with explicit small-sample caveats (Supplementary Table 1 footnote).

Reproduces Table S1 and Figure 3 (forest plot) from:
  Collier AM, Shalhout SZ. JAMIA, 2026.

Subgroups: Asian, Black, Hispanic, White, Other  (alphabetical per JAMA style)
Metric: AUROC (95% CI via bootstrap, n=2000), ΔAUROC vs baseline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score
import os
import warnings

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

# BUG-3 FIX: lowered from 50 to 10 so Hispanic (n=26) and Asian (n=24)
# are included. Groups with 10 <= n < 50 get a small_sample_warning flag.
MIN_N_HARD  = 10   # absolute minimum — skip entirely below this
MIN_N_WARN  = 50   # threshold for small-sample warning flag

# FIX BUG-7: standardised outcome column name
OUTCOME = "hospital_mortality"


def bootstrap_auroc_ci(y, pred, n=N_BOOT, seed=SEED):
    """Bootstrap 95% CI for AUROC (percentile method)."""
    rng  = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            continue
        aucs.append(roc_auc_score(y[idx], pred[idx]))
    if len(aucs) == 0:
        return np.nan, np.nan
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def calibration_slope_intercept(y_true, y_pred, eps=1e-7):
    """
    Logistic regression of outcome ~ logit(predicted prob).
    Returns (slope, intercept). Slope = 1.0 is perfect calibration.
    """
    from sklearn.linear_model import LogisticRegression
    logit = np.log(
        np.clip(y_pred, eps, 1 - eps) / (1 - np.clip(y_pred, eps, 1 - eps))
    )
    lr = LogisticRegression(fit_intercept=True, max_iter=1000)
    lr.fit(logit.reshape(-1, 1), y_true)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def main():
    os.makedirs(os.path.join(RESULTS_DIR, "tables"),  exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)

    # Load predictions and cohort
    preds  = pd.read_csv(PREDS_PATH)
    cohort = pd.read_csv(COHORT_PATH)

    # FIX BUG-2 (upstream): cohort_selection.py v2 now saves 'race'.
    if "race" not in cohort.columns:
        raise RuntimeError(
            "'race' column missing from cohort data.\n"
            "Ensure you ran cohort_selection.py v3 which saves the 'race' column."
        )

    df = preds.merge(cohort[["stay_id", "race"]], on="stay_id", how="left")

    records = []
    for label, race_val in SUBGROUPS.items():
        sub = df[df["race"].str.contains(race_val, case=False, na=False)]
        n   = len(sub)

        # BUG-8 FIX: hard skip only below MIN_N_HARD (10), not 50
        if n < MIN_N_HARD:
            print(f"  {label}: n={n} — below hard minimum ({MIN_N_HARD}), skipping")
            continue

        # Warn but DO NOT skip for n < MIN_N_WARN (50)
        small_sample_warning = n < MIN_N_WARN
        if small_sample_warning:
            warnings.warn(
                f"  {label}: n={n} is below recommended minimum (n={MIN_N_WARN}). "
                f"Results are reported for equity transparency but should be "
                f"interpreted with extreme caution. CIs will be wide.",
                UserWarning, stacklevel=2
            )
            print(f"  ⚠ {label}: n={n} — small sample (< {MIN_N_WARN}), "
                  f"computing with warning flag")

        y  = sub[OUTCOME].values
        p0 = sub["baseline_prob"].values
        p1 = sub["idi_prob"].values

        if y.sum() == 0 or y.sum() == n:
            print(f"  {label}: all same outcome label, skipping")
            continue

        auc0 = roc_auc_score(y, p0)
        auc1 = roc_auc_score(y, p1)
        lo1, hi1 = bootstrap_auroc_ci(y, p1)

        # Calibration slope and intercept (added in v3 to match Supp Table 1)
        cal_slope0, cal_int0 = calibration_slope_intercept(y, p0)
        cal_slope1, cal_int1 = calibration_slope_intercept(y, p1)

        records.append({
            "Subgroup":              label,
            "N":                     n,
            "small_sample_warning":  small_sample_warning,
            "Mortality (%)":         f"{y.mean()*100:.1f}",
            "Baseline AUROC":        round(auc0, 3),
            "IDI AUROC":             round(auc1, 3),
            "95% CI Lower":          round(lo1, 3),
            "95% CI Upper":          round(hi1, 3),
            "ΔAUROC":                round(auc1 - auc0, 3),
            "Baseline Cal Slope":    round(cal_slope0, 2),
            "IDI Cal Slope":         round(cal_slope1, 2),
            "IDI Cal Intercept":     round(cal_int1, 2),
        })
        print(f"  {label:10s}  n={n:5,}  "
              f"Baseline={auc0:.3f}  IDI={auc1:.3f}  "
              f"Δ={auc1-auc0:+.3f}  "
              f"95% CI [{lo1:.3f}–{hi1:.3f}]  "
              f"CalSlope={cal_slope1:.2f}")

    if not records:
        print("No subgroups met the minimum size threshold.")
        return

    res = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_DIR, "tables", "equity_subgroup_auc.csv")
    res.to_csv(csv_path, index=False)
    print(f"\nSubgroup table saved → {csv_path}")

    # Flag small-sample rows in console summary
    small = res[res["small_sample_warning"]]
    if len(small):
        print(f"\n⚠  Small-sample subgroups (n < {MIN_N_WARN}) included with warning:")
        for _, row in small.iterrows():
            print(f"   {row['Subgroup']}: n={row['N']}  "
                  f"ΔAUROC={row['ΔAUROC']:+.3f}  CalSlope={row['IDI Cal Slope']:.2f}")

    # ── Forest plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, max(3, len(res) * 1.0)))
    y_pos  = list(range(len(res)))
    colors = ["darkorange" if row["small_sample_warning"] else "firebrick"
              for _, row in res.iterrows()]

    for idx, (_, row) in enumerate(res.iterrows()):
        ax.errorbar(
            row["IDI AUROC"], idx,
            xerr=[[row["IDI AUROC"] - row["95% CI Lower"]],
                  [row["95% CI Upper"] - row["IDI AUROC"]]],
            fmt="o", color=colors[idx], capsize=4, markersize=7,
        )
        ax.scatter(row["Baseline AUROC"], idx,
                   marker="D", color="steelblue", zorder=5)

    ax.axvline(res["IDI AUROC"].mean(), color="gray", ls="--", lw=0.8,
               label="Mean IDI AUROC")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(res["Subgroup"])
    ax.set_xlabel("AUROC")
    ax.set_title("IDI Model Performance by Racial/Ethnic Subgroup\n"
                 "(Orange = small sample n < 50, interpret with caution)")

    # Dynamic x-axis limits (FIX MINOR-17)
    all_aucs = pd.concat([res["Baseline AUROC"], res["95% CI Lower"],
                           res["95% CI Upper"]])
    margin = 0.06
    ax.set_xlim(all_aucs.min() - margin, all_aucs.max() + margin)

    # Legend
    normal_patch = mpatches.Patch(color="firebrick",
                                  label="IDI-Enhanced AUROC (95% CI, n ≥ 50)")
    small_patch  = mpatches.Patch(color="darkorange",
                                  label="IDI-Enhanced AUROC (95% CI, n < 50 ⚠)")
    blue_patch   = mpatches.Patch(color="steelblue", label="Baseline AUROC")
    ax.legend(handles=[normal_patch, small_patch, blue_patch], loc="lower right",
              fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "figures", "equity_forest.png")
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Forest plot saved → {fig_path}")


if __name__ == "__main__":
    main()
