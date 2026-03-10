"""
metrics.py
----------
Computes and reports all performance metrics from test_predictions.csv:
  - AUROC with 95% CI (DeLong method)
  - AUPRC
  - Brier score
  - Calibration slope (logistic regression of outcome on logit(predicted prob))
  - Hosmer-Lemeshow chi-squared test
  - Sensitivity at 80% specificity
  - Specificity at 80% sensitivity
  - PPV and NPV at optimal threshold
  - DeLong test for AUROC difference (baseline vs IDI)

Saves:
  results/tables/performance_metrics.csv
  results/figures/roc_curve.png
  results/figures/calibration_plot.png

Reference:
  Collier AM, Shalhout SZ. JAMIA, 2026.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                              average_precision_score, brier_score_loss)
from sklearn.linear_model import LogisticRegression
from scipy import stats
import os

RESULTS_DIR = "results"
PREDS_PATH  = os.path.join(RESULTS_DIR, "test_predictions.csv")


# ── DeLong AUROC confidence interval ─────────────────────────────────────────
def delong_roc_variance(ground_truth, predictions):
    """Variance of AUROC via DeLong et al. 1988."""
    order  = np.argsort(-predictions)
    label  = ground_truth[order]
    n1 = int(label.sum())
    n0 = len(label) - n1
    tp = np.cumsum(label)
    fp = np.cumsum(1 - label)
    tp_fp = np.column_stack([tp, fp])
    auc   = roc_auc_score(ground_truth, predictions)
    v10   = np.zeros(n1)
    v01   = np.zeros(n0)
    pos_idx = np.where(label == 1)[0]
    neg_idx = np.where(label == 0)[0]
    for i, pi in enumerate(pos_idx):
        v10[i] = (fp[pi] / n0) if fp[pi] > 0 else 0
    for j, nj in enumerate(neg_idx):
        v01[j] = 1 - (tp[nj] / n1) if tp[nj] > 0 else 1
    s10 = np.var(v10, ddof=1) / n1
    s01 = np.var(v01, ddof=1) / n0
    return auc, s10 + s01


def auroc_ci(y_true, y_pred, alpha=0.05):
    auc, var = delong_roc_variance(np.array(y_true), np.array(y_pred))
    se   = np.sqrt(var)
    z    = stats.norm.ppf(1 - alpha / 2)
    return auc, auc - z * se, auc + z * se


def delong_test(y_true, pred1, pred2):
    """Two-sided DeLong test for difference in AUROCs."""
    auc1, var1 = delong_roc_variance(np.array(y_true), np.array(pred1))
    auc2, var2 = delong_roc_variance(np.array(y_true), np.array(pred2))
    z = (auc2 - auc1) / np.sqrt(var1 + var2)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, auc2 - auc1, p


# ── Calibration slope ─────────────────────────────────────────────────────────
def calibration_slope(y_true, y_pred):
    """Logistic regression slope of outcome ~ logit(predicted prob)."""
    eps   = 1e-7
    logit = np.log(np.clip(y_pred, eps, 1 - eps) /
                   (1 - np.clip(y_pred, eps, 1 - eps)))
    lr    = LogisticRegression(fit_intercept=True, max_iter=1000)
    lr.fit(logit.reshape(-1, 1), y_true)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


# ── Hosmer-Lemeshow ────────────────────────────────────────────────────────────
def hosmer_lemeshow(y_true, y_pred, g=10):
    df  = pd.DataFrame({"y": y_true, "p": y_pred})
    df["decile"] = pd.qcut(df["p"], g, duplicates="drop", labels=False)
    observed = df.groupby("decile")["y"].sum()
    expected = df.groupby("decile")["p"].sum()
    n        = df.groupby("decile").size()
    hl = ((observed - expected) ** 2 / (expected * (1 - expected / n))).sum()
    p  = 1 - stats.chi2.cdf(hl, df=g - 2)
    return hl, p


# ── Sensitivity / specificity at fixed operating point ────────────────────────
def sens_at_spec(y_true, y_pred, target_spec=0.80):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    spec = 1 - fpr
    idx  = np.argmin(np.abs(spec - target_spec))
    return tpr[idx]


def spec_at_sens(y_true, y_pred, target_sens=0.80):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    idx = np.argmin(np.abs(tpr - target_sens))
    return (1 - fpr)[idx]


def ppv_npv(y_true, y_pred):
    threshold = np.percentile(y_pred, 100 * (1 - y_true.mean()))
    pred_pos  = y_pred >= threshold
    tp = ((pred_pos == 1) & (y_true == 1)).sum()
    fp = ((pred_pos == 1) & (y_true == 0)).sum()
    fn = ((pred_pos == 0) & (y_true == 1)).sum()
    tn = ((pred_pos == 0) & (y_true == 0)).sum()
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    return ppv, npv


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.join(RESULTS_DIR, "tables"),  exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)

    df = pd.read_csv(PREDS_PATH)
    y  = df["mortality"].values
    p0 = df["baseline_prob"].values
    p1 = df["idi_prob"].values

    print(f"Test set: {len(y):,} | Deaths: {y.sum():,} ({y.mean()*100:.2f}%)\n")

    # ── AUROC ──
    auc0, lo0, hi0 = auroc_ci(y, p0)
    auc1, lo1, hi1 = auroc_ci(y, p1)
    _, _, delta_auc, p_delong = delong_test(y, p0, p1)

    # ── Calibration ──
    slope0, intercept0 = calibration_slope(y, p0)
    slope1, intercept1 = calibration_slope(y, p1)
    hl0, phl0 = hosmer_lemeshow(y, p0)
    hl1, phl1 = hosmer_lemeshow(y, p1)

    # ── Other metrics ──
    brier0 = brier_score_loss(y, p0)
    brier1 = brier_score_loss(y, p1)
    auprc0 = average_precision_score(y, p0)
    auprc1 = average_precision_score(y, p1)
    sens0  = sens_at_spec(y, p0); sens1  = sens_at_spec(y, p1)
    spec0  = spec_at_sens(y, p0); spec1  = spec_at_sens(y, p1)
    ppv0, npv0 = ppv_npv(y, p0);  ppv1, npv1 = ppv_npv(y, p1)

    # ── Print results ──
    print("=" * 55)
    print(f"{'Metric':<30} {'Baseline':>10} {'IDI':>10}")
    print("=" * 55)
    print(f"{'AUROC (95% CI)':<30} {auc0:.4f}    {auc1:.4f}")
    print(f"{'  lower CI':<30} {lo0:.4f}    {lo1:.4f}")
    print(f"{'  upper CI':<30} {hi0:.4f}    {hi1:.4f}")
    print(f"{'Delta AUROC':<30} {delta_auc:+.4f}  (p={p_delong:.4f})")
    print(f"{'AUPRC':<30} {auprc0:.4f}    {auprc1:.4f}")
    print(f"{'Brier Score':<30} {brier0:.4f}    {brier1:.4f}")
    print(f"{'Calibration Slope':<30} {slope0:.4f}    {slope1:.4f}")
    print(f"{'Calibration Intercept':<30} {intercept0:.4f}    {intercept1:.4f}")
    print(f"{'H-L chi2 (p)':<30} {hl0:.1f} ({phl0:.2f})  {hl1:.1f} ({phl1:.2f})")
    print(f"{'Sensitivity @ 80% Spec':<30} {sens0:.3f}    {sens1:.3f}")
    print(f"{'Specificity @ 80% Sens':<30} {spec0:.3f}    {spec1:.3f}")
    print(f"{'PPV':<30} {ppv0:.2f}      {ppv1:.2f}")
    print(f"{'NPV':<30} {npv0:.2f}      {npv1:.2f}")
    print("=" * 55)

    # ── Save table ──
    results = {
        "Metric": ["AUROC", "AUROC 95% CI Lower", "AUROC 95% CI Upper",
                   "ΔAUROC", "p (DeLong)", "AUPRC", "Brier Score",
                   "Calibration Slope", "H-L chi2", "H-L p",
                   "Sensitivity @ 80% Spec", "Specificity @ 80% Sens",
                   "PPV", "NPV"],
        "Baseline": [auc0, lo0, hi0, "", "", auprc0, brier0, slope0,
                     hl0, phl0, sens0, spec0, ppv0, npv0],
        "IDI-Enhanced": [auc1, lo1, hi1, delta_auc, p_delong,
                         auprc1, brier1, slope1, hl1, phl1,
                         sens1, spec1, ppv1, npv1],
    }
    pd.DataFrame(results).to_csv(
        os.path.join(RESULTS_DIR, "tables", "performance_metrics.csv"),
        index=False)
    print("\nMetrics saved → results/tables/performance_metrics.csv")

    # ── ROC curve ──
    fpr0, tpr0, _ = roc_curve(y, p0)
    fpr1, tpr1, _ = roc_curve(y, p1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr0, tpr0, label=f"Baseline  AUC={auc0:.3f}", color="steelblue")
    ax.plot(fpr1, tpr1, label=f"IDI       AUC={auc1:.3f}", color="firebrick")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("1 – Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("ROC Curves — Temporal Validation (2019 Test Set)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "figures", "roc_curve.png"), dpi=150)
    plt.close()
    print("ROC curve saved → results/figures/roc_curve.png")

    # ── Calibration plot ──
    fig, ax = plt.subplots(figsize=(6, 6))
    for probs, label, color in [(p0, "Baseline", "steelblue"),
                                  (p1, "IDI", "firebrick")]:
        df_cal = pd.DataFrame({"y": y, "p": probs})
        df_cal["bin"] = pd.qcut(df_cal["p"], 10, duplicates="drop", labels=False)
        cal = df_cal.groupby("bin").agg(mean_pred=("p", "mean"),
                                         obs_rate=("y", "mean")).reset_index()
        ax.plot(cal["mean_pred"], cal["obs_rate"], "o-", label=label, color=color)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Mortality Rate")
    ax.set_title("Calibration Plot — Temporal Validation (2019 Test Set)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "figures", "calibration_plot.png"), dpi=150)
    plt.close()
    print("Calibration plot saved → results/figures/calibration_plot.png")


if __name__ == "__main__":
    main()
