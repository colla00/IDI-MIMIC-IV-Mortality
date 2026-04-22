# Intensive Documentation Index (IDI) — MIMIC-IV Development & Validation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/1177377775.svg)](https://doi.org/10.5281/zenodo.18943865)

**Paper:** Development and Validation of the Intensive Documentation Index for ICU Mortality Prediction
**Journal:** Journal of the American Medical Informatics Association (JAMIA, under review, 2026)
**Authors:** Alexis M. Collier, DHA, MHA¹ · Sophia Z. Shalhout, PhD²³
**Affiliations:**
¹ College of Health & Wellness, University of North Georgia, Dahlonega, GA, USA
² Department of Otolaryngology–Head and Neck Surgery, Harvard Medical School, Boston, MA, USA
³ Mass Eye and Ear, Mass General Brigham, Boston, MA, USA

**Companion paper (multinational validation):** [IDI-Multinational-Validation](https://github.com/colla00/IDI-Multinational-Validation) *(Journal of Biomedical Informatics, under review, 2026)*

---

## Overview

This repository contains the full analysis code for the **Intensive Documentation Index (IDI)** — a zero-burden prognostic framework that extracts temporal documentation rhythm features from nursing chartevents timestamps in the first 24 hours of ICU admission to predict in-hospital mortality.

Applied to **26,153 heart failure ICU admissions** from MIMIC-IV (2008–2019), the IDI modestly but reliably improves mortality prediction beyond traditional clinical variables.

---

## Key Results

Results reflect **temporal validation** (training: 2008–2018; test: 2019):

| Model | AUROC (95% CI) | Calibration Slope | Brier Score |
|-------|---------------|-------------------|-------------|
| Baseline (age, sex, ICU LOS) | 0.658 (0.609–0.710) | 0.92 | 0.1091 |
| IDI-Enhanced | **0.683 (0.631–0.732)** | **0.96** | **0.1080** |

**ΔAUROC = +0.025 (p = 0.015, DeLong test)**
Strongest predictor: `idi_cv_interevent` OR = 1.53 per SD (95% CI 1.35–1.74, p < 0.001)
Temporal stability: mean AUC 0.654 (SD 0.016) across leave-one-year-out cross-validation (2008–2019)

---

## Repository Structure

```
IDI-MIMIC-IV-Mortality/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── idi_features.py        # IDI feature extraction from chartevents timestamps
│   ├── cohort_selection.py    # Inclusion/exclusion criteria applied to MIMIC-IV
│   ├── model.py               # Logistic regression training & temporal validation
│   ├── metrics.py             # AUROC, calibration slope, Brier, DeLong test
│   └── equity_analysis.py     # Subgroup AUC analysis by race/ethnicity
│
├── data/
│   └── README.md              # Data access instructions (PhysioNet DUA required)
│
└── results/
    ├── figures/               # ROC curves, calibration plots, forest plots
    └── tables/                # CSV versions of manuscript tables
```

---

## Data Access

**Raw data are NOT included** in this repository due to PhysioNet Data Use Agreement restrictions.

To reproduce this analysis:
1. Apply for credentialed access at [PhysioNet](https://physionet.org/register/)
2. Download **MIMIC-IV version 2.2**: https://physionet.org/content/mimiciv/2.2/
3. Place the following files in `data/mimic-iv/`:
   - `hosp/admissions.csv`
   - `hosp/patients.csv`
   - `hosp/diagnoses_icd.csv`
   - `icu/icustays.csv`
   - `icu/chartevents.csv`

---

## Installation

```bash
git clone https://github.com/colla00/IDI-MIMIC-IV-Mortality.git
cd IDI-MIMIC-IV-Mortality
pip install -r requirements.txt
```

**Python version:** 3.8+

---

## Usage

Run scripts in order:

```bash
# 1. Select cohort (outputs data/cohort.csv)
python src/cohort_selection.py

# 2. Extract IDI features (outputs data/idi_features.csv)
python src/idi_features.py

# 3. Train models and run temporal validation (outputs results/)
python src/model.py

# 4. Compute performance metrics
python src/metrics.py

# 5. Run equity analysis (outputs results/figures/equity_forest.png)
python src/equity_analysis.py
```

---

## IDI Features

Nine temporal features extracted from nursing chartevents in the first 24 ICU hours:

| Feature | Domain | Description |
|---------|--------|-------------|
| `idi_events_24h` | Volume | Total documentation events |
| `idi_events_per_hour` | Volume | Event rate per hour |
| `idi_max_gap_min` | Surveillance Gap | Maximum inter-event interval (min) |
| `idi_gap_count_60m` | Surveillance Gap | Intervals > 60 minutes |
| `idi_gap_count_120m` | Surveillance Gap | Intervals > 120 minutes |
| `idi_mean_interevent_min` | Rhythm | Mean inter-event interval (min) |
| `idi_std_interevent_min` | Rhythm | SD of inter-event intervals |
| `idi_cv_interevent` | Rhythm | Coefficient of variation (SD/mean) |
| `idi_burstiness` | Rhythm | Burstiness index B = (σ−μ)/(σ+μ) |

---

## Temporal Leakage Prevention

Features with absolute Pearson correlation > 0.30 with ICU length of stay were removed prior to modeling to prevent reverse-causal leakage (longer stay → more documentation → spurious mortality prediction). See Methods section of the manuscript for full details.

---

## Citation

If you use this code, please cite:

```bibtex
@article{collier2026idi,
  title     = {Development and Validation of the Intensive Documentation Index for ICU Mortality Prediction},
  author    = {Collier, Alexis M. and Shalhout, Sophia Z.},
  journal   = {Journal of the American Medical Informatics Association},
  year      = {2026},
  note      = {Under review}
}
```

---

## Patent Notice

The IDI framework is the subject of multiple U.S. provisional patent applications (Patent Pending). VitaSignal LLC is the intended assignee. Licensing inquiries: info@vitasignal.ai

---

## Funding

This research was, in part, funded by the National Institutes of Health through the NIH AIM-AHEAD program.

---

## License

[MIT License](LICENSE) — see LICENSE file for details.
