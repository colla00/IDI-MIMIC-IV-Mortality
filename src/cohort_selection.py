"""
cohort_selection.py  [FIXED v2]
-------------------
Selects the MIMIC-IV heart failure ICU cohort used in:
  Collier & Shalhout, JAMIA 2026

FIXES APPLIED:
  BUG-1: Removed subject_id from admissions merge subset to prevent
         subject_id_x / subject_id_y column collision that crashed
         groupby("subject_id") and all downstream merges.
  BUG-2: Added 'race' column from admissions table so equity_analysis.py
         can run. Without this fix equity_analysis silently returned early.

Inclusion criteria:
  - Adult ICU admissions (age >= 18)
  - Heart failure as primary or secondary diagnosis (ICD-9/ICD-10)
  - ICU LOS >= 24 hours
  - First ICU admission per patient only
  - >= 10 nursing documentation events in first 24 h (applied in idi_features.py)

Output: data/cohort.csv
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = "data/mimic-iv"
OUT_PATH = "data/cohort.csv"

# ICD codes for heart failure
HF_ICD9  = ["4280", "4281", "42820", "42821", "42822", "42823",
             "42830", "42831", "42832", "42833", "42840", "42841",
             "42842", "42843", "4289"]
HF_ICD10 = ["I50", "I500", "I501", "I5020", "I5021", "I5022",
             "I5023", "I5030", "I5031", "I5032", "I5033",
             "I5040", "I5041", "I5042", "I5043", "I509",
             "I5081", "I5082", "I5083", "I5084", "I5089"]

# MIMIC-IV race strings to standardised label
# Using str.contains() in equity_analysis.py, so these are substrings.
RACE_MAP = {
    "WHITE":                     "White",
    "BLACK":                     "Black",
    "HISPANIC":                  "Hispanic",
    "ASIAN":                     "Asian",
}


def load_tables():
    print("Loading MIMIC-IV tables...")
    admissions = pd.read_csv(
        os.path.join(DATA_DIR, "hosp/admissions.csv"),
        parse_dates=["admittime", "dischtime", "deathtime"],
        # FIX BUG-2: include 'race' column
        usecols=["hadm_id", "admittime", "dischtime", "deathtime",
                 "hospital_expire_flag", "race"],
    )
    patients   = pd.read_csv(os.path.join(DATA_DIR, "hosp/patients.csv"))
    diagnoses  = pd.read_csv(os.path.join(DATA_DIR, "hosp/diagnoses_icd.csv"))
    icustays   = pd.read_csv(
        os.path.join(DATA_DIR, "icu/icustays.csv"),
        parse_dates=["intime", "outtime"],
    )
    print(f"  Admissions : {len(admissions):,}")
    print(f"  ICU stays  : {len(icustays):,}")
    return admissions, patients, diagnoses, icustays


def flag_heart_failure(diagnoses):
    """Return set of hadm_ids with any HF diagnosis."""
    icd9_mask  = diagnoses["icd_version"].eq(9)  & diagnoses["icd_code"].isin(HF_ICD9)
    icd10_mask = diagnoses["icd_version"].eq(10) & diagnoses["icd_code"].str.startswith(
        tuple(HF_ICD10))
    return set(diagnoses.loc[icd9_mask | icd10_mask, "hadm_id"].unique())


def compute_anchor_age(df, patients):
    """
    Add 'age' column to df (which already contains admittime).
    Uses MIMIC-IV anchor_age + anchor_year approach.
    df must already contain 'subject_id' and 'admittime'.
    """
    df = df.merge(
        patients[["subject_id", "anchor_age", "anchor_year", "gender"]],
        on="subject_id", how="left",
    )
    df["admit_year"] = df["admittime"].dt.year
    df["age"] = df["anchor_age"] + (df["admit_year"] - df["anchor_year"])
    return df


def standardise_race(race_series: pd.Series) -> pd.Series:
    """
    Map MIMIC-IV verbose race strings to a compact label.
    Unmatched values become 'Other'.
    """
    s = race_series.fillna("UNKNOWN").str.upper()
    result = pd.Series("Other", index=race_series.index)
    for keyword, label in RACE_MAP.items():
        result[s.str.contains(keyword, na=False)] = label
    return result


def select_cohort():
    admissions, patients, diagnoses, icustays = load_tables()

    # ── Step 1: merge ICU stays with admissions ───────────────────────────────
    # FIX BUG-1: admissions subset does NOT include subject_id (already in
    # icustays), eliminating the subject_id_x / subject_id_y column collision.
    df = icustays.merge(
        admissions[["hadm_id", "admittime", "dischtime", "deathtime",
                    "hospital_expire_flag", "race"]],   # no subject_id here
        on="hadm_id", how="inner",
    )
    print(f"\nAfter ICU-admission merge: {len(df):,} rows")

    # ── Step 2: add age and gender from patients ──────────────────────────────
    df = compute_anchor_age(df, patients)

    # ── Step 3: adult admissions (age >= 18) ─────────────────────────────────
    df = df[df["age"] >= 18]
    print(f"After age >= 18: {len(df):,}")

    # ── Step 4: heart failure diagnosis ──────────────────────────────────────
    hf_hadms = flag_heart_failure(diagnoses)
    df = df[df["hadm_id"].isin(hf_hadms)]
    print(f"After HF diagnosis filter: {len(df):,}")

    # ── Step 5: ICU LOS >= 24 hours ──────────────────────────────────────────
    df["icu_los_hours"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 3600
    df = df[df["icu_los_hours"] >= 24]
    print(f"After ICU LOS >= 24 h: {len(df):,}")

    # ── Step 6: first ICU admission per patient ───────────────────────────────
    df = df.sort_values(["subject_id", "intime"])
    df = df.groupby("subject_id", as_index=False).first()
    print(f"After first-admission-per-patient: {len(df):,}")

    # ── Step 7: define outcome (in-hospital mortality) ────────────────────────
    df["hospital_mortality"] = df["hospital_expire_flag"].astype(int)

    # ── Step 8: standardise race (FIX BUG-2) ─────────────────────────────────
    df["race"] = standardise_race(df["race"])
    print(f"\nRace distribution:\n{df['race'].value_counts().to_string()}")

    # ── Step 9: drop rows missing key baseline covariates ────────────────────
    df = df.dropna(subset=["age", "icu_los_hours"])

    print(f"\nFinal cohort (before doc-event filter): {len(df):,} admissions")
    print(f"In-hospital mortality: {df['hospital_mortality'].mean()*100:.2f}%  "
          f"(n={df['hospital_mortality'].sum():,})")

    cols = ["subject_id", "hadm_id", "stay_id", "intime", "outtime",
            "age", "gender", "race", "icu_los_hours", "hospital_mortality"]
    df[cols].to_csv(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")
    return df[cols]


if __name__ == "__main__":
    select_cohort()
