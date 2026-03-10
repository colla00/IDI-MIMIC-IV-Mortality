"""
cohort_selection.py
-------------------
Selects the MIMIC-IV heart failure ICU cohort used in:
  Collier & Shalhout, JAMIA 2026

Inclusion criteria:
  - Adult ICU admissions (age >= 18)
  - Heart failure as primary or secondary diagnosis (ICD-9/ICD-10)
  - ICU LOS >= 24 hours
  - >= 10 nursing documentation events in first 24 hours
  - First ICU admission per patient only

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


def load_tables():
    print("Loading MIMIC-IV tables...")
    admissions  = pd.read_csv(os.path.join(DATA_DIR, "hosp/admissions.csv"),
                              parse_dates=["admittime", "dischtime", "deathtime"])
    patients    = pd.read_csv(os.path.join(DATA_DIR, "hosp/patients.csv"))
    diagnoses   = pd.read_csv(os.path.join(DATA_DIR, "hosp/diagnoses_icd.csv"))
    icustays    = pd.read_csv(os.path.join(DATA_DIR, "icu/icustays.csv"),
                              parse_dates=["intime", "outtime"])
    print(f"  Admissions: {len(admissions):,}")
    print(f"  ICU stays:  {len(icustays):,}")
    return admissions, patients, diagnoses, icustays


def flag_heart_failure(diagnoses):
    """Return set of hadm_ids with a HF diagnosis."""
    icd9_mask  = diagnoses["icd_version"].eq(9) & diagnoses["icd_code"].isin(HF_ICD9)
    icd10_mask = diagnoses["icd_version"].eq(10) & diagnoses["icd_code"].str.startswith(
        tuple(HF_ICD10))
    hf_hadms = diagnoses.loc[icd9_mask | icd10_mask, "hadm_id"].unique()
    return set(hf_hadms)


def compute_anchor_age(patients, admissions):
    """Compute age at ICU admission using MIMIC-IV anchor_age + anchor_year."""
    df = admissions.merge(
        patients[["subject_id", "anchor_age", "anchor_year"]],
        on="subject_id", how="left")
    df["admit_year"] = df["admittime"].dt.year
    df["age"] = df["anchor_age"] + (df["admit_year"] - df["anchor_year"])
    return df


def select_cohort():
    admissions, patients, diagnoses, icustays = load_tables()

    # --- Step 1: merge ICU stays with admissions & patients ---
    df = icustays.merge(admissions[["hadm_id", "admittime", "dischtime",
                                     "deathtime", "hospital_expire_flag",
                                     "subject_id"]],
                        on="hadm_id", how="inner")
    df = compute_anchor_age(patients, df.rename(columns={"admittime_x": "admittime"})
                            if "admittime_x" in df.columns else df)
    print(f"\nAfter ICU-admission merge: {len(df):,} rows")

    # --- Step 2: adult admissions (age >= 18) ---
    df = df[df["age"] >= 18]
    print(f"After age >= 18: {len(df):,}")

    # --- Step 3: heart failure diagnosis ---
    hf_hadms = flag_heart_failure(diagnoses)
    df = df[df["hadm_id"].isin(hf_hadms)]
    print(f"After HF diagnosis filter: {len(df):,}")

    # --- Step 4: ICU LOS >= 24 hours ---
    df["icu_los_hours"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 3600
    df = df[df["icu_los_hours"] >= 24]
    print(f"After ICU LOS >= 24 h: {len(df):,}")

    # --- Step 5: first ICU admission per patient ---
    df = df.sort_values(["subject_id", "intime"])
    df = df.groupby("subject_id", as_index=False).first()
    print(f"After first-admission-per-patient: {len(df):,}")

    # --- Step 6: define outcome (in-hospital mortality) ---
    df["mortality"] = df["hospital_expire_flag"].astype(int)

    # --- Step 7: drop rows with missing baseline covariates ---
    df = df.dropna(subset=["age", "icu_los_hours"])

    # Note: the >= 10 documentation events criterion is applied after
    # IDI feature extraction in idi_features.py

    print(f"\nFinal cohort (before doc-event filter): {len(df):,} admissions")
    print(f"In-hospital mortality: {df['mortality'].mean()*100:.2f}%  "
          f"(n={df['mortality'].sum():,})")

    cols = ["subject_id", "hadm_id", "stay_id", "intime", "outtime",
            "age", "gender", "icu_los_hours", "mortality"]
    # gender may be in patients table
    if "gender" not in df.columns:
        df = df.merge(patients[["subject_id", "gender"]], on="subject_id", how="left")

    df[cols].to_csv(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")
    return df


if __name__ == "__main__":
    select_cohort()
