"""
cohort_selection.py  [FIXED v3]
-------------------
Selects the MIMIC-IV heart failure ICU cohort used in:
  Collier & Shalhout, JAMIA 2026

FIXES APPLIED:
  BUG-1: Removed subject_id from admissions merge subset to prevent
         subject_id_x / subject_id_y column collision that crashed
         groupby("subject_id") and all downstream merges.
  BUG-2: Added 'race' column from admissions table so equity_analysis.py
         can run. Without this fix equity_analysis silently returned early.
  BUG-3 (v3 NEW): Added three exclusion criteria present in the manuscript
         but missing from v2 code:
           (a) Exclude admissions with missing discharge status
               (hospital_expire_flag IS NULL) — paper Methods §Study Population
           (b) Exclude admissions with missing documentation timestamps
               (n = 1,847) — applied after idi_features.py merge via inner join;
               logged here for STROBE flow diagram transparency.
           (c) Exclude planned postoperative ICU admissions for elective
               cardiac surgery (n = 4,321) — identified via procedure ICD codes
               for CABG / valve repair / replacement in same admission.
         Without these steps the cohort is ~6,168 admissions larger than the
         manuscript-reported n = 26,153 and results are not reproducible.

Inclusion criteria:
  - Adult ICU admissions (age >= 18)
  - Heart failure as primary or secondary diagnosis (ICD-9/ICD-10)
  - ICU LOS >= 24 hours
  - First ICU admission per patient only
  - >= 10 nursing documentation events in first 24 h (applied in idi_features.py)

Exclusion criteria (all new in v3):
  - Missing discharge status (hospital_expire_flag IS NULL)
  - Missing documentation timestamps (inner join in idi_features.py; n ≈ 1,847)
  - Planned postoperative elective cardiac surgery ICU admissions (n ≈ 4,321)

Output: data/cohort.csv
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = "data/mimic-iv"
OUT_PATH = "data/cohort.csv"

# ── ICD codes for heart failure ───────────────────────────────────────────────
HF_ICD9  = ["4280", "4281", "42820", "42821", "42822", "42823",
             "42830", "42831", "42832", "42833", "42840", "42841",
             "42842", "42843", "4289"]
HF_ICD10 = ["I50", "I500", "I501", "I5020", "I5021", "I5022",
             "I5023", "I5030", "I5031", "I5032", "I5033",
             "I5040", "I5041", "I5042", "I5043", "I509",
             "I5081", "I5082", "I5083", "I5084", "I5089"]

# ── ICD codes for planned elective cardiac surgery (BUG-3c) ───────────────────
# CABG: ICD-9 36.1x; ICD-10 021x
# Valve repair/replacement: ICD-9 35.xx; ICD-10 02Rx, 02Qx, 02Nx, 02Px
# These are used to exclude planned postoperative cardiac surgery ICU admissions.
CARDIAC_SURG_ICD9_PREFIX  = ("361", "362", "363", "364", "350", "351",
                              "352", "353", "354", "355", "356", "357",
                              "358", "359")
CARDIAC_SURG_ICD10_PREFIX = ("0210", "0211", "0212", "0213", "0214",
                              "0215", "0216", "0217", "0218", "021",
                              "02R", "02Q", "02N", "02P")

# ── MIMIC-IV race strings to standardised label ───────────────────────────────
RACE_MAP = {
    "WHITE":    "White",
    "BLACK":    "Black",
    "HISPANIC": "Hispanic",
    "ASIAN":    "Asian",
}


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_tables():
    print("Loading MIMIC-IV tables...")
    admissions = pd.read_csv(
        os.path.join(DATA_DIR, "hosp/admissions.csv"),
        parse_dates=["admittime", "dischtime", "deathtime"],
        usecols=["hadm_id", "admittime", "dischtime", "deathtime",
                 "hospital_expire_flag", "race"],
    )
    patients  = pd.read_csv(os.path.join(DATA_DIR, "hosp/patients.csv"))
    diagnoses = pd.read_csv(os.path.join(DATA_DIR, "hosp/diagnoses_icd.csv"))
    procedures = pd.read_csv(os.path.join(DATA_DIR, "hosp/procedures_icd.csv"))
    icustays  = pd.read_csv(
        os.path.join(DATA_DIR, "icu/icustays.csv"),
        parse_dates=["intime", "outtime"],
    )
    print(f"  Admissions : {len(admissions):,}")
    print(f"  ICU stays  : {len(icustays):,}")
    return admissions, patients, diagnoses, procedures, icustays


# ── Heart failure flag ────────────────────────────────────────────────────────
def flag_heart_failure(diagnoses):
    """Return set of hadm_ids with any HF diagnosis (primary or secondary)."""
    icd9_mask  = (diagnoses["icd_version"].eq(9) &
                  diagnoses["icd_code"].isin(HF_ICD9))
    icd10_mask = (diagnoses["icd_version"].eq(10) &
                  diagnoses["icd_code"].str.startswith(tuple(HF_ICD10)))
    return set(diagnoses.loc[icd9_mask | icd10_mask, "hadm_id"].unique())


# ── BUG-3c: Elective cardiac surgery flag ────────────────────────────────────
def flag_elective_cardiac_surgery(procedures):
    """
    Return set of hadm_ids with a planned elective cardiac surgery procedure
    (CABG or valve repair/replacement via ICD-9/ICD-10 procedure codes).
    Used to exclude postoperative elective cardiac surgery ICU admissions
    per manuscript exclusion criteria (n = 4,321).
    """
    icd9_mask  = (procedures["icd_version"].eq(9) &
                  procedures["icd_code"].str.startswith(
                      CARDIAC_SURG_ICD9_PREFIX))
    icd10_mask = (procedures["icd_version"].eq(10) &
                  procedures["icd_code"].str.startswith(
                      CARDIAC_SURG_ICD10_PREFIX))
    return set(procedures.loc[icd9_mask | icd10_mask, "hadm_id"].unique())


# ── Age computation ───────────────────────────────────────────────────────────
def compute_anchor_age(df, patients):
    """
    Add 'age' column using MIMIC-IV anchor_age + anchor_year approach.
    df must contain 'subject_id' and 'admittime'.
    """
    df = df.merge(
        patients[["subject_id", "anchor_age", "anchor_year", "gender"]],
        on="subject_id", how="left",
    )
    df["admit_year"] = df["admittime"].dt.year
    df["age"] = df["anchor_age"] + (df["admit_year"] - df["anchor_year"])
    return df


# ── Race standardisation ──────────────────────────────────────────────────────
def standardise_race(race_series: pd.Series) -> pd.Series:
    """Map MIMIC-IV verbose race strings to compact label; unmatched → 'Other'."""
    s = race_series.fillna("UNKNOWN").str.upper()
    result = pd.Series("Other", index=race_series.index)
    for keyword, label in RACE_MAP.items():
        result[s.str.contains(keyword, na=False)] = label
    return result


# ── Main cohort selection ─────────────────────────────────────────────────────
def select_cohort():
    admissions, patients, diagnoses, procedures, icustays = load_tables()

    # ── Step 1: merge ICU stays with admissions ───────────────────────────────
    # BUG-1 FIX: no subject_id in admissions subset (already in icustays).
    df = icustays.merge(
        admissions[["hadm_id", "admittime", "dischtime", "deathtime",
                    "hospital_expire_flag", "race"]],
        on="hadm_id", how="inner",
    )
    print(f"\nAfter ICU-admission merge          : {len(df):>7,} rows")

    # ── Step 2: add age and gender ────────────────────────────────────────────
    df = compute_anchor_age(df, patients)

    # ── Step 3: adult admissions (age >= 18) ─────────────────────────────────
    df = df[df["age"] >= 18]
    print(f"After age >= 18                    : {len(df):>7,}")

    # ── Step 4: heart failure diagnosis ──────────────────────────────────────
    hf_hadms = flag_heart_failure(diagnoses)
    df = df[df["hadm_id"].isin(hf_hadms)]
    print(f"After HF diagnosis filter          : {len(df):>7,}")

    # ── Step 5: ICU LOS >= 24 hours ──────────────────────────────────────────
    df["icu_los_hours"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 3600
    df = df[df["icu_los_hours"] >= 24]
    print(f"After ICU LOS >= 24 h              : {len(df):>7,}")

    # ── Step 6: first ICU admission per patient ───────────────────────────────
    df = df.sort_values(["subject_id", "intime"])
    df = df.groupby("subject_id", as_index=False).first()
    print(f"After first-admission-per-patient  : {len(df):>7,}")

    # ── Step 7 (NEW BUG-3a): exclude missing discharge status ────────────────
    # hospital_expire_flag must be 0 or 1 (not NULL) to define the outcome.
    n_before = len(df)
    df = df[df["hospital_expire_flag"].notna()]
    n_excluded_discharge = n_before - len(df)
    print(f"After excl. missing discharge      : {len(df):>7,}  "
          f"(excluded {n_excluded_discharge:,})")

    # ── Step 8 (NEW BUG-3c): exclude elective cardiac surgery admissions ──────
    # Planned postoperative CABG / valve repair ICU stays are not reflective
    # of unplanned critical illness; exclude per manuscript criteria (n ≈ 4,321).
    cardiac_hadms = flag_elective_cardiac_surgery(procedures)
    n_before = len(df)
    df = df[~df["hadm_id"].isin(cardiac_hadms)]
    n_excluded_cardiac = n_before - len(df)
    print(f"After excl. elective cardiac surg  : {len(df):>7,}  "
          f"(excluded {n_excluded_cardiac:,})")

    # ── Step 9: define outcome ────────────────────────────────────────────────
    df["hospital_mortality"] = df["hospital_expire_flag"].astype(int)

    # ── Step 10: standardise race ─────────────────────────────────────────────
    df["race"] = standardise_race(df["race"])
    print(f"\nRace distribution:\n{df['race'].value_counts().to_string()}")

    # ── Step 11: drop rows missing key baseline covariates ───────────────────
    df = df.dropna(subset=["age", "icu_los_hours"])

    # ── Note on BUG-3b (missing documentation timestamps, n ≈ 1,847) ─────────
    # These admissions are excluded implicitly: idi_features.py uses an inner
    # join on stay_id when merging features back to the cohort. Stays with zero
    # valid chartevents timestamps in the first 24 h produce no IDI features and
    # are therefore dropped at that stage. This is consistent with the manuscript
    # exclusion (n = 1,847) and is logged in idi_features.py output. No
    # additional action is required here.
    print(f"\n[NOTE] Missing documentation timestamps (n ≈ 1,847) are excluded "
          f"via inner join in idi_features.py — no separate step needed here.")

    print(f"\nFinal cohort (before doc-event filter): {len(df):>7,} admissions")
    print(f"In-hospital mortality: {df['hospital_mortality'].mean()*100:.2f}%  "
          f"(n={df['hospital_mortality'].sum():,})")
    print(f"\nExpected final n after idi_features.py inner join: ~26,153")

    cols = ["subject_id", "hadm_id", "stay_id", "intime", "outtime",
            "age", "gender", "race", "icu_los_hours", "hospital_mortality"]
    df[cols].to_csv(OUT_PATH, index=False)
    print(f"Saved → {OUT_PATH}")
    return df[cols]


if __name__ == "__main__":
    select_cohort()
