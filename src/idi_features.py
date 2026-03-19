"""
idi_features.py
---------------
Extracts the nine IDI features from MIMIC-IV nursing chartevents timestamps.

Features derived from the first 24 hours of ICU admission:

  Volume:
    idi_events_24h          - total documentation events
    idi_events_per_hour     - event rate per hour

  Surveillance Gap:
    idi_max_gap_min         - maximum inter-event interval (minutes)
    idi_gap_count_60m       - number of gaps > 60 minutes
    idi_gap_count_120m      - number of gaps > 120 minutes

  Rhythm Regularity:
    idi_mean_interevent_min - mean inter-event interval (minutes)
    idi_std_interevent_min  - SD of inter-event intervals (ddof=1)
    idi_cv_interevent       - coefficient of variation (SD / mean)
    idi_burstiness          - burstiness index B = (sigma - mu) / (sigma + mu)

Inputs:
  data/cohort.csv         - output of cohort_selection.py
  data/mimic-iv/icu/chartevents.csv

Output:
  data/idi_features.csv   - cohort rows merged with IDI feature columns

Reference:
  Collier AM, Shalhout SZ. Development and Validation of the Intensive
  Documentation Index for ICU Mortality Prediction. JAMIA, 2026.
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

DATA_DIR   = "data/mimic-iv"
COHORT     = "data/cohort.csv"
OUT_PATH   = "data/idi_features.csv"

MIN_EVENTS = 10   # minimum documentation events required


def load_chartevents_for_cohort(cohort_df):
    """
    Load chartevents for cohort stay_ids, keeping only the first 24 hours.
    chartevents.csv is large (~30 GB); we read in chunks.
    """
    stay_ids = set(cohort_df["stay_id"].astype(str))
    chunks = []
    print("Reading chartevents in chunks (this may take several minutes)...")
    for chunk in tqdm(pd.read_csv(
            os.path.join(DATA_DIR, "icu/chartevents.csv"),
            usecols=["stay_id", "charttime", "storetime"],
            parse_dates=["charttime", "storetime"],
            chunksize=500_000)):
        chunk = chunk[chunk["stay_id"].astype(str).isin(stay_ids)]
        if len(chunk):
            chunks.append(chunk)
    ce = pd.concat(chunks, ignore_index=True)
    print(f"  Loaded {len(ce):,} chartevent rows for {ce['stay_id'].nunique():,} stays")
    return ce


def extract_idi_for_stay(timestamps_sorted):
    """
    Compute IDI features from a sorted list of charttime timestamps (minutes).
    Returns a dict of 9 features, or None if fewer than 2 events.

    Inter-event intervals are computed in seconds then converted to float
    minutes to avoid integer truncation from timedelta64[m].
    Sample standard deviation (ddof=1) is used throughout.
    """
    n = len(timestamps_sorted)
    if n < 2:
        return None

    ts   = pd.Series(timestamps_sorted)
    gaps = ts.diff().dropna()           # inter-event intervals in minutes

    mu  = gaps.mean()
    sig = gaps.std(ddof=1) if len(gaps) > 1 else 0.0

    cv         = (sig / mu)                if mu > 0         else 0.0
    burstiness = ((sig - mu) / (sig + mu)) if (sig + mu) > 0 else 0.0

    return {
        "idi_events_24h":          n,
        "idi_events_per_hour":     n / 24.0,
        "idi_max_gap_min":         gaps.max(),
        "idi_gap_count_60m":       int((gaps > 60).sum()),
        "idi_gap_count_120m":      int((gaps > 120).sum()),
        "idi_mean_interevent_min": mu,
        "idi_std_interevent_min":  sig,
        "idi_cv_interevent":       cv,
        "idi_burstiness":          burstiness,
    }


def extract_all_idi(cohort_df, ce):
    """
    For each ICU stay, filter chartevents to first 24 hours,
    deduplicate timestamps, then compute IDI features.
    """
    ce = ce.merge(cohort_df[["stay_id", "intime"]], on="stay_id", how="inner")
    ce["intime"]    = pd.to_datetime(ce["intime"])
    ce["charttime"] = pd.to_datetime(ce["charttime"])

    # Keep only first 24 hours from ICU admission
    ce["hours_from_admit"] = (ce["charttime"] - ce["intime"]).dt.total_seconds() / 3600
    ce = ce[(ce["hours_from_admit"] >= 0) & (ce["hours_from_admit"] <= 24)]

    # Convert to float minutes to avoid timedelta64[m] integer truncation
    ce["minutes"] = ce["hours_from_admit"] * 60.0
    ce = ce.sort_values(["stay_id", "minutes"])
    ce = ce.drop_duplicates(subset=["stay_id", "minutes"])

    records = []
    for stay_id, grp in tqdm(ce.groupby("stay_id"), desc="Extracting IDI"):
        ts = grp["minutes"].tolist()
        if len(ts) < MIN_EVENTS:
            continue
        feats = extract_idi_for_stay(ts)
        if feats is not None:
            feats["stay_id"] = stay_id
            records.append(feats)

    feat_df = pd.DataFrame(records)
    print(f"\nIDI features extracted for {len(feat_df):,} stays "
          f"(excluded {len(cohort_df) - len(feat_df):,} with < {MIN_EVENTS} events)")
    return feat_df


def main():
    os.makedirs("data", exist_ok=True)

    cohort = pd.read_csv(COHORT, parse_dates=["intime", "outtime"])
    print(f"Cohort loaded: {len(cohort):,} stays")

    ce      = load_chartevents_for_cohort(cohort)
    feat_df = extract_all_idi(cohort, ce)

    out = cohort.merge(feat_df, on="stay_id", how="inner")

    print(f"\nFinal dataset: {len(out):,} stays | "
          f"mortality {out['hospital_mortality'].mean() * 100:.2f}%")
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
