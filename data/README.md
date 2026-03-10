# Data Directory

## Raw data are NOT included in this repository.

Due to PhysioNet Data Use Agreement (DUA) restrictions, MIMIC-IV patient data
cannot be redistributed. You must obtain access independently.

## How to get access

1. Create a PhysioNet account: https://physionet.org/register/
2. Complete the required CITI training course
3. Sign the PhysioNet DUA
4. Download **MIMIC-IV version 2.2**: https://physionet.org/content/mimiciv/2.2/

## Required files

Place the following files in `data/mimic-iv/` before running any scripts:

```
data/
└── mimic-iv/
    ├── hosp/
    │   ├── admissions.csv
    │   ├── patients.csv
    │   └── diagnoses_icd.csv
    └── icu/
        ├── icustays.csv
        └── chartevents.csv   # large file (~30 GB uncompressed)
```

## Notes

- `chartevents.csv` is approximately 30 GB uncompressed. Ensure sufficient disk space.
- All analysis uses MIMIC-IV **version 2.2** specifically. Other versions may produce different cohort sizes.
- Date fields in MIMIC-IV are shifted for de-identification. Year 2019 in the dataset corresponds to the most recent calendar stratum.
