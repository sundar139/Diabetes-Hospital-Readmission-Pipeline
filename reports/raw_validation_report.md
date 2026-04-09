# Raw Data Validation Report

## Dataset Shape

- Rows: 101766
- Columns: 50

## Required-Column Status

| Column | Status |
| --- | --- |
| encounter_id | PASS |
| patient_nbr | PASS |
| readmitted | PASS |
| age | PASS |
| race | PASS |
| gender | PASS |
| time_in_hospital | PASS |
| num_lab_procedures | PASS |
| num_medications | PASS |
| number_outpatient | PASS |
| number_emergency | PASS |
| number_inpatient | PASS |

## Duplicate Summary

- Duplicate full rows: 0
- encounter_id unique: True
- patient_nbr unique count: 71518

## Missingness Summary

| Column | Missing Count | Missing Rate | ? Count | Empty Count | Whitespace Count |
| --- | --- | --- | --- | --- | --- |
| weight | 98569 | 96.86% | 98569 | 0 | 0 |
| max_glu_serum | 96420 | 94.75% | 0 | 0 | 0 |
| A1Cresult | 84748 | 83.28% | 0 | 0 | 0 |
| medical_specialty | 49949 | 49.08% | 49949 | 0 | 0 |
| payer_code | 40256 | 39.56% | 40256 | 0 | 0 |
| race | 2273 | 2.23% | 2273 | 0 | 0 |
| diag_3 | 1423 | 1.40% | 1423 | 0 | 0 |
| diag_2 | 358 | 0.35% | 358 | 0 | 0 |
| diag_1 | 21 | 0.02% | 21 | 0 | 0 |
| encounter_id | 0 | 0.00% | 0 | 0 | 0 |
| patient_nbr | 0 | 0.00% | 0 | 0 | 0 |
| gender | 0 | 0.00% | 0 | 0 | 0 |
| age | 0 | 0.00% | 0 | 0 | 0 |
| admission_type_id | 0 | 0.00% | 0 | 0 | 0 |
| discharge_disposition_id | 0 | 0.00% | 0 | 0 | 0 |
| admission_source_id | 0 | 0.00% | 0 | 0 | 0 |
| time_in_hospital | 0 | 0.00% | 0 | 0 | 0 |
| num_lab_procedures | 0 | 0.00% | 0 | 0 | 0 |
| num_procedures | 0 | 0.00% | 0 | 0 | 0 |
| num_medications | 0 | 0.00% | 0 | 0 | 0 |

## Target Distribution (readmitted)

| readmitted | Count |
| --- | --- |
| NO | 54864 |
| >30 | 35545 |
| <30 | 11357 |

## Identifier Observations

- encounter_id exists: True
- encounter_id unique: True
- encounter_id duplicate count: 0
- patient_nbr exists: True
- patient_nbr unique count: 71518
- patient_nbr repeated row count: 47021

## Major Warnings

- Found 192849 '?' token(s) representing missing data.
- High missingness (>=30%) detected in: weight, max_glu_serum, A1Cresult, medical_specialty, payer_code.

## Recommended Next Preprocessing Actions

- Normalize null-like tokens ('?', empty, and whitespace-only values) into standard missing values.
- Define a clear strategy for high-missingness columns before feature engineering.
- Decide leakage-safe grouping strategy for patient-level records before train/validation split.
- Confirm target label integrity and document any out-of-vocabulary readmitted values.
- Resolve duplicate records and confirm deduplication policy for encounter-level analytics.
