# Processed Data Report

## Original Shape

- Rows: 101766
- Columns: 50

## Shape After Preprocessing

- Rows: 101766
- Columns: 51

## Missingness Summary Before Normalization

| Column | Missing Count | Missing Rate |
| --- | --- | --- |
| weight | 98569 | 96.86% |
| max_glu_serum | 96420 | 94.75% |
| A1Cresult | 84748 | 83.28% |
| medical_specialty | 49949 | 49.08% |
| payer_code | 40256 | 39.56% |
| race | 2273 | 2.23% |
| diag_3 | 1423 | 1.40% |
| diag_2 | 358 | 0.35% |
| diag_1 | 21 | 0.02% |
| encounter_id | 0 | 0.00% |
| patient_nbr | 0 | 0.00% |
| gender | 0 | 0.00% |
| age | 0 | 0.00% |
| admission_type_id | 0 | 0.00% |
| discharge_disposition_id | 0 | 0.00% |
| admission_source_id | 0 | 0.00% |
| time_in_hospital | 0 | 0.00% |
| num_lab_procedures | 0 | 0.00% |
| num_procedures | 0 | 0.00% |
| num_medications | 0 | 0.00% |

## Missingness Summary After Normalization

| Column | Missing Count | Missing Rate |
| --- | --- | --- |
| weight | 98569 | 96.86% |
| max_glu_serum | 96420 | 94.75% |
| A1Cresult | 84748 | 83.28% |
| medical_specialty | 49949 | 49.08% |
| payer_code | 40256 | 39.56% |
| race | 2273 | 2.23% |
| diag_3 | 1423 | 1.40% |
| diag_2 | 358 | 0.35% |
| diag_1 | 21 | 0.02% |
| encounter_id | 0 | 0.00% |
| patient_nbr | 0 | 0.00% |
| gender | 0 | 0.00% |
| age | 0 | 0.00% |
| admission_type_id | 0 | 0.00% |
| discharge_disposition_id | 0 | 0.00% |
| admission_source_id | 0 | 0.00% |
| time_in_hospital | 0 | 0.00% |
| num_lab_procedures | 0 | 0.00% |
| num_procedures | 0 | 0.00% |
| num_medications | 0 | 0.00% |

## Columns Excluded From Feature Candidates

- encounter_id
- patient_nbr

## Target Distribution (readmitted)

| readmitted | Count |
| --- | --- |
| NO | 54864 |
| >30 | 35545 |
| <30 | 11357 |

## Target Distribution (readmitted_30d)

| readmitted_30d | Count |
| --- | --- |
| 0 | 90409 |
| 1 | 11357 |

## Split Row Counts

- Train rows: 71520
- Validation rows: 15237
- Test rows: 15009

## Split Patient Counts

- Train patients: 50062
- Validation patients: 10728
- Test patients: 10728

## Leakage Check Result

- leakage_check_passed: True

## Warnings

- None

## Assumptions

- Grouped split uses patient_nbr and allows approximate row ratios due to group boundaries.
- readmitted is expected to have only NO, >30, <30 labels without missing values.
- encounter_id and patient_nbr are excluded from model feature candidates.

## Recommended Next Step For Feature Engineering/Modeling

- Implement feature engineering on train split only, then apply transformations to val/test.
- Define explicit imputation strategy for high-missingness columns.
- Build model-ready feature matrix using candidate_feature_columns metadata.
