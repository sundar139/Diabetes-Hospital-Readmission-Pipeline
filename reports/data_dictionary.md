# Data Dictionary

| Column | Role | DType | Non-null | Unique | Missing-token rate | Example values | Description | Tags |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| encounter_id | identifier | int64 | 101766 | 101766 | 0.00% | 2278392, 149190, 64410 | Encounter-level identifier for each hospital visit. | identifier |
| patient_nbr | identifier | int64 | 101766 | 71518 | 0.00% | 8222157, 55629189, 86047875 | Patient-level identifier shared across multiple encounters. | identifier |
| race | categorical | object | 101766 | 6 | 2.23% | Caucasian, AfricanAmerican, Other | Self-reported race category. |  |
| gender | categorical | object | 101766 | 3 | 0.00% | Female, Male, Unknown/Invalid | Patient gender category. |  |
| age | categorical | object | 101766 | 10 | 0.00% | [0-10), [10-20), [20-30) | Age bucket at time of encounter. |  |
| weight | categorical | object | 101766 | 10 | 96.86% | [75-100), [50-75), [0-25) | Categorical feature representing grouped or coded patient/encounter context. |  |
| admission_type_id | numeric | int64 | 101766 | 8 | 0.00% | 6, 1, 2 | Numeric feature likely suitable for descriptive statistics and scaling decisions. |  |
| discharge_disposition_id | numeric | int64 | 101766 | 26 | 0.00% | 25, 1, 3 | Numeric feature likely suitable for descriptive statistics and scaling decisions. |  |
| admission_source_id | numeric | int64 | 101766 | 17 | 0.00% | 1, 7, 2 | Numeric feature likely suitable for descriptive statistics and scaling decisions. |  |
| time_in_hospital | numeric | int64 | 101766 | 14 | 0.00% | 1, 3, 2 | Length of stay in days. | utilization |
| payer_code | categorical | object | 101766 | 18 | 39.56% | MC, MD, HM | Categorical feature representing grouped or coded patient/encounter context. |  |
| medical_specialty | categorical | object | 101766 | 73 | 49.08% | Pediatrics-Endocrinology, InternalMedicine, Family/GeneralPractice | Categorical feature representing grouped or coded patient/encounter context. |  |
| num_lab_procedures | numeric | int64 | 101766 | 118 | 0.00% | 41, 59, 11 | Number of laboratory procedures during encounter. | utilization |
| num_procedures | numeric | int64 | 101766 | 7 | 0.00% | 0, 5, 1 | Numeric feature likely suitable for descriptive statistics and scaling decisions. | utilization |
| num_medications | numeric | int64 | 101766 | 75 | 0.00% | 1, 18, 13 | Number of medications prescribed during encounter. | utilization |
| number_outpatient | numeric | int64 | 101766 | 39 | 0.00% | 0, 2, 1 | Prior outpatient visits count. | utilization |
| number_emergency | numeric | int64 | 101766 | 33 | 0.00% | 0, 1, 2 | Prior emergency visits count. | utilization |
| number_inpatient | numeric | int64 | 101766 | 21 | 0.00% | 0, 1, 2 | Prior inpatient visits count. | utilization |
| diag_1 | diagnosis-like | object | 101766 | 717 | 0.02% | 250.83, 276, 648 | Diagnosis code feature captured during or around the encounter. | diagnosis |
| diag_2 | diagnosis-like | object | 101766 | 749 | 0.35% | 250.01, 250, 250.43 | Diagnosis code feature captured during or around the encounter. | diagnosis |
| diag_3 | diagnosis-like | object | 101766 | 790 | 1.40% | 255, V27, 403 | Diagnosis code feature captured during or around the encounter. | diagnosis |
| number_diagnoses | numeric | int64 | 101766 | 16 | 0.00% | 1, 9, 6 | Numeric feature likely suitable for descriptive statistics and scaling decisions. |  |
| max_glu_serum | categorical | object | 5346 | 3 | 94.75% | >300, Norm, >200 | Categorical feature representing grouped or coded patient/encounter context. |  |
| A1Cresult | categorical | object | 17018 | 3 | 83.28% | >7, >8, Norm | Categorical feature representing grouped or coded patient/encounter context. |  |
| metformin | medication-status | object | 101766 | 4 | 0.00% | No, Steady, Up | Medication exposure or medication-change indicator for the encounter. | medication |
| repaglinide | medication-status | object | 101766 | 4 | 0.00% | No, Up, Steady | Medication exposure or medication-change indicator for the encounter. | medication |
| nateglinide | medication-status | object | 101766 | 4 | 0.00% | No, Steady, Down | Medication exposure or medication-change indicator for the encounter. | medication |
| chlorpropamide | medication-status | object | 101766 | 4 | 0.00% | No, Steady, Down | Medication exposure or medication-change indicator for the encounter. | medication |
| glimepiride | medication-status | object | 101766 | 4 | 0.00% | No, Steady, Down | Medication exposure or medication-change indicator for the encounter. | medication |
| acetohexamide | medication-status | object | 101766 | 2 | 0.00% | No, Steady | Medication exposure or medication-change indicator for the encounter. | medication |
| glipizide | medication-status | object | 101766 | 4 | 0.00% | No, Steady, Up | Medication exposure or medication-change indicator for the encounter. | medication |
| glyburide | medication-status | object | 101766 | 4 | 0.00% | No, Steady, Up | Medication exposure or medication-change indicator for the encounter. | medication |
| tolbutamide | medication-status | object | 101766 | 2 | 0.00% | No, Steady | Medication exposure or medication-change indicator for the encounter. | medication |
| pioglitazone | medication-status | object | 101766 | 4 | 0.00% | No, Steady, Up | Medication exposure or medication-change indicator for the encounter. | medication |
| rosiglitazone | medication-status | object | 101766 | 4 | 0.00% | No, Steady, Up | Medication exposure or medication-change indicator for the encounter. | medication |
| acarbose | medication-status | object | 101766 | 4 | 0.00% | No, Steady, Up | Medication exposure or medication-change indicator for the encounter. | medication |
| miglitol | medication-status | object | 101766 | 4 | 0.00% | No, Steady, Down | Medication exposure or medication-change indicator for the encounter. | medication |
| troglitazone | medication-status | object | 101766 | 2 | 0.00% | No, Steady | Medication exposure or medication-change indicator for the encounter. | medication |
| tolazamide | medication-status | object | 101766 | 3 | 0.00% | No, Steady, Up | Medication exposure or medication-change indicator for the encounter. | medication |
| examide | medication-status | object | 101766 | 1 | 0.00% | No | Medication exposure or medication-change indicator for the encounter. | medication |
| citoglipton | medication-status | object | 101766 | 1 | 0.00% | No | Medication exposure or medication-change indicator for the encounter. | medication |
| insulin | medication-status | object | 101766 | 4 | 0.00% | No, Up, Steady | Medication exposure or medication-change indicator for the encounter. | medication |
| glyburide-metformin | medication-status | object | 101766 | 4 | 0.00% | No, Steady, Down | Medication exposure or medication-change indicator for the encounter. | medication |
| glipizide-metformin | medication-status | object | 101766 | 2 | 0.00% | No, Steady | Medication exposure or medication-change indicator for the encounter. | medication |
| glimepiride-pioglitazone | medication-status | object | 101766 | 2 | 0.00% | No, Steady | Medication exposure or medication-change indicator for the encounter. | medication |
| metformin-rosiglitazone | medication-status | object | 101766 | 2 | 0.00% | No, Steady | Medication exposure or medication-change indicator for the encounter. | medication |
| metformin-pioglitazone | medication-status | object | 101766 | 2 | 0.00% | No, Steady | Medication exposure or medication-change indicator for the encounter. | medication |
| change | medication-status | object | 101766 | 2 | 0.00% | No, Ch | Medication exposure or medication-change indicator for the encounter. | medication |
| diabetesMed | medication-status | object | 101766 | 2 | 0.00% | No, Yes | Medication exposure or medication-change indicator for the encounter. | medication |
| readmitted | target | object | 101766 | 3 | 0.00% | NO, >30, <30 | Readmission label indicating timing of return admission. | target |
