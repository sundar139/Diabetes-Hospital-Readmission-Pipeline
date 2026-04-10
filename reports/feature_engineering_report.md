# Feature Engineering Report

## Engineered Feature Logic

- recurrency: number of encounters for the same patient_nbr within the split minus 1, clipped at 0.
- patient_severity: 0.30*(time_in_hospital/14 clipped) + 0.25*(diagnoses_count/16 clipped) + 0.20*(number_inpatient/10 clipped) + 0.15*(number_outpatient/10 clipped) + 0.10*(number_emergency/10 clipped).
- medication_change_ratio: count(status in {Up, Down}) / count(status in {No, Steady, Up, Down}) across medication columns.
- utilization_intensity: number_inpatient + number_outpatient + number_emergency.
- complex_discharge_flag: 1 when discharge_disposition_id is not in {1, 6, 8}; missing/malformed defaults to 1.
- age_bucket_risk: lower decade extracted from age bucket [a-b) and mapped to a//10; missing/malformed maps to -1.

## Feature Source Columns

| Engineered Feature | Source Columns |
| --- | --- |
| recurrency | patient_nbr |
| patient_severity | time_in_hospital, number_diagnoses, number_inpatient, number_outpatient, number_emergency |
| medication_change_ratio | metformin, repaglinide, nateglinide, chlorpropamide, glimepiride, acetohexamide, glipizide, glyburide, tolbutamide, pioglitazone, rosiglitazone, acarbose, miglitol, troglitazone, tolazamide, examide, citoglipton, insulin, glyburide-metformin, glipizide-metformin, glimepiride-pioglitazone, metformin-rosiglitazone, metformin-pioglitazone |
| utilization_intensity | number_inpatient, number_outpatient, number_emergency |
| complex_discharge_flag | discharge_disposition_id |
| age_bucket_risk | age |

## Clinical and Operational Relevance

- recurrency captures repeated-patient encounter burden, a strong operational readmission context signal.
- patient_severity summarizes acute burden and prior utilization into a single clinically interpretable index.
- medication_change_ratio approximates treatment-instability intensity during admission.
- utilization_intensity captures cumulative prior care-touch volume.
- complex_discharge_flag reflects transition-of-care complexity and potential post-discharge risk.
- age_bucket_risk injects age-associated baseline vulnerability in ordinal form.

## Assumptions

- recurrency uses split-local repeated-patient encounter counts because explicit temporal ordering is unavailable in split files.
- complex_discharge_flag assumes discharge disposition IDs {1, 6, 8} represent home-like discharges.
- medication_change_ratio uses diabetes medication-status columns detected from train split and applied consistently to val/test.

## Limitations

- recurrency is a deterministic proxy and does not claim encounter chronology.
- age_bucket_risk uses coarse decade-level bins and does not model nonlinear age effects yet.
- no learned scaling or encoding is applied at this stage by design.

## Fallback Behavior

- train: medication_change_ratio_zero_denominator=0, complex_discharge_flag_missing_or_malformed=0, age_bucket_risk_missing_or_malformed=0
- val: medication_change_ratio_zero_denominator=0, complex_discharge_flag_missing_or_malformed=0, age_bucket_risk_missing_or_malformed=0
- test: medication_change_ratio_zero_denominator=0, complex_discharge_flag_missing_or_malformed=0, age_bucket_risk_missing_or_malformed=0

## Engineered Feature Stats (train)

| Feature | Mean | Std | Min | Max |
| --- | --- | --- | --- | --- |
| recurrency | 1.2907 | 2.5330 | 0.0000 | 39.0000 |
| patient_severity | 0.2303 | 0.0868 | 0.0371 | 0.7406 |
| medication_change_ratio | 0.0125 | 0.0211 | 0.0000 | 0.1739 |
| utilization_intensity | 1.2170 | 2.3247 | 0.0000 | 80.0000 |
| complex_discharge_flag | 0.2811 | 0.4496 | 0.0000 | 1.0000 |
| age_bucket_risk | 6.1015 | 1.5990 | 0.0000 | 9.0000 |

## Engineered Feature Missingness (train)

| Feature | Missing Count | Missing Rate |
| --- | --- | --- |
| recurrency | 0 | 0.00% |
| patient_severity | 0 | 0.00% |
| medication_change_ratio | 0 | 0.00% |
| utilization_intensity | 0 | 0.00% |
| complex_discharge_flag | 0 | 0.00% |
| age_bucket_risk | 0 | 0.00% |

## Engineered Feature Stats (val)

| Feature | Mean | Std | Min | Max |
| --- | --- | --- | --- | --- |
| recurrency | 1.2017 | 2.2025 | 0.0000 | 20.0000 |
| patient_severity | 0.2302 | 0.0870 | 0.0371 | 0.6413 |
| medication_change_ratio | 0.0128 | 0.0215 | 0.0000 | 0.1304 |
| utilization_intensity | 1.2036 | 2.2906 | 0.0000 | 42.0000 |
| complex_discharge_flag | 0.2829 | 0.4504 | 0.0000 | 1.0000 |
| age_bucket_risk | 6.0727 | 1.5912 | 0.0000 | 9.0000 |

## Engineered Feature Missingness (val)

| Feature | Missing Count | Missing Rate |
| --- | --- | --- |
| recurrency | 0 | 0.00% |
| patient_severity | 0 | 0.00% |
| medication_change_ratio | 0 | 0.00% |
| utilization_intensity | 0 | 0.00% |
| complex_discharge_flag | 0 | 0.00% |
| age_bucket_risk | 0 | 0.00% |

## Engineered Feature Stats (test)

| Feature | Mean | Std | Min | Max |
| --- | --- | --- | --- | --- |
| recurrency | 1.1665 | 2.2185 | 0.0000 | 22.0000 |
| patient_severity | 0.2290 | 0.0860 | 0.0371 | 0.7021 |
| medication_change_ratio | 0.0124 | 0.0212 | 0.0000 | 0.1739 |
| utilization_intensity | 1.1339 | 2.1282 | 0.0000 | 29.0000 |
| complex_discharge_flag | 0.2734 | 0.4457 | 0.0000 | 1.0000 |
| age_bucket_risk | 6.0983 | 1.5733 | 0.0000 | 9.0000 |

## Engineered Feature Missingness (test)

| Feature | Missing Count | Missing Rate |
| --- | --- | --- |
| recurrency | 0 | 0.00% |
| patient_severity | 0 | 0.00% |
| medication_change_ratio | 0 | 0.00% |
| utilization_intensity | 0 | 0.00% |
| complex_discharge_flag | 0 | 0.00% |
| age_bucket_risk | 0 | 0.00% |

## Recommended Next Step For Modeling

- Train baseline models with model_candidate_columns and compare against ablations that remove engineered features.
