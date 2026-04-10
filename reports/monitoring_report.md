# Monitoring Report

Generated at (UTC): 2026-04-10T04:12:48.206909+00:00

## Model Version

- Binary model family: xgboost
- Binary training timestamp: 2026-04-10T03:31:39.462434+00:00
- Multiclass model family: xgboost
- Multiclass training timestamp: 2026-04-10T03:50:12.158080+00:00

## Sample Sizes

- Reference rows: 5000
- Current feature rows: 5000
- Prediction records: 5000

## Prediction Distribution

- Binary counts: {"0": 3256, "1": 1744}
- Multiclass counts: {"<30": 34, ">30": 1710, "NO": 3256}

## Binary Probability Summary

- Mean: 0.37229757674634456
- Std: 0.24301805220114023
- Min: 0.0148596977815032
- Max: 0.9195299744606018

## Drift Checks

- Binary probability drift status: stable
- Binary probability PSI: 0.002432362862340654

## Inference Runtime

- binary_model: requested=auto, training=cuda, inference=cpu, fallback_path=True
- multiclass_model: requested=auto, training=cuda, inference=cpu, fallback_path=True

### Feature Drift

- age_bucket_risk: status=stable, psi=0.006518621434331657
- complex_discharge_flag: status=stable, psi=0.0
- medication_change_ratio: status=stable, psi=0.0001981004942454904
- num_medications: status=stable, psi=0.005016086247528025
- number_diagnoses: status=stable, psi=0.0003836264034891352
- patient_severity: status=stable, psi=0.0021979060528009887
- recurrency: status=stable, psi=0.0005371152606493759
- time_in_hospital: status=stable, psi=0.004729828101458827
- utilization_intensity: status=stable, psi=6.5718516753963e-05

## Label Availability

- Labels available: True
- true_label count: 5000
- true_label_30d count: 5000
- Performance monitoring: labels_present_but_not_scored_in_this_report

## Warnings

- binary_model is using CPU-compatible inference path for device stability.
- multiclass_model is using CPU-compatible inference path for device stability.

## Optional Narrative Summary

Mode: fallback

Monitoring summary generated for local model operations. Compared 5000 prediction records against 5000 reference rows. Binary-probability drift status is 'stable'. Inference fallback models: 2. Warnings reported: 2.
