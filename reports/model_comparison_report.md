# Model Comparison Report

## Task Framing

- Binary objective predicts early readmission risk (`readmitted_30d`).
- Multiclass objective predicts full horizon readmission category (`readmitted`: NO, >30, <30).

## Models Trained

- Binary candidates: logistic_regression (sampling=none), logistic_regression (sampling=over), logistic_regression (sampling=under), random_forest (sampling=none), random_forest (sampling=over), random_forest (sampling=under), xgboost (sampling=none), xgboost (sampling=over), xgboost (sampling=under)
- Multiclass candidates: logistic_regression (sampling=none), random_forest (sampling=none), xgboost (sampling=none)

## Best Model Per Task

- Binary best: xgboost (sampling=none).
- Multiclass best: xgboost.

## Key Metrics Comparison

| Task | Best Model | Primary Metric | Value |
| --- | --- | --- | --- |
| Binary (readmitted_30d) | xgboost | F1 | 0.3661 |
| Multiclass (readmitted) | xgboost | Macro F1 | 0.5250 |

## Binary Imbalance Handling Impact

- Sampling best test F1=0.3645 vs no-sampling best test F1=0.3661; improved=False.

## Feature Selection Usage

- Optional feature selection used: False.
- Binary feature selection strategy: none.
- Multiclass feature selection strategy: none.

## Interpretability Artifacts

- Binary SHAP artifacts: generated.
- Multiclass SHAP artifacts: generated.

## Recommended Production Candidate

- binary/xgboost for early-intervention workflows, with multiclass/xgboost as secondary triage support.
