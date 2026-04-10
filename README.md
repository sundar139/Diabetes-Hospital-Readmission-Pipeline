# End-to-End Hospital Readmission Prediction for Diabetic Patients

This repository provides a production-style foundation for a tabular machine learning project with MLOps and LLMOps support. The project targets hospital readmission prediction for diabetic patients and is organized for clean iteration from data ingestion to model serving.

## Project Overview

- Problem scope: diabetic patient readmission prediction
- Initial tasks: robust project scaffold, configuration, quality gates, and utility scripts
- Planned serving stack: FastAPI for model APIs and Ollama-backed explanation endpoints
- Experiment tracking: MLflow
- Package management: uv

## Architecture Summary

- `src/config`: typed runtime settings and path management
- `src/data`: data ingestion and validation components
- `src/features`: feature engineering and transformation modules
- `src/models`: training and inference logic
- `src/serving`: API schemas and service handlers
- `src/monitoring`: model and data monitoring logic
- `src/llm`: natural-language explanation workflows
- `scripts`: health and diagnostics utilities
- `tests`: smoke and unit tests

## Dataset

Primary dataset source:

- UCI/Kaggle Diabetes 130-US hospitals dataset

Expected local path:

- `data/raw/diabetic_data.csv`

This scaffold assumes the raw dataset is stored locally and does not include dataset download automation yet.

## Setup

1. Install uv.

```powershell
pip install uv
```

1. Sync dependencies, including dev tools and optional EDA extras.

```powershell
uv sync --group dev --extra eda
```

1. Create a local environment file.

```powershell
Copy-Item .env.example .env
```

1. Place dataset at `data/raw/diabetic_data.csv`.

## Raw Validation Before Preprocessing

Run raw-data validation and reporting:

```powershell
uv run python scripts/run_raw_validation.py
```

Why this step exists:

- verifies schema assumptions before any transformations
- quantifies missing and null-like values (`?`, empty strings, whitespace-only tokens, and standard nulls)
- surfaces identifier and target integrity risks early
- creates deterministic artifacts for reproducibility and review

Generated artifacts:

- `reports/raw_validation_report.md`
- `reports/raw_validation_summary.json`
- `reports/data_dictionary.md`
- `artifacts/raw_validation_summary.json`
- `reports/figures/readmitted_class_distribution.png` (generated when matplotlib is available)

## Processed Data Build

Run preprocessing, binary-target derivation, and grouped split:

```powershell
uv run python scripts/build_processed_data.py
```

Binary target derivation:

- source target: `readmitted`
- derived target: `readmitted_30d`
- mapping: `<30 -> 1`, `NO -> 0`, `>30 -> 0`

Patient-level leakage prevention:

- split strategy uses grouped splitting by `patient_nbr`
- each patient appears in exactly one split (train, val, or test)
- `encounter_id` and `patient_nbr` are excluded from model feature candidates

Generated outputs:

- `data/processed/train.parquet`
- `data/processed/val.parquet`
- `data/processed/test.parquet`
- `artifacts/split_manifest.json`
- `reports/processed_data_report.md`

## Clinical Feature Engineering

Run feature set construction:

```powershell
uv run python scripts/build_feature_sets.py
```

Engineered features:

- `recurrency`: split-local repeated patient encounter count proxy (`encounter_count_for_patient - 1`)
- `patient_severity`: weighted severity index from stay length, diagnoses, and prior utilization
- `medication_change_ratio`: share of diabetes medication statuses in `{Up, Down}` per encounter
- `utilization_intensity`: `number_inpatient + number_outpatient + number_emergency`
- `complex_discharge_flag`: indicator for non-home-like discharge disposition
- `age_bucket_risk`: ordinal risk derived from age bucket lower bound

Why these features matter clinically:

- they capture prior utilization burden and return-visit tendencies
- they summarize acuity and care complexity into model-ready signals
- they represent treatment adjustment intensity and transition-of-care difficulty
- they inject age-associated baseline vulnerability in transparent form

Generated outputs:

- `data/processed/train_features.parquet`
- `data/processed/val_features.parquet`
- `data/processed/test_features.parquet`
- `artifacts/feature_metadata.json`
- `reports/feature_engineering_report.md`

## Model Training and Evaluation

Two prediction tasks are supported:

- binary early-readmission task (`readmitted_30d`)
- multiclass readmission task (`readmitted` with classes `NO`, `>30`, `<30`)

Model strategy:

- baselines: LogisticRegression and RandomForestClassifier
- primary model: XGBClassifier
- optional binary sampling strategies: `none`, `over`, `under`
- optional feature selection strategy: `none`, `boruta` (gracefully skipped if unavailable)

Train binary models:

```powershell
uv run python scripts/train_binary.py
```

Train multiclass models:

```powershell
uv run python scripts/train_multiclass.py
```

Run final saved-model evaluation and generate comparison report:

```powershell
uv run python scripts/run_evaluation.py
```

Launch MLflow UI locally:

```powershell
uv run mlflow ui --backend-store-uri ./mlruns
```

Primary modeling artifacts:

- `artifacts/binary_model.joblib`
- `artifacts/binary_model_metadata.json`
- `artifacts/binary_training_results.json`
- `artifacts/multiclass_model.joblib`
- `artifacts/multiclass_model_metadata.json`
- `artifacts/multiclass_training_results.json`
- `artifacts/evaluations/binary/`
- `artifacts/evaluations/multiclass/`
- `reports/model_comparison_report.md`

Interpretability outputs:

- tree-based runs generate SHAP summary JSON and optional plots under run-level evaluation folders
- single-row contribution utilities are available in `src/models/predict.py`
- SHAP is optional in the default path and may be unavailable on Python 3.13 environments

## Cross-Platform Command Reference

Run lint:

```powershell
uv run ruff check .
```

Run tests:

```powershell
uv run pytest
```

Run healthcheck:

```powershell
uv run python scripts/healthcheck.py
```

Run raw validation:

```powershell
uv run python scripts/run_raw_validation.py
```

Build processed datasets:

```powershell
uv run python scripts/build_processed_data.py
```

Build clinical feature sets:

```powershell
uv run python scripts/build_feature_sets.py
```

Train binary models:

```powershell
uv run python scripts/train_binary.py
```

Train multiclass models:

```powershell
uv run python scripts/train_multiclass.py
```

Run final evaluation and model comparison report:

```powershell
uv run python scripts/run_evaluation.py
```

Run MLflow UI:

```powershell
uv run mlflow ui --backend-store-uri ./mlruns
```

Print project paths and active config:

```powershell
uv run python scripts/print_tree.py
```

## Planned Pipeline Phases

1. Data ingestion and schema validation
2. Data cleaning and target construction (`NO`, `>30`, `<30` and binary 30-day target)
3. Feature engineering and leakage prevention
4. Baseline model training and evaluation
5. Imbalance handling and feature selection experiments
6. Experiment tracking and model registry with MLflow
7. FastAPI inference endpoints for batch and online predictions
8. Ollama-based explanation endpoint for natural-language rationale
9. Monitoring hooks for drift and service health

## Repository Structure

```text
.
|-- artifacts/
|-- data/
|   |-- processed/
|   `-- raw/
|-- notebooks/
|-- reports/
|   `-- figures/
|-- scripts/
|   |-- healthcheck.py
|   `-- print_tree.py
|-- src/
|   |-- config/
|   |   `-- settings.py
|   |-- data/
|   |-- features/
|   |-- llm/
|   |-- models/
|   |-- monitoring/
|   `-- serving/
|-- tests/
|   `-- test_smoke_imports.py
|-- .env.example
|-- .gitignore
|-- pyproject.toml
`-- README.md
```

## Notes

- The repository currently focuses on clean scaffolding and developer ergonomics.
- Model training, feature logic, API handlers, and LLM explanation orchestration will be added in subsequent phases.
