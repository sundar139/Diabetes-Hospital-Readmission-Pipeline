# Diabetes Hospital Readmission Pipeline

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/api-FastAPI-009688)
![MLflow](https://img.shields.io/badge/mlflow-SQLite%20tracking%20server-0A7DB8)
![Ruff](https://img.shields.io/badge/lint-ruff-46A2F1)
![Pytest](https://img.shields.io/badge/tests-pytest-0A9EDC)

Production-minded, local-first machine learning pipeline for diabetic patient readmission prediction.

This repository demonstrates a complete workflow from raw data validation to model training, explainable serving, monitoring, packaging, and CI. The stack is intentionally local and reproducible, with no fake cloud claims.

## Problem Overview

Hospital readmissions are expensive for health systems and stressful for patients. Early readmission risk estimation helps prioritize case management, discharge planning, and follow-up resources.

This project models:

- Binary early-readmission risk (`readmitted_30d`, where `<30` is positive)
- Multiclass readmission category (`NO`, `>30`, `<30`)

## Why This Matters

Readmission prediction is a practical healthcare ML use case with real operational impact:

- Supports targeted interventions for high-risk discharges
- Improves resource allocation in care coordination teams
- Encourages transparent, monitorable model operations rather than one-off notebooks

## Dataset

Primary source:

- UCI / Kaggle Diabetes 130-US hospitals dataset

Expected local dataset path:

- `data/raw/diabetic_data.csv`

The repository does not auto-download external data. Place the dataset locally before running pipeline scripts.

## Architecture Summary

- `src/config`: typed settings and path/URI resolution
- `src/data`: raw loading, schema checks, preprocessing, grouped split
- `src/features`: clinical feature engineering
- `src/models`: estimator factory, training, evaluation, prediction helpers
- `src/serving`: FastAPI app, prediction/explanation contracts
- `src/llm`: Ollama-backed explanation helpers with deterministic fallback
- `src/monitoring`: local drift and monitoring summary utilities
- `scripts`: operational entry points
- `tests`: unit and smoke coverage

## Repository Structure

```text
.
|-- .github/workflows/ci.yml
|-- artifacts/
|-- data/
|   |-- processed/
|   `-- raw/
|-- reports/
|-- scripts/
|   |-- build_processed_data.py
|   |-- build_feature_sets.py
|   |-- run_api.py
|   |-- run_evaluation.py
|   |-- run_mlflow_server.py
|   |-- run_monitoring_report.py
|   |-- run_raw_validation.py
|   |-- reset_mlflow_dev_store.py
|   |-- train_binary.py
|   `-- train_multiclass.py
|-- src/
|   |-- config/
|   |-- data/
|   |-- features/
|   |-- llm/
|   |-- models/
|   |-- monitoring/
|   `-- serving/
|-- tests/
|-- .dockerignore
|-- .env.example
|-- Dockerfile
|-- pyproject.toml
`-- README.md
```

## Setup

1. Install uv

```powershell
pip install uv
```

2. Install dependencies

```powershell
uv sync --group dev --extra eda
```

3. Create local environment file

```powershell
Copy-Item .env.example .env
```

4. Place dataset at `data/raw/diabetic_data.csv`

## Raw Validation Flow

Validate raw data quality and schema assumptions before transformation:

```powershell
uv run python scripts/run_raw_validation.py
```

Generated outputs include:

- `reports/raw_validation_report.md`
- `reports/raw_validation_summary.json`
- `reports/data_dictionary.md`
- `artifacts/raw_validation_summary.json`

## Preprocessing and Grouped Splitting Flow

Build cleaned data and leakage-aware patient-level split:

```powershell
uv run python scripts/build_processed_data.py
```

Generated outputs:

- `data/processed/train.parquet`
- `data/processed/val.parquet`
- `data/processed/test.parquet`
- `artifacts/split_manifest.json`
- `reports/processed_data_report.md`

## Feature Engineering Summary

Construct clinically oriented features:

```powershell
uv run python scripts/build_feature_sets.py
```

Engineered outputs:

- `data/processed/train_features.parquet`
- `data/processed/val_features.parquet`
- `data/processed/test_features.parquet`
- `artifacts/feature_metadata.json`
- `reports/feature_engineering_report.md`

## Training and Evaluation

Train binary models:

```powershell
uv run python scripts/train_binary.py
```

Train multiclass models:

```powershell
uv run python scripts/train_multiclass.py
```

Run saved-model evaluation:

```powershell
uv run python scripts/run_evaluation.py
```

Core artifacts:

- `artifacts/binary_model.joblib`
- `artifacts/binary_model_metadata.json`
- `artifacts/binary_training_results.json`
- `artifacts/multiclass_model.joblib`
- `artifacts/multiclass_model_metadata.json`
- `artifacts/multiclass_training_results.json`
- `reports/model_comparison_report.md`

## MLflow Local Workflow (SQLite-Backed)

Start MLflow server:

```powershell
uv run python scripts/run_mlflow_server.py
```

Default local setup:

- Tracking URI (client): `http://127.0.0.1:5000`
- Backend metadata: `sqlite:///mlflow.db`
- Artifact destination (configured): `./mlartifacts`
- Artifact destination (server-ready): normalized to `file:///.../mlartifacts`
- Server workers: `1` in Windows local mode for startup stability

Why this matters on Windows:

- Using a raw path like `C:\...\mlartifacts` can break artifact uploads with artifact repository resolution errors.
- This project normalizes local artifact destinations to file URI format before launching `mlflow server`.
- Windows startup can be unstable with multi-worker MLflow server mode; this project forces single-worker launch mode locally to avoid intermittent socket startup errors such as WinError 10022.

If earlier runs were created under broken artifact destination metadata, reset local dev store:

```powershell
uv run python scripts/reset_mlflow_dev_store.py --yes
```

This reset deletes only:

- `mlflow.db`
- `mlartifacts/`

It does not delete:

- `artifacts/`
- `data/processed/`

## API Usage (FastAPI)

Start API:

```powershell
uv run python scripts/run_api.py
```

Docs and OpenAPI:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/openapi.json`

Endpoints:

- `GET /health`
- `POST /predict`
- `POST /predict-batch`
- `POST /explain`

Example payload files:

- `artifacts/sample_payload.json` for `POST /predict`
- `artifacts/sample_batch_payload.json` for `POST /predict-batch`
- `artifacts/sample_explain_payload.json` for `POST /explain`

Run local demo API requests and persist real response examples:

```powershell
uv run python scripts/demo_prediction_examples.py --mode all
```

## Ollama Explanation Workflow

Optional local Ollama setup:

```powershell
ollama pull llama3.1:8b
ollama serve
```

Explanation behavior:

- `/explain` prefers Ollama when requested
- falls back deterministically when Ollama is unavailable
- output is non-diagnostic and for model transparency only

## Monitoring and Drift Reporting

Generate local monitoring outputs:

```powershell
uv run python scripts/run_monitoring_report.py
```

Optional Ollama narrative summary for monitoring:

```powershell
uv run python scripts/run_monitoring_report.py --prefer-ollama
```

Monitoring records are stored locally as JSONL:

- `artifacts/monitoring/prediction_log.jsonl`

Monitoring outputs:

- `reports/monitoring_summary.json`
- `reports/monitoring_report.md`

Monitoring includes:

- model version metadata from saved model metadata files
- prediction distribution summary
- binary probability summary and drift PSI
- numeric feature drift summary (PSI-based)
- explicit warnings for small sample sizes or missing reference data
- explicit statement when labels are unavailable (no fake live performance metrics)

## XGBoost Device Selection

Config key:

- `PIPELINE_XGBOOST_DEVICE` in `{auto, cuda, cpu}`

Behavior:

- `auto`: attempts CUDA for XGBoost, falls back to CPU with warning
- `cuda`: requests CUDA, falls back safely if unavailable
- `cpu`: forces CPU

Deterministic runtime behavior:

- XGBoost can train on CUDA when available and requested.
- Inference for sklearn/imblearn pipeline outputs is pinned to CPU for XGBoost to avoid device mismatch warnings from CPU-resident transformed matrices.
- This is intentional and persisted in metadata/results via:
	- `xgboost_device_requested`
	- `xgboost_device_used_for_training`
	- `xgboost_device_used_for_inference`
	- `xgboost_inference_used_fallback_path`
- The same inference runtime logic is used across evaluation, API predictions, and monitoring report generation.

Notes:

- Device selection applies only to XGBoost
- LogisticRegression and RandomForest remain CPU baselines
- runtime device usage is captured in training metadata/results and MLflow params/tags

## Docker Usage

Build image:

```powershell
docker build -t diabetes-readmission-api .
```

Run container:

```powershell
docker run --rm -p 8000:8000 diabetes-readmission-api
```

Container docs URL:

- `http://127.0.0.1:8000/docs`

## CI

GitHub Actions workflow:

- `.github/workflows/ci.yml`

CI runs on push and pull request and executes:

- `uv run ruff check .`
- `uv run pytest`
- `uv run python scripts/healthcheck.py`
- FastAPI import smoke check

## Local Demo Flow

Terminal 1 (MLflow server):

```powershell
uv run python scripts/run_mlflow_server.py
```

Terminal 2 (API server):

```powershell
uv run python scripts/run_api.py
```

Terminal 3 (demo requests and showcase artifacts):

```powershell
uv run python scripts/demo_smoke_run.py
```

If models were not retrained in the current workspace session, run once before demo smoke run:

```powershell
uv run python scripts/train_binary.py
uv run python scripts/train_multiclass.py
uv run python scripts/run_evaluation.py
```

Generated demo artifacts:

- `artifacts/demo/sample_health_response.json`
- `artifacts/demo/sample_prediction_response.json`
- `artifacts/demo/sample_batch_response.json`
- `artifacts/demo/sample_explanation_response.json`
- `artifacts/demo/demo_manifest.json`
- `reports/monitoring_summary.json`
- `reports/monitoring_report.md`
- `reports/demo_summary.md`

Local URLs:

- MLflow UI: `http://127.0.0.1:5000`
- FastAPI Docs: `http://127.0.0.1:8000/docs`

Demo checklist:

- Show `/health` response from `artifacts/demo/sample_health_response.json`
- Show `/predict` and `/predict-batch` response examples from `artifacts/demo/`
- Show `/explain` response and explanation mode from `artifacts/demo/sample_explanation_response.json`
- Show drift and runtime notes in `reports/monitoring_report.md`
- Show one-page recap in `reports/demo_summary.md`

## Limitations

- Trained on a single public tabular dataset without external validation cohorts
- Monitoring is local and batch-style, not a streaming production service
- Label-latency handling and continuous performance backfill are not implemented
- No cloud deployment in this repository by design

## Future Improvements

- Automated scheduled monitoring runs and trend snapshots
- Data quality contracts for serving-time payload validation
- Expanded explainability with calibrated uncertainty reporting
- Stronger CI matrix (multiple Python versions and platform checks)

## Resume-Ready Highlights

- Designed and implemented end-to-end healthcare ML pipeline with reproducible local operations
- Added leakage-aware grouped splits, feature engineering, model selection, and evaluation reporting
- Built FastAPI inference service with batch and explanation endpoints plus graceful LLM fallback
- Migrated MLflow to SQLite-backed tracking server and fixed Windows artifact URI handling
- Added drift monitoring reports, Docker packaging, and CI automation for demo-ready delivery
