# Diabetes Hospital Readmission Pipeline

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/api-FastAPI-009688)
![MLflow](https://img.shields.io/badge/mlflow-local%20sqlite-0A7DB8)
![Docker](https://img.shields.io/badge/docker-supported-2496ED)
![CI](https://img.shields.io/badge/ci-github%20actions-2088FF)
![Ruff](https://img.shields.io/badge/lint-ruff-46A2F1)
![Pytest](https://img.shields.io/badge/tests-pytest-0A9EDC)

Local-first, production-style machine learning pipeline for diabetes hospital readmission prediction, with reproducible training, evaluation, serving, monitoring, and demo workflows.

## Project Summary

This repository packages a complete tabular ML lifecycle:

- raw data validation and schema checks
- leakage-aware preprocessing and splitting
- feature engineering with clinically motivated fields
- model selection across multiple estimator families
- local MLflow tracking
- FastAPI prediction and explanation endpoints
- Streamlit portfolio frontend with artifact-direct predictions
- drift-oriented monitoring reports
- demo artifact generation for portfolio presentation

The implementation is designed for technical transparency and reproducibility in a local Windows workflow.

## Business and Clinical Motivation

Hospital readmissions increase cost and care complexity. Early identification of patients at higher risk can support discharge planning and follow-up prioritization.

This project is an educational and operational demonstration of ML workflow quality, not a clinical decision system. Outputs must not be interpreted as medical advice.

## Architecture Overview

Core modules:

- `src/config`: typed environment settings and URI/path resolution
- `src/data`: raw loading, validation, preprocessing, leakage-aware split logic
- `src/features`: engineered features and metadata
- `src/models`: estimator factory, training, evaluation, prediction helpers
- `src/serving`: FastAPI app and request/response schemas
- `src/llm`: explanation generation with Ollama-first, deterministic fallback
- `src/monitoring`: prediction logging and drift/monitoring reports
- `src/frontend`: Streamlit UI modules for prediction, analytics, monitoring, and overview

Operational entry points:

- `scripts`: training, evaluation, API, MLflow, monitoring, demo workflows
- `tests`: unit and smoke checks for core behaviors

## Dataset and Task Framing

Dataset source:

- UCI/Kaggle Diabetes 130-US hospitals data

Expected local dataset location:

- `data/raw/diabetic_data.csv`

Prediction tasks:

- Binary task: `readmitted_30d` (positive class `<30`)
- Multiclass task: `readmitted` with labels `NO`, `>30`, `<30`

## Preprocessing and Leakage Prevention

The pipeline enforces a patient-level grouping strategy during splitting to reduce leakage from repeated encounters. Preprocessing and split manifests are persisted so data lineage can be audited.

Primary commands:

```powershell
uv run python scripts/run_raw_validation.py
uv run python scripts/build_processed_data.py
```

Primary outputs:

- `reports/raw_validation_report.md`
- `reports/raw_validation_summary.json`
- `reports/data_dictionary.md`
- `reports/processed_data_report.md`
- `artifacts/raw_validation_summary.json`
- `artifacts/split_manifest.json`

## Feature Engineering

Feature generation includes baseline tabular transforms plus project-specific risk/usage signals such as recurrency, patient severity, medication change ratio, utilization intensity, and discharge complexity features.

Command:

```powershell
uv run python scripts/build_feature_sets.py
```

Outputs:

- `data/processed/train_features.parquet`
- `data/processed/val_features.parquet`
- `data/processed/test_features.parquet`
- `artifacts/feature_metadata.json`
- `reports/feature_engineering_report.md`

## Modeling Summary

Train and evaluate:

```powershell
uv run python scripts/train_binary.py
uv run python scripts/train_multiclass.py
uv run python scripts/run_evaluation.py
```

Observed best models from latest comparison report (`reports/model_comparison_report.md`):

- Binary winner: XGBoost (sampling=none), test F1 = 0.3661
- Multiclass winner: XGBoost, test macro F1 = 0.5250

Key model artifacts:

- `artifacts/binary_model.joblib`
- `artifacts/binary_model_metadata.json`
- `artifacts/binary_training_results.json`
- `artifacts/multiclass_model.joblib`
- `artifacts/multiclass_model_metadata.json`
- `artifacts/multiclass_training_results.json`

## XGBoost Runtime Notes

Config:

- `PIPELINE_XGBOOST_DEVICE` in `auto`, `cuda`, `cpu`

Runtime behavior:

- training can use CUDA when available and requested
- inference is pinned to CPU for sklearn/imblearn pipeline outputs to keep behavior deterministic with CPU-resident transformed matrices
- runtime fields are persisted in metadata/results and surfaced in evaluation/monitoring outputs

## MLflow Local Workflow

Start local MLflow server:

```powershell
uv run python scripts/run_mlflow_server.py
```

Default local settings:

- tracking URI: `http://127.0.0.1:5000`
- backend store: `sqlite:///mlflow.db`
- artifacts destination: `./mlartifacts` (normalized to file URI)
- Windows local launch mode: single worker for startup stability

Reset local MLflow tracking state when needed:

```powershell
uv run python scripts/reset_mlflow_dev_store.py --yes
```

This removes `mlflow.db` and `mlartifacts` only.

## API Usage

Start API:

```powershell
uv run python scripts/run_api.py
```

Docs URL:

- `http://127.0.0.1:8000/docs`

Endpoints:

- `GET /health`
- `POST /predict`
- `POST /predict-batch`
- `POST /explain`

Sample payload files:

- `artifacts/sample_payload.json`
- `artifacts/sample_batch_payload.json`
- `artifacts/sample_explain_payload.json`

Run request examples and persist real responses:

```powershell
uv run python scripts/demo_prediction_examples.py --mode all
```

## Streamlit Frontend (Artifact-Direct)

The Streamlit app is designed for public portfolio demos and loads saved model artifacts directly.

Key behavior:

- no dependency on a running FastAPI server for predictions
- no dependency on local Ollama for explanation output
- deterministic explanation mode in public app flow
- uses saved files from `artifacts/` and report summaries from `reports/`

Run locally:

```powershell
uv run streamlit run streamlit_app.py
```

Expected required artifacts:

- `artifacts/binary_model.joblib`
- `artifacts/multiclass_model.joblib`
- `artifacts/binary_model_metadata.json`
- `artifacts/multiclass_model_metadata.json`

Optional but recommended frontend inputs:

- `artifacts/sample_payload.json`
- `artifacts/sample_explain_payload.json`
- `reports/model_comparison_report.md`
- `reports/monitoring_summary.json`
- `reports/monitoring_report.md`

Dummy example behavior:

- **Load baseline example**: seeds form values from `artifacts/sample_payload.json`
- **Load dummy example**: seeds form values from `artifacts/sample_explain_payload.json` (or batch fallback)

## Streamlit Community Cloud Deployment

Deployment target:

- app entrypoint: `streamlit_app.py`

Repository deployment notes:

- this repository uses a lightweight root `requirements.txt` for frontend deployment/runtime only
- lightweight frontend runtime dependencies include Streamlit, pandas, numpy, scikit-learn, XGBoost, and joblib
- full local development and training dependencies (including SHAP, MLflow, FastAPI, Boruta) remain in `pyproject.toml`
- install the full stack locally with `uv sync --group dev --extra eda`
- `.streamlit/config.toml` provides theme and server configuration
- `runtime.txt` pins Python runtime to `3.11`

CI notes:

- `.github/workflows/ci.yml` runs a lightweight frontend job on push/pull request using `requirements.txt`
- full-stack CI checks are preserved in the same workflow and run via manual `workflow_dispatch`

Community Cloud setup steps:

1. Push this repository to GitHub with required model artifacts included.
2. In Streamlit Community Cloud, create a new app from the GitHub repo.
3. Set `streamlit_app.py` as the main file path.
4. Deploy; no secrets are required for core public prediction flow.

Local-only advanced workflows remain available:

- FastAPI serving and request demos
- Ollama-preferred explanation/narrative generation
- MLflow local experiment tracking

## Monitoring Usage

Generate monitoring reports:

```powershell
uv run python scripts/run_monitoring_report.py
```

Outputs:

- `reports/monitoring_summary.json`
- `reports/monitoring_report.md`
- `artifacts/monitoring/prediction_log.jsonl`

Latest report currently shows stable probability drift with PSI 0.00243.

## Local Demo Flow

Terminal 1:

```powershell
uv run python scripts/run_mlflow_server.py
```

Terminal 2:

```powershell
uv run python scripts/run_api.py
```

Terminal 3:

```powershell
uv run python scripts/demo_smoke_run.py
```

If models are not yet trained in this workspace session:

```powershell
uv run python scripts/train_binary.py
uv run python scripts/train_multiclass.py
uv run python scripts/run_evaluation.py
```

Generated demo outputs:

- `artifacts/demo/sample_health_response.json`
- `artifacts/demo/sample_prediction_response.json`
- `artifacts/demo/sample_batch_response.json`
- `artifacts/demo/sample_explanation_response.json`
- `artifacts/demo/demo_requests_manifest.json`
- `artifacts/demo/demo_manifest.json`
- `reports/demo_summary.md`

Suggested demo narrative:

- show health and docs endpoint
- show single and batch prediction responses
- show explanation output and fallback behavior note
- show monitoring report drift/runtime section
- close with `reports/demo_summary.md`

## Docker Usage

Build:

```powershell
docker build -t diabetes-readmission-api .
```

Run:

```powershell
docker run --rm -p 8000:8000 diabetes-readmission-api
```

## CI Summary

Workflow file:

- `.github/workflows/ci.yml`

Current CI checks:

- `uv run ruff check .`
- `uv run pytest`
- `uv run python scripts/healthcheck.py`
- FastAPI import smoke check

## Supporting Docs

- `docs/local_workflow.md`
- `docs/results_summary.md`
- `docs/troubleshooting.md`
- `docs/release_checklist.md`

## Limitations

- single public dataset; no external validation cohort
- local batch-style monitoring, not streaming production telemetry
- no delayed-label backfill loop for live performance tracking
- public deployment target is Streamlit Community Cloud demo hosting (not production clinical deployment)

## Future Improvements

- scheduled monitoring trend snapshots
- stronger data contracts for serving payload quality checks
- expanded uncertainty and calibration analysis
- broader CI matrix across OS/Python versions

## Resume-Ready Highlights

- built an end-to-end local healthcare ML pipeline with reproducible artifacts and reports
- implemented leakage-aware splitting, engineered features, and comparative model selection
- delivered FastAPI serving with batch and explanation workflows and deterministic fallback behavior
- hardened local MLflow and XGBoost runtime behavior for Windows stability
- added drift reporting, demo artifact generation, Docker packaging, and CI checks
