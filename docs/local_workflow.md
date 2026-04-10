# Local Workflow

This guide is the operator-oriented local workflow for preparing, training, serving, monitoring, and demonstrating the project on Windows.

## 1. Environment Setup

```powershell
pip install uv
uv sync --group dev --extra eda
Copy-Item .env.example .env
```

Dataset location:

- `data/raw/diabetic_data.csv`

Quick environment check:

```powershell
uv run python scripts/healthcheck.py
```

## 2. Data Preparation

```powershell
uv run python scripts/run_raw_validation.py
uv run python scripts/build_processed_data.py
uv run python scripts/build_feature_sets.py
```

Primary outputs:

- `reports/raw_validation_report.md`
- `reports/processed_data_report.md`
- `reports/feature_engineering_report.md`
- `artifacts/raw_validation_summary.json`
- `artifacts/split_manifest.json`
- `artifacts/feature_metadata.json`

## 3. Training and Evaluation

```powershell
uv run python scripts/train_binary.py
uv run python scripts/train_multiclass.py
uv run python scripts/run_evaluation.py
```

Primary outputs:

- `artifacts/binary_model.joblib`
- `artifacts/multiclass_model.joblib`
- `artifacts/binary_model_metadata.json`
- `artifacts/multiclass_model_metadata.json`
- `reports/model_comparison_report.md`

## 4. Local MLflow Tracking

Start server:

```powershell
uv run python scripts/run_mlflow_server.py
```

Default URL:

- `http://127.0.0.1:5000`

If local run metadata/artifacts need reset:

```powershell
uv run python scripts/reset_mlflow_dev_store.py --yes
```

## 5. API Serving

Start API server:

```powershell
uv run python scripts/run_api.py
```

Docs URL:

- `http://127.0.0.1:8000/docs`

Sample request payloads:

- `artifacts/sample_payload.json`
- `artifacts/sample_batch_payload.json`
- `artifacts/sample_explain_payload.json`

## 6. Monitoring

```powershell
uv run python scripts/run_monitoring_report.py
```

Outputs:

- `reports/monitoring_summary.json`
- `reports/monitoring_report.md`
- `artifacts/monitoring/prediction_log.jsonl`

## 7. Demo Artifact Generation

Generate endpoint response examples and a one-page demo summary:

```powershell
uv run python scripts/demo_smoke_run.py
```

Outputs:

- `artifacts/demo/sample_health_response.json`
- `artifacts/demo/sample_prediction_response.json`
- `artifacts/demo/sample_batch_response.json`
- `artifacts/demo/sample_explanation_response.json`
- `artifacts/demo/demo_requests_manifest.json`
- `artifacts/demo/demo_manifest.json`
- `reports/demo_summary.md`

## 8. Validation

```powershell
uv run ruff check .
uv run pytest
```
