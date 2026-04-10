# Troubleshooting

This guide covers common local issues and recovery steps for Windows-first usage.

## MLflow Local Reset and Startup

Symptoms:

- MLflow UI shows old/broken runs after earlier artifact URI configuration changes
- artifact logging paths look inconsistent across runs

Reset local MLflow state:

```powershell
uv run python scripts/reset_mlflow_dev_store.py --yes
```

Then restart MLflow server:

```powershell
uv run python scripts/run_mlflow_server.py
```

Notes:

- reset removes `mlflow.db` and `mlartifacts`
- model/data artifacts in `artifacts` and `data/processed` are not removed

## API Startup Issues

Symptoms:

- request scripts fail to connect to `http://127.0.0.1:8000`
- `demo_prediction_examples.py` reports connection error

Checks:

1. Start API server:

```powershell
uv run python scripts/run_api.py
```

1. Confirm health endpoint:

```powershell
uv run python scripts/demo_prediction_examples.py --mode health
```

1. Confirm no other process is using port 8000. If needed, set `PIPELINE_API_PORT` in `.env` and restart.

## Ollama Unavailable Fallback Behavior

Symptoms:

- explanation or monitoring summary does not use Ollama output
- response mode indicates fallback

Expected behavior:

- `/explain` and monitoring narrative logic are designed to fall back deterministically when Ollama is unavailable or times out
- this is intentional local reliability behavior, not an error

If you want Ollama mode:

```powershell
ollama pull llama3.1:8b
ollama serve
```

Then rerun explain or monitoring with Ollama-preferred options.

## Windows Notes

- MLflow startup uses single-worker launch mode in Windows local usage to improve stability.
- Keep paths under the project root short and avoid moving the workspace while servers are running.
- Use local host bindings (`127.0.0.1`) for MLflow and API unless you intentionally need external access.

## XGBoost Device Notes

- `PIPELINE_XGBOOST_DEVICE` controls requested training device (`auto`, `cuda`, `cpu`).
- Training can use CUDA when available.
- Inference is pinned to CPU for deterministic behavior with CPU-resident transformed feature matrices.
- Device/runtime fields are recorded in model metadata, evaluation outputs, and monitoring summaries.

## Monitoring Generation Issues

Symptoms:

- monitoring script fails due to missing model/data artifacts

Checks:

1. Ensure trained models exist:

- `artifacts/binary_model.joblib`
- `artifacts/multiclass_model.joblib`

1. Ensure feature sets exist:

- `data/processed/val_features.parquet`
- `data/processed/test_features.parquet`

1. Re-run training/evaluation before monitoring if needed.

## Demo Script Prerequisite Errors

`demo_smoke_run.py` validates required model and payload files before running.

If it reports missing files, run:

```powershell
uv run python scripts/train_binary.py
uv run python scripts/train_multiclass.py
uv run python scripts/run_evaluation.py
```

Then re-run demo smoke flow.

## Streamlit Frontend Issues

Symptoms:

- Streamlit page reports missing artifacts
- prediction page is disabled in deployed app

Checks:

1. Confirm required files exist in repository:

- `artifacts/binary_model.joblib`
- `artifacts/multiclass_model.joblib`
- `artifacts/binary_model_metadata.json`
- `artifacts/multiclass_model_metadata.json`

1. Confirm app entrypoint is correct for deployment:

- `streamlit_app.py`

1. Confirm dependencies are installable by platform:

- `requirements.txt` contains `.`
- `pyproject.toml` contains `streamlit` dependency

1. Run local logic check:

```powershell
uv run python scripts/test_streamlit_frontend_logic.py
```

If this command succeeds locally, artifact loading and prediction helpers are healthy.

## Streamlit and FastAPI/Ollama Scope

- Public Streamlit flow does not call local FastAPI.
- Public Streamlit flow does not require Ollama.
- FastAPI and Ollama remain optional local advanced workflows.
