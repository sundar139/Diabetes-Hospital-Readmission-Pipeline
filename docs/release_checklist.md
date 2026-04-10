# Release Checklist

Use this checklist before tagging or presenting the repository as a final local release.

## Environment and Data

- [ ] `.env` created from `.env.example`
- [ ] Dependencies installed (`uv sync --group dev --extra eda`)
- [ ] Dataset available at `data/raw/diabetic_data.csv`
- [ ] `uv run python scripts/healthcheck.py` runs successfully

## Data Pipeline

- [ ] `uv run python scripts/run_raw_validation.py`
- [ ] `uv run python scripts/build_processed_data.py`
- [ ] `uv run python scripts/build_feature_sets.py`
- [ ] Reports/artifacts created in expected `reports` and `artifacts` paths

## Training and Evaluation

- [ ] `uv run python scripts/train_binary.py`
- [ ] `uv run python scripts/train_multiclass.py`
- [ ] `uv run python scripts/run_evaluation.py`
- [ ] `reports/model_comparison_report.md` refreshed

## MLflow Validation

- [ ] `uv run python scripts/run_mlflow_server.py` starts successfully
- [ ] MLflow UI reachable at `http://127.0.0.1:5000`
- [ ] Latest training runs visible in local experiment

## API Validation

- [ ] `uv run python scripts/run_api.py` starts successfully
- [ ] Docs reachable at `http://127.0.0.1:8000/docs`
- [ ] `uv run python scripts/demo_prediction_examples.py --mode all` succeeds

## Monitoring Validation

- [ ] `uv run python scripts/run_monitoring_report.py`
- [ ] `reports/monitoring_summary.json` exists
- [ ] `reports/monitoring_report.md` exists

## Demo Artifact Generation

- [ ] `uv run python scripts/demo_smoke_run.py`
- [ ] `artifacts/demo/demo_manifest.json` exists
- [ ] `reports/demo_summary.md` exists

## Streamlit Frontend Validation

- [ ] `uv run python scripts/test_streamlit_frontend_logic.py`
- [ ] `uv run streamlit run streamlit_app.py` starts without artifact errors
- [ ] Prediction page returns binary + multiclass outputs from local artifacts
- [ ] Monitoring page loads `reports/monitoring_summary.json` or shows friendly generation hint
- [ ] Project overview clearly states non-clinical demo scope

## Lint and Tests

- [ ] `uv run ruff check .`
- [ ] `uv run pytest`

## Final Doc Consistency

- [ ] README commands match existing scripts
- [ ] Referenced report/artifact paths exist
- [ ] Troubleshooting and workflow docs align with current behavior
- [ ] Streamlit Community Cloud deployment notes match `streamlit_app.py` entrypoint
