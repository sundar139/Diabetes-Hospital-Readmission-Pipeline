FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts/run_api.py ./scripts/run_api.py
COPY artifacts/binary_model.joblib ./artifacts/binary_model.joblib
COPY artifacts/binary_model_metadata.json ./artifacts/binary_model_metadata.json
COPY artifacts/multiclass_model.joblib ./artifacts/multiclass_model.joblib
COPY artifacts/multiclass_model_metadata.json ./artifacts/multiclass_model_metadata.json

RUN python -m pip install --upgrade pip \
    && pip install .

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "scripts/run_api.py", "--host", "0.0.0.0", "--port", "8000"]
