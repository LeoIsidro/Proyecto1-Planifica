import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    PredictionResponse,
    TransactionRequest,
    VersionResponse,
)

API_VERSION = "1.0.0"
MODEL_VERSION = "xgb-1.0.0"

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"

REVIEW_THRESHOLD = 0.30
BLOCK_THRESHOLD = 0.70

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = ARTIFACTS_DIR / "xgb_model.joblib"
    preprocessor_path = ARTIFACTS_DIR / "preprocessor.joblib"

    if not model_path.exists() or not preprocessor_path.exists():
        raise RuntimeError(
            f"No se encontraron los artifacts en {ARTIFACTS_DIR}. "
            "Ejecuta los notebooks 02 y 03 antes de levantar la API."
        )

    state["model"] = joblib.load(model_path)
    state["preprocessor"] = joblib.load(preprocessor_path)
    state["loaded_at"] = time.time()

    yield

    state.clear()


app = FastAPI(
    title="IEEE-CIS Fraud Detection API",
    version=API_VERSION,
    description=(
        "Servicio de inferencia para clasificación de transacciones como fraudulentas. "
        "Carga un XGBoost entrenado en GPU + el preprocessor del pipeline."
    ),
    lifespan=lifespan,
)


def _preprocess_one(features: dict, preprocessor: dict) -> pd.DataFrame:
    feature_cols = preprocessor["feature_cols"]
    cat_cols = preprocessor["cat_cols"]
    encoder = preprocessor["encoder"]

    row = {col: features.get(col, np.nan) for col in feature_cols}
    df = pd.DataFrame([row], columns=feature_cols)

    for c in cat_cols:
        df[c] = df[c].fillna("__missing__").astype(str)
    df[cat_cols] = encoder.transform(df[cat_cols])

    num_cols = preprocessor["num_cols"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _preprocess_batch(transactions: list[TransactionRequest], preprocessor: dict) -> pd.DataFrame:
    feature_cols = preprocessor["feature_cols"]
    cat_cols = preprocessor["cat_cols"]
    num_cols = preprocessor["num_cols"]
    encoder = preprocessor["encoder"]

    rows = [{col: t.features.get(col, np.nan) for col in feature_cols} for t in transactions]
    df = pd.DataFrame(rows, columns=feature_cols)

    for c in cat_cols:
        df[c] = df[c].fillna("__missing__").astype(str)
    df[cat_cols] = encoder.transform(df[cat_cols])

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _decision(proba: float) -> str:
    if proba >= BLOCK_THRESHOLD:
        return "block"
    if proba >= REVIEW_THRESHOLD:
        return "review"
    return "allow"


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest) -> PredictionResponse:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    t0 = time.perf_counter()
    try:
        X = _preprocess_one(request.features, state["preprocessor"])
        proba = float(state["model"].predict_proba(X)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error en inferencia: {exc}") from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        transaction_id=request.transaction_id,
        fraud_probability=round(proba, 6),
        decision=_decision(proba),
        review_threshold=REVIEW_THRESHOLD,
        block_threshold=BLOCK_THRESHOLD,
        model_version=MODEL_VERSION,
        inference_time_ms=round(elapsed_ms, 3),
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> BatchResponse:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    t0 = time.perf_counter()
    try:
        X = _preprocess_batch(request.transactions, state["preprocessor"])
        probas = state["model"].predict_proba(X)[:, 1]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error en inferencia: {exc}") from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000

    predictions = [
        PredictionResponse(
            transaction_id=t.transaction_id,
            fraud_probability=round(float(p), 6),
            decision=_decision(float(p)),
            review_threshold=REVIEW_THRESHOLD,
            block_threshold=BLOCK_THRESHOLD,
            model_version=MODEL_VERSION,
            inference_time_ms=round(elapsed_ms / len(probas), 3),
        )
        for t, p in zip(request.transactions, probas)
    ]

    return BatchResponse(
        predictions=predictions,
        n=len(predictions),
        total_time_ms=round(elapsed_ms, 3),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if "model" in state else "degraded",
        model_loaded="model" in state,
        preprocessor_loaded="preprocessor" in state,
    )


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    if "preprocessor" not in state:
        raise HTTPException(status_code=503, detail="Preprocessor no cargado")

    pp = state["preprocessor"]
    return VersionResponse(
        api_version=API_VERSION,
        model_type=MODEL_VERSION,
        n_features=len(pp["feature_cols"]),
        n_categorical_features=len(pp["cat_cols"]),
        n_numeric_features=len(pp["num_cols"]),
        train_end=pp.get("train_end"),
        val_end=pp.get("val_end"),
        loaded_at=state.get("loaded_at"),
    )


@app.get("/")
def root() -> dict:
    return {
        "service": "IEEE-CIS Fraud Detection API",
        "version": API_VERSION,
        "docs": "/docs",
        "endpoints": ["/predict", "/predict/batch", "/health", "/version"],
    }
