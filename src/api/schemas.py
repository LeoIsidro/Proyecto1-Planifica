from typing import Any, Optional, Union

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    transaction_id: Optional[Union[str, int]] = Field(
        None, description="Identificador opcional de la transacción para trazabilidad."
    )
    features: dict[str, Any] = Field(
        ...,
        description=(
            "Diccionario con las features crudas de la transacción. "
            "Las claves faltantes se imputan automáticamente."
        ),
    )


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(
        ..., min_length=1, max_length=1000
    )


class PredictionResponse(BaseModel):
    transaction_id: Optional[Union[str, int]] = None
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    decision: str = Field(..., description="allow | review | block")
    review_threshold: float
    block_threshold: float
    model_version: str
    inference_time_ms: float


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    n: int
    total_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    preprocessor_loaded: bool


class VersionResponse(BaseModel):
    api_version: str
    model_type: str
    n_features: int
    n_categorical_features: int
    n_numeric_features: int
    train_end: Optional[str]
    val_end: Optional[str]
    loaded_at: Optional[float]


class ErrorResponse(BaseModel):
    detail: str
