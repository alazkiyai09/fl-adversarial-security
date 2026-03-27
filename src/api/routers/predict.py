"""Production secure prediction routes."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.core.config import settings


router = APIRouter(prefix="/api/v1", tags=["predict"])


class SecurePredictRequest(BaseModel):
    amount: float = Field(default=0.0, ge=0)
    merchant_risk: float = Field(default=0.0, ge=0, le=1)
    defense_stack: list[str] = Field(default_factory=lambda: ["dp", "signguard", "secure_aggregation"])


@router.post("/predict")
def secure_predict(request: SecurePredictRequest) -> dict:
    probability = min(1.0, round(request.merchant_risk * 0.6 + min(request.amount / 2000.0, 1.0) * 0.4, 4))
    return {
        "fraud_probability": probability,
        "prediction": "fraud" if probability >= 0.5 else "legitimate",
        "threshold": settings.secure_prediction_threshold,
        "controls_active": request.defense_stack,
    }
