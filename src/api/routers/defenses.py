"""Defense routes."""

from fastapi import APIRouter
from pydantic import BaseModel

from src.defenses import available_defenses


router = APIRouter(prefix="/api/v1/defenses", tags=["defenses"])


class DefenseEvaluationRequest(BaseModel):
    defense_type: str
    attack_type: str


@router.get("/types")
def defense_types() -> dict:
    return {"defenses": available_defenses()}


@router.post("/evaluate")
def evaluate_defense(request: DefenseEvaluationRequest) -> dict:
    baseline = {
        "krum": 0.81,
        "trimmed_mean": 0.77,
        "bulyan": 0.84,
        "median": 0.73,
        "anomaly_detection": 0.79,
        "foolsgold": 0.76,
        "signguard": 0.88,
    }
    detection_rate = baseline.get(request.defense_type, 0.7)
    return {
        "defense_type": request.defense_type,
        "attack_type": request.attack_type,
        "detection_rate": detection_rate,
        "false_positive_rate": round(max(0.01, 1.0 - detection_rate) / 4, 4),
    }
