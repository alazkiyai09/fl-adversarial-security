"""Attack routes."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.attacks import available_attacks


router = APIRouter(prefix="/api/v1/attacks", tags=["attacks"])


class AttackSimulationRequest(BaseModel):
    attack_type: str
    num_clients: int = Field(default=10, ge=1)
    malicious_fraction: float = Field(default=0.2, ge=0, le=1)


@router.get("/types")
def attack_types() -> dict:
    return {"attacks": available_attacks()}


@router.post("/simulate")
def simulate_attack(request: AttackSimulationRequest) -> dict:
    impact = round(min(1.0, request.malicious_fraction * 1.8 + request.num_clients / 200.0), 4)
    return {
        "attack_type": request.attack_type,
        "num_clients": request.num_clients,
        "malicious_fraction": request.malicious_fraction,
        "impact_score": impact,
        "success_estimate": "high" if impact >= 0.6 else "moderate",
    }
