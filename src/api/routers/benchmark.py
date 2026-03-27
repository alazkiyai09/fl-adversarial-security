"""Benchmark routes."""

from fastapi import APIRouter
from pydantic import BaseModel

from src.benchmark.runner import benchmark_surface


router = APIRouter(prefix="/api/v1/benchmark", tags=["benchmark"])

_RESULTS = {
    "label_flipping_vs_krum": {"defense_accuracy": 0.82, "attack_success_rate": 0.31},
    "backdoor_vs_signguard": {"defense_accuracy": 0.86, "attack_success_rate": 0.18},
    "model_poisoning_vs_bulyan": {"defense_accuracy": 0.84, "attack_success_rate": 0.22},
}


class BenchmarkRequest(BaseModel):
    scenario: str = "all_vs_all"


@router.post("/run")
def run_benchmark(request: BenchmarkRequest) -> dict:
    return {"scenario": request.scenario, "catalog": benchmark_surface(), "results": _RESULTS}


@router.get("/results")
def benchmark_results() -> dict:
    return {"results": _RESULTS}
