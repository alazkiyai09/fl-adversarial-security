"""Unified FastAPI app for fl-adversarial-security."""

from fastapi import FastAPI

from src.api.routers import attacks, benchmark, defenses, predict
from src.core.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title=settings.api_name, version=settings.api_version)
    app.include_router(attacks.router)
    app.include_router(defenses.router)
    app.include_router(benchmark.router)
    app.include_router(predict.router)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": settings.api_name, "version": settings.api_version}

    @app.get("/metrics")
    def metrics() -> dict:
        return {"families": ["attacks", "defenses", "benchmark", "production"], "service": settings.api_name}

    return app


app = create_app()
