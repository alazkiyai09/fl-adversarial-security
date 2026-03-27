"""FastAPI application for fraud detection serving."""

from typing import Optional, Dict, Any
import time
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from loguru import logger
import torch

from .prediction import Predictor, PredictionRequest, PredictionResponse
from .model_store import ModelStore


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: Optional[str]
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str
    timestamp: str


# Global state
model_store: Optional[ModelStore] = None
predictor: Optional[Predictor] = None
start_time: float = time.time()
config: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Fraud Detection API")

    # Load active model
    global predictor
    if model_store:
        try:
            predictor = model_store.get_active_predictor(
                config=config.get("model", {}),
                device=torch.device(config.get("device", "cpu")),
            )
            if predictor:
                logger.info(f"Loaded model version {predictor.metadata.version}")
            else:
                logger.warning("No active model found")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    yield

    # Shutdown
    logger.info("Shutting down Fraud Detection API")


def create_app(
    store: ModelStore,
    app_config: Optional[Dict[str, Any]] = None,
) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        store: Model store instance
        app_config: Application configuration

    Returns:
        FastAPI application
    """
    global model_store, config
    model_store = store
    config = app_config or {}

    # Create FastAPI app
    app = FastAPI(
        title="Fraud Detection API",
        description="Privacy-preserving federated learning fraud detection service",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests."""
        start_time = time.time()

        # Log request
        logger.info(f"{request.method} {request.url.path}")

        # Process request
        response = await call_next(request)

        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code}, "
            f"Time: {process_time:.3f}s"
        )

        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)

        return response

    # Health check endpoint
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """
        Health check endpoint.

        Returns API status and model information.
        """
        uptime = time.time() - start_time

        return HealthResponse(
            status="healthy",
            model_loaded=predictor is not None,
            model_version=predictor.metadata.version if predictor else None,
            uptime_seconds=uptime,
        )

    # Model info endpoint
    @app.get("/model/info", tags=["Model"])
    async def get_model_info():
        """
        Get information about the current model.

        Returns model version, type, and performance metrics.
        """
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No model loaded",
            )

        return predictor.get_model_info()

    # List model versions
    @app.get("/model/versions", tags=["Model"])
    async def list_model_versions():
        """
        List all available model versions.
        """
        if model_store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model store not available",
            )

        versions = model_store.list_versions()
        return {"versions": versions}

    # Activate model version
    @app.post("/model/activate/{version}", tags=["Model"])
    async def activate_model_version(version: str):
        """
        Activate a specific model version.

        - **version**: Model version to activate
        """
        if model_store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model store not available",
            )

        try:
            model_store.activate_model(version)

            # Reload predictor
            global predictor
            predictor = model_store.get_active_predictor(
                config=config.get("model", {}),
                device=torch.device(config.get("device", "cpu")),
            )

            return {
                "status": "success",
                "message": f"Activated model version {version}",
                "version": version,
            }

        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e),
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to activate model: {e}",
            )

    # Rollback model
    @app.post("/model/rollback", tags=["Model"])
    async def rollback_model(target_version: Optional[str] = None):
        """
        Rollback to a previous model version.

        - **target_version**: Optional specific version to rollback to
        """
        if model_store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model store not available",
            )

        success = model_store.rollback(target_version)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Rollback failed",
            )

        # Reload predictor
        global predictor
        predictor = model_store.get_active_predictor(
            config=config.get("model", {}),
            device=torch.device(config.get("device", "cpu")),
        )

        return {
            "status": "success",
            "message": "Rolled back successfully",
            "version": model_store.get_active_version(),
        }

    # Prediction endpoint
    @app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
    async def predict(request: PredictionRequest, http_request: Request):
        """
        Make fraud predictions for transactions.

        - **transactions**: List of transaction dictionaries
        - **model_version**: Optional model version to use
        - **return_probabilities**: Whether to return class probabilities
        - **threshold**: Optional prediction threshold (default: 0.5)

        Returns predictions with fraud probabilities.
        """
        start_time = time.time()

        # Check if predictor is available
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No model loaded",
            )

        try:
            # Make predictions
            predictions = predictor.predict(
                transactions=request.transactions,
                return_probabilities=request.return_probabilities,
                threshold=request.threshold,
            )

            processing_time = (time.time() - start_time) * 1000

            return PredictionResponse(
                predictions=predictions,
                model_info=predictor.get_model_info(),
                processing_time_ms=processing_time,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {e}",
            )

    # Single transaction prediction (convenience endpoint)
    @app.post("/predict/single", tags=["Prediction"])
    async def predict_single(transaction: Dict[str, Any]):
        """
        Make a fraud prediction for a single transaction.

        Convenience endpoint for single transactions.
        """
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No model loaded",
            )

        try:
            prediction = predictor.predict_single(transaction)

            return {
                "prediction": prediction,
                "model_version": predictor.metadata.version,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {e}",
            )

    # Batch prediction endpoint
    @app.post("/predict/batch", tags=["Prediction"])
    async def predict_batch(
        transactions: list[Dict[str, Any]],
        threshold: Optional[float] = None,
    ):
        """
        Batch prediction endpoint for high-throughput scenarios.

        Optimized for processing large batches of transactions.
        """
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No model loaded",
            )

        start_time = time.time()

        try:
            predictions = predictor.predict(
                transactions=transactions,
                return_probabilities=False,
                threshold=threshold,
            )

            processing_time = (time.time() - start_time) * 1000

            return {
                "predictions": predictions,
                "processing_time_ms": processing_time,
                "n_transactions": len(transactions),
                "model_version": predictor.metadata.version,
            }

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch prediction failed: {e}",
            )

    # Update prediction threshold
    @app.post("/model/threshold", tags=["Model"])
    async def update_threshold(threshold: float = Field(..., ge=0, le=1)):
        """
        Update the prediction threshold.

        - **threshold**: New threshold value (0-1)
        """
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No model loaded",
            )

        try:
            predictor.update_threshold(threshold)

            return {
                "status": "success",
                "message": f"Updated threshold to {threshold}",
                "threshold": threshold,
            }

        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

    # Exception handlers
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle ValueError exceptions."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Invalid input",
                "detail": str(exc),
                "timestamp": datetime.now().isoformat(),
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {exc}")

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": str(exc),
                "timestamp": datetime.now().isoformat(),
            },
        )

    return app


class FraudDetectionAPI:
    """
    Fraud detection API server.

    Wraps FastAPI application for easier management.
    """

    def __init__(
        self,
        model_store: ModelStore,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize API server.

        Args:
            model_store: Model store instance
            config: Application configuration
        """
        self.model_store = model_store
        self.config = config or {}

        # Create FastAPI app
        self.app = create_app(model_store, self.config)

        logger.info("FraudDetectionAPI initialized")

    def get_app(self) -> FastAPI:
        """Get FastAPI application instance."""
        return self.app

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        **kwargs,
    ) -> None:
        """
        Run the API server.

        Args:
            host: Host to bind to
            port: Port to bind to
            **kwargs: Additional arguments for uvicorn
        """
        import uvicorn

        uvicorn.run(
            self.app,
            host=host,
            port=port,
            **kwargs,
        )
