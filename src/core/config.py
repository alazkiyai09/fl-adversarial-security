"""Runtime settings."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    api_name: str = os.getenv("FL_SECURITY_API_NAME", "fl-adversarial-security")
    api_version: str = os.getenv("FL_SECURITY_API_VERSION", "0.1.0")
    secure_prediction_threshold: float = float(os.getenv("SECURE_PREDICTION_THRESHOLD", "0.7"))


settings = Settings()
