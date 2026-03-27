"""Target model training for membership inference attacks."""

from .fl_target import FraudDetectionNN, FLTargetTrainer, create_client_splits

__all__ = ['FraudDetectionNN', 'FLTargetTrainer', 'create_client_splits']
