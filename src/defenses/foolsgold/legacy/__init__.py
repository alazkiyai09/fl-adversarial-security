"""
FoolsGold Defense - Sybil-Resistant Federated Learning

Main package for FoolsGold implementation.
"""

__version__ = "1.0.0"
__author__ = "30Days Project"

from .aggregators import FoolsGoldAggregator
from .attacks import SybilAttack, CollusionAttack
from .clients import FraudClient
from .server import FoolsGoldServer
from .models import FraudNet
from .experiments import run_defense_comparison, run_ablation_study

__all__ = [
    "FoolsGoldAggregator",
    "SybilAttack",
    "CollusionAttack",
    "FraudClient",
    "FraudNet",
    "run_defense_comparison",
    "run_ablation_study",
]
