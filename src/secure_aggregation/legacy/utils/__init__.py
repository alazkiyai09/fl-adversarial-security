"""
Utility functions for secure aggregation.
"""

from .serialization import (
    tensor_to_dict,
    dict_to_tensor,
    serialize_model_update,
    deserialize_model_update
)

from .dropout import (
    simulate_dropouts,
    DropoutSimulator
)

__all__ = [
    'tensor_to_dict',
    'dict_to_tensor',
    'serialize_model_update',
    'deserialize_model_update',
    'simulate_dropouts',
    'DropoutSimulator',
]
