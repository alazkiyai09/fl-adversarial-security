"""Model serialization utilities for SignGuard."""

import torch
import pickle
import json
import gzip
from pathlib import Path
from typing import Dict, Any, Optional
from src.defenses.signguard_full.legacy.core.types import ModelUpdate, SignedUpdate


def serialize_model(
    model: torch.nn.Module,
    filepath: Path | str,
    compress: bool = True,
) -> None:
    """Serialize model to disk.

    Args:
        model: PyTorch model to save
        filepath: Path to save model
        compress: Whether to use gzip compression
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()

    if compress:
        with gzip.open(filepath, "wb") as f:
            pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        torch.save(state_dict, filepath)


def deserialize_model(
    filepath: Path | str,
    model: Optional[torch.nn.Module] = None,
    device: str = "cpu",
) -> Dict[str, torch.Tensor] | torch.nn.Module:
    """Deserialize model from disk.

    Args:
        filepath: Path to saved model
        model: Optional model to load state into
        device: Device to load tensors onto

    Returns:
        State dict if model is None, otherwise the model itself
    """
    filepath = Path(filepath)

    if filepath.suffix == ".gz":
        with gzip.open(filepath, "rb") as f:
            state_dict = pickle.load(f)
    else:
        state_dict = torch.load(filepath, map_location=device)

    if model is None:
        return state_dict

    model.load_state_dict(state_dict)
    return model


def serialize_update(
    update: ModelUpdate | SignedUpdate,
    filepath: Path | str,
    format: str = "json",
) -> None:
    """Serialize model update to disk.

    Args:
        update: Model update to save
        filepath: Path to save update
        format: Serialization format ('json' or 'pickle')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        if isinstance(update, SignedUpdate):
            data = {
                "update": update.update.to_serializable(),
                "signature": update.signature,
                "public_key": update.public_key,
                "algorithm": update.algorithm,
            }
        else:
            data = update.to_serializable()

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    elif format == "pickle":
        with open(filepath, "wb") as f:
            pickle.dump(update, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f"Unknown format: {format}")


def deserialize_update(
    filepath: Path | str,
    device: str = "cpu",
) -> ModelUpdate | SignedUpdate:
    """Deserialize model update from disk.

    Args:
        filepath: Path to saved update
        device: Device to load tensors onto

    Returns:
        Deserialized update
    """
    filepath = Path(filepath)

    if filepath.suffix == ".json":
        with open(filepath, "r") as f:
            data = json.load(f)

        if "signature" in data:
            # SignedUpdate
            update = ModelUpdate.from_serializable(data["update"], device)
            return SignedUpdate(
                update=update,
                signature=data["signature"],
                public_key=data["public_key"],
                algorithm=data.get("algorithm", "ECDSA"),
            )
        else:
            # ModelUpdate
            return ModelUpdate.from_serializable(data, device)

    elif filepath.suffix == ".pkl":
        with open(filepath, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown file format: {filepath.suffix}")


def parameters_to_vector(
    parameters: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Convert parameter dictionary to flat vector.

    Args:
        parameters: Dictionary of layer name -> tensor

    Returns:
        Flattened parameter vector
    """
    return torch.cat([param.flatten() for param in parameters.values()])


def vector_to_parameters(
    vector: torch.Tensor,
    reference_parameters: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Convert flat vector back to parameter dictionary.

    Args:
        vector: Flattened parameter vector
        reference_parameters: Reference dict for structure

    Returns:
        Dictionary of layer name -> tensor
    """
    parameters = {}
    idx = 0

    for name, ref_param in reference_parameters.items():
        size = ref_param.numel()
        param = vector[idx : idx + size].view_as(ref_param)
        parameters[name] = param
        idx += size

    return parameters


def compute_parameter_difference(
    params1: Dict[str, torch.Tensor],
    params2: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute element-wise difference between two parameter sets.

    Args:
        params1: First parameter set
        params2: Second parameter set

    Returns:
        Dictionary of parameter differences
    """
    return {
        name: param1 - param2
        for name, (param1, param2) in zip(params1.items(), params2.items())
    }


def hash_parameters(
    parameters: Dict[str, torch.Tensor],
) -> str:
    """Compute hash of model parameters.

    Args:
        parameters: Model parameters

    Returns:
        Hex digest of parameter hash
    """
    import hashlib

    # Concatenate and serialize
    vector = parameters_to_vector(parameters)
    bytes_data = vector.cpu().numpy().tobytes()

    # Compute hash
    return hashlib.sha256(bytes_data).hexdigest()


def save_checkpoint(
    round_num: int,
    global_model: torch.nn.Module,
    reputations: Dict[str, float],
    filepath: Path | str,
) -> None:
    """Save training checkpoint.

    Args:
        round_num: Current training round
        global_model: Global model state
        reputations: Client reputations
        filepath: Path to save checkpoint
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "round_num": round_num,
        "global_model": global_model.state_dict(),
        "reputations": reputations,
    }

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Path | str,
    model: Optional[torch.nn.Module] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load training checkpoint.

    Args:
        filepath: Path to checkpoint
        model: Optional model to load state into
        device: Device to load onto

    Returns:
        Dictionary with 'round_num', 'global_model', 'reputations'
    """
    filepath = Path(filepath)
    checkpoint = torch.load(filepath, map_location=device)

    if model is not None:
        model.load_state_dict(checkpoint["global_model"])
        checkpoint["global_model"] = model

    return checkpoint
