"""
Serialization utilities for model updates.

Convert between PyTorch tensors and serializable formats for
network transmission and storage.
"""

import torch
import json
import base64
from typing import Dict, Any, List
import struct


def tensor_to_dict(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Convert a tensor to a dictionary for serialization.

    Args:
        tensor: PyTorch tensor

    Returns:
        Dictionary with tensor data and metadata
    """
    return {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'data': tensor.tolist()  # Convert to list for JSON serialization
    }


def dict_to_tensor(data: Dict[str, Any]) -> torch.Tensor:
    """
    Convert a dictionary back to a tensor.

    Args:
        data: Dictionary from tensor_to_dict

    Returns:
        PyTorch tensor
    """
    dtype_map = {
        'torch.float32': torch.float32,
        'torch.float64': torch.float64,
        'torch.float16': torch.float16,
        'torch.int32': torch.int32,
        'torch.int64': torch.int64,
    }

    dtype = dtype_map.get(data['dtype'], torch.float32)
    tensor = torch.tensor(data['data'], dtype=dtype)

    return tensor


def serialize_model_update(update: torch.Tensor) -> str:
    """
    Serialize a model update to a string for transmission.

    Args:
        update: Model update tensor

    Returns:
        JSON string representation
    """
    data = tensor_to_dict(update)
    return json.dumps(data)


def deserialize_model_update(serialized: str) -> torch.Tensor:
    """
    Deserialize a model update from a string.

    Args:
        serialized: JSON string from serialize_model_update

    Returns:
        Model update tensor
    """
    data = json.loads(serialized)
    return dict_to_tensor(data)


def pack_tensor_binary(tensor: torch.Tensor) -> bytes:
    """
    Pack tensor into binary format for efficient transmission.

    Args:
        tensor: PyTorch tensor

    Returns:
        Binary representation
    """
    # Convert tensor to bytes
    buffer = tensor.numpy().tobytes()

    # Pack with metadata (shape and dtype)
    shape_bytes = struct.pack(f'{len(tensor.shape)}I', *tensor.shape)
    dtype_bytes = str(tensor.dtype).encode('utf-8')

    # Format: dtype_len(1) + dtype + shape_len(1) + shape + data
    result = (
        bytes([len(dtype_bytes)]) +
        dtype_bytes +
        bytes([len(tensor.shape)]) +
        shape_bytes +
        buffer
    )

    return result


def unpack_tensor_binary(data: bytes) -> torch.Tensor:
    """
    Unpack tensor from binary format.

    Args:
        data: Binary data from pack_tensor_binary

    Returns:
        PyTorch tensor
    """
    import numpy as np

    idx = 0

    # Read dtype
    dtype_len = data[idx]
    idx += 1
    dtype = data[idx:idx + dtype_len].decode('utf-8')
    idx += dtype_len

    # Read shape
    shape_len = data[idx]
    idx += 1
    shape = struct.unpack(f'{shape_len}I', data[idx:idx + shape_len * 4])
    idx += shape_len * 4

    # Read tensor data
    tensor_data = data[idx:]

    # Convert to numpy array then to torch tensor
    dtype_map = {
        'torch.float32': np.float32,
        'torch.float64': np.float64,
        'torch.float16': np.float16,
        'torch.int32': np.int32,
        'torch.int64': np.int64,
    }

    np_dtype = dtype_map.get(dtype, np.float32)
    np_array = np.frombuffer(tensor_data, dtype=np_dtype).reshape(shape)

    return torch.from_numpy(np_array)


def get_tensor_size_bytes(tensor: torch.Tensor) -> int:
    """
    Get the size of a tensor in bytes.

    Args:
        tensor: PyTorch tensor

    Returns:
        Size in bytes
    """
    return tensor.element_size() * tensor.numel()
