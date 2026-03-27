"""
Prepare ground-truth gradients for gradient leakage attacks.
Computes gradients from real (x, y) pairs to be used as attack targets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from pathlib import Path


def compute_gradients(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: Optional[nn.Module] = None,
    create_graph: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Compute gradients from given input and label.

    Args:
        model: Target model
        x: Input data
        y: Ground truth labels
        criterion: Loss function (default: CrossEntropyLoss)
        create_graph: Whether to create computation graph (for higher-order gradients)

    Returns:
        Dictionary mapping parameter names to gradient tensors
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Forward pass
    output = model(x)
    loss = criterion(output, y)

    # Backward pass
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad.data.zero_()

    loss.backward(create_graph=create_graph)

    # Store gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.clone()

        # Clear gradients
        param.grad = None

    return gradients


def prepare_ground_truth_gradients(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: Optional[nn.Module] = None,
    device: torch.device = torch.device('cpu')
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Prepare ground-truth gradients for attack.

    This is the main function to generate the target gradients
    that the attacker will try to match.

    Args:
        model: Target model
        x: Real input data (what attacker wants to recover)
        y: Real labels (what attacker wants to recover)
        criterion: Loss function
        device: Device to use

    Returns:
        (gradients, loss_value) tuple
    """
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)

    # Set model to eval mode
    model.eval()

    # Compute gradients
    gradients = compute_gradients(model, x, y, criterion)

    # Compute loss value for reference
    with torch.no_grad():
        output = model(x)
        loss_value = criterion(output, y) if criterion is not None else F.cross_entropy(output, y)

    return gradients, loss_value.item()


def prepare_batch_gradients(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_batches: int = 10,
    criterion: Optional[nn.Module] = None,
    device: torch.device = torch.device('cpu')
) -> list:
    """
    Prepare ground-truth gradients for multiple batches.

    Args:
        model: Target model
        dataloader: Data loader
        num_batches: Number of batches to process
        criterion: Loss function
        device: Device to use

    Returns:
        List of (x, y, gradients) tuples
    """
    model = model.to(device)
    model.eval()

    results = []

    for i, (x, y) in enumerate(dataloader):
        if i >= num_batches:
            break

        x = x.to(device)
        y = y.to(device)

        # Compute gradients
        gradients, loss = prepare_ground_truth_gradients(
            model, x, y, criterion, device
        )

        results.append({
            'x': x.cpu(),
            'y': y.cpu(),
            'gradients': {k: v.cpu() for k, v in gradients.items()},
            'loss': loss
        })

    return results


def save_gradients(
    gradients: Dict[str, torch.Tensor],
    save_path: str,
    metadata: Optional[dict] = None
) -> None:
    """
    Save gradients to disk.

    Args:
        gradients: Gradient dictionary
        save_path: Path to save gradients
        metadata: Optional metadata to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare save dictionary
    save_dict = {
        'gradients': {k: v.cpu() for k, v in gradients.items()}
    }

    if metadata is not None:
        save_dict['metadata'] = metadata

    # Save
    torch.save(save_dict, save_path)
    print(f"Gradients saved to {save_path}")


def load_gradients(
    load_path: str
) -> Tuple[Dict[str, torch.Tensor], Optional[dict]]:
    """
    Load gradients from disk.

    Args:
        load_path: Path to load gradients from

    Returns:
        (gradients, metadata) tuple
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"Gradient file not found: {load_path}")

    save_dict = torch.load(load_path)

    gradients = save_dict['gradients']
    metadata = save_dict.get('metadata', None)

    return gradients, metadata


def gradient_to_vector(gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Flatten gradient dictionary to a single vector.

    Args:
        gradients: Gradient dictionary

    Returns:
        Flattened gradient vector
    """
    return torch.cat([v.flatten() for v in gradients.values()])


def vector_to_gradient(
    vector: torch.Tensor,
    gradient_shapes: Dict[str, torch.Size]
) -> Dict[str, torch.Tensor]:
    """
    Reshape vector back to gradient dictionary.

    Args:
        vector: Flattened gradient vector
        gradient_shapes: Dictionary of parameter names to shapes

    Returns:
        Gradient dictionary
    """
    gradients = {}
    idx = 0

    for name, shape in gradient_shapes.items():
        numel = shape.numel()
        param_vec = vector[idx:idx + numel]
        gradients[name] = param_vec.reshape(shape)
        idx += numel

    return gradients


def get_gradient_statistics(
    gradients: Dict[str, torch.Tensor]
) -> dict:
    """
    Compute statistics about gradients.

    Args:
        gradients: Gradient dictionary

    Returns:
        Dictionary of gradient statistics
    """
    stats = {}

    for name, grad in gradients.items():
        stats[name] = {
            'shape': list(grad.shape),
            'mean': grad.mean().item(),
            'std': grad.std().item(),
            'min': grad.min().item(),
            'max': grad.max().item(),
            'norm': torch.norm(grad).item(),
            'numel': grad.numel()
        }

    # Total norm
    total_norm = torch.sqrt(sum(torch.norm(grad)**2 for grad in gradients.values()))
    stats['total_norm'] = total_norm.item()
    stats['total_numel'] = sum(s['numel'] for s in stats.values() if isinstance(s, dict))

    return stats


if __name__ == "__main__":
    # Test gradient preparation
    from models.simple_cnn import SimpleCNN

    print("Testing gradient preparation...")

    # Create model and data
    model = SimpleCNN(input_channels=1, num_classes=10)
    x = torch.randn(2, 1, 28, 28)
    y = torch.randint(0, 10, (2,))

    # Compute gradients
    gradients, loss = prepare_ground_truth_gradients(model, x, y)

    print(f"\nLoss: {loss:.4f}")
    print(f"\nGradient statistics:")
    stats = get_gradient_statistics(gradients)
    for name, stat in stats.items():
        if isinstance(stat, dict):
            print(f"  {name}: shape={stat['shape']}, norm={stat['norm']:.4f}")
        else:
            print(f"  {name}: {stat}")

    # Test save/load
    print("\nTesting save/load...")
    save_path = "data/test_gradients.pt"
    save_gradients(gradients, save_path, metadata={'loss': loss})
    loaded_grads, metadata = load_gradients(save_path)
    print(f"Loaded loss: {metadata['loss']:.4f}")

    # Test vectorization
    print("\nTesting vectorization...")
    grad_shapes = {k: v.shape for k, v in gradients.items()}
    vec = gradient_to_vector(gradients)
    print(f"Vector shape: {vec.shape}")

    reconstructed = vector_to_gradient(vec, grad_shapes)
    for name in gradients.keys():
        assert torch.allclose(gradients[name], reconstructed[name])
    print("Vectorization test passed!")
