"""
Reconstruction quality metrics.
Measure how well reconstructed data matches original data.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Union
from skimage.metrics import structural_similarity as ssim


def compute_mse(
    original: Union[torch.Tensor, np.ndarray],
    reconstructed: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Compute Mean Squared Error between original and reconstructed.

    Args:
        original: Original data
        reconstructed: Reconstructed data

    Returns:
        MSE value (lower is better)
    """
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

    return float(np.mean((original - reconstructed) ** 2))


def compute_psnr(
    original: Union[torch.Tensor, np.ndarray],
    reconstructed: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        original: Original data
        reconstructed: Reconstructed data
        data_range: Maximum possible pixel value (1.0 for normalized data)

    Returns:
        PSNR in dB (higher is better)
    """
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

    mse = np.mean((original - reconstructed) ** 2)

    if mse == 0:
        return float('inf')

    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
    return float(psnr)


def compute_ssim(
    original: Union[torch.Tensor, np.ndarray],
    reconstructed: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
    multichannel: bool = False
) -> float:
    """
    Compute Structural Similarity Index.

    Args:
        original: Original data
        reconstructed: Reconstructed data
        data_range: Maximum possible pixel value
        multichannel: Whether to treat last dimension as channels

    Returns:
        SSIM value (higher is better, range [-1, 1])
    """
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

    # Remove batch dimension if present
    if original.ndim == 4:  # [batch, channels, height, width]
        original = original[0]
        reconstructed = reconstructed[0]

    # Transpose from [channels, height, width] to [height, width, channels]
    if original.ndim == 3 and original.shape[0] in [1, 3]:
        original = np.transpose(original, (1, 2, 0))
        reconstructed = np.transpose(reconstructed, (1, 2, 0))
        multichannel = True

    # Squeeze single channel
    if original.ndim == 3 and original.shape[2] == 1:
        original = original[:, :, 0]
        reconstructed = reconstructed[:, :, 0]
        multichannel = False

    try:
        ssim_value = ssim(
            original,
            reconstructed,
            data_range=data_range,
            channel_axis=-1 if multichannel else None
        )
        return float(ssim_value)
    except Exception as e:
        print(f"Warning: SSIM computation failed: {e}")
        return 0.0


def compute_label_accuracy(
    original_labels: Union[torch.Tensor, np.ndarray],
    reconstructed_labels: Union[torch.Tensor, np.ndarray]
) -> Tuple[float, bool]:
    """
    Compute label classification accuracy.

    Args:
        original_labels: Original labels
        reconstructed_labels: Reconstructed labels

    Returns:
        (accuracy, exact_match) tuple
    """
    if isinstance(original_labels, torch.Tensor):
        original_labels = original_labels.detach().cpu().numpy()
    if isinstance(reconstructed_labels, torch.Tensor):
        reconstructed_labels = reconstructed_labels.detach().cpu().numpy()

    # Flatten
    original_labels = original_labels.flatten()
    reconstructed_labels = reconstructed_labels.flatten().astype(int)

    accuracy = np.mean(original_labels == reconstructed_labels)
    exact_match = bool(accuracy == 1.0)

    return float(accuracy), exact_match


def compute_reconstruction_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    original_labels: torch.Tensor,
    reconstructed_labels: torch.Tensor,
    data_range: float = 1.0
) -> Dict[str, float]:
    """
    Compute all reconstruction quality metrics.

    Args:
        original: Original input data
        reconstructed: Reconstructed input data
        original_labels: Original labels
        reconstructed_labels: Reconstructed labels
        data_range: Data range for PSNR/SSIM

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # Data reconstruction metrics
    metrics['mse'] = compute_mse(original, reconstructed)
    metrics['psnr'] = compute_psnr(original, reconstructed, data_range)
    metrics['ssim'] = compute_ssim(original, reconstructed, data_range)

    # Label metrics
    label_acc, exact_match = compute_label_accuracy(original_labels, reconstructed_labels)
    metrics['label_accuracy'] = label_acc
    metrics['label_match'] = float(exact_match)

    # Combined score (weighted)
    # Higher is better
    metrics['combined_score'] = (
        0.4 * metrics['ssim'] +
        0.3 * (metrics['psnr'] / 50.0) +  # Normalize PSNR to ~[0, 1]
        0.3 * metrics['label_accuracy']
    )

    return metrics


def evaluate_batch_reconstruction(
    original_batch: torch.Tensor,
    reconstructed_batch: torch.Tensor,
    original_labels: torch.Tensor,
    reconstructed_labels: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluate reconstruction quality for a batch.

    Args:
        original_batch: Batch of original data [N, ...]
        reconstructed_batch: Batch of reconstructed data [N, ...]
        original_labels: Original labels [N]
        reconstructed_labels: Reconstructed labels [N]

    Returns:
        Dictionary with aggregate metrics
    """
    batch_size = original_batch.shape[0]

    # Compute per-sample metrics
    mse_values = []
    ssim_values = []
    psnr_values = []
    label_matches = []

    for i in range(batch_size):
        original = original_batch[i]
        reconstructed = reconstructed_batch[i]

        mse_values.append(compute_mse(original, reconstructed))
        ssim_values.append(compute_ssim(original, reconstructed))
        psnr_values.append(compute_psnr(original, reconstructed))

        if original_labels[i] == reconstructed_labels[i]:
            label_matches.append(1.0)
        else:
            label_matches.append(0.0)

    # Aggregate
    return {
        'mse_mean': float(np.mean(mse_values)),
        'mse_std': float(np.std(mse_values)),
        'ssim_mean': float(np.mean(ssim_values)),
        'ssim_std': float(np.std(ssim_values)),
        'psnr_mean': float(np.mean(psnr_values)),
        'psnr_std': float(np.std(psnr_values)),
        'label_accuracy': float(np.mean(label_matches)),
        'num_samples': batch_size
    }


def format_metrics_report(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    Format metrics as a readable report.

    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for each line

    Returns:
        Formatted string
    """
    lines = [f"{prefix}Reconstruction Quality Report:"]
    lines.append(f"{prefix}  Data Quality:")
    lines.append(f"{prefix}    MSE: {metrics.get('mse', 0):.6f}")
    lines.append(f"{prefix}    PSNR: {metrics.get('psnr', 0):.2f} dB")
    lines.append(f"{prefix}    SSIM: {metrics.get('ssim', 0):.4f}")
    lines.append(f"{prefix}  Label Recovery:")
    lines.append(f"{prefix}    Accuracy: {metrics.get('label_accuracy', 0):.2%}")
    lines.append(f"{prefix}    Exact Match: {metrics.get('label_match', False)}")

    if 'combined_score' in metrics:
        lines.append(f"{prefix}  Combined Score: {metrics['combined_score']:.4f}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics
    print("Testing reconstruction quality metrics...")

    # Create test data
    original = torch.rand(1, 1, 28, 28)
    reconstructed = original + torch.randn_like(original) * 0.1

    original_labels = torch.tensor([5])
    reconstructed_labels = torch.tensor([5])

    # Compute metrics
    metrics = compute_reconstruction_metrics(
        original, reconstructed,
        original_labels, reconstructed_labels
    )

    print("\n" + format_metrics_report(metrics))

    # Test with wrong label
    reconstructed_labels_wrong = torch.tensor([3])
    metrics_wrong = compute_reconstruction_metrics(
        original, reconstructed,
        original_labels, reconstructed_labels_wrong
    )

    print("\n" + format_metrics_report(metrics_wrong, prefix="Wrong Label: "))
