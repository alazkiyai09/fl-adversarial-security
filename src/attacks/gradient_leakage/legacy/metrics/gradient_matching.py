"""
Gradient matching metrics.
Measure distance between true gradients and reconstructed gradients.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Union


def gradient_mse_distance(
    grad1: Dict[str, torch.Tensor],
    grad2: Dict[str, torch.Tensor]
) -> float:
    """
    Compute MSE distance between two gradient dictionaries.

    Args:
        grad1: First gradient dictionary
        grad2: Second gradient dictionary

    Returns:
        MSE distance (lower is better)
    """
    distances = []

    for name in grad1.keys():
        if name in grad2:
            dist = ((grad1[name] - grad2[name]) ** 2).sum().item()
            distances.append(dist)

    return sum(distances)


def gradient_cosine_similarity(
    grad1: Dict[str, torch.Tensor],
    grad2: Dict[str, torch.Tensor]
) -> float:
    """
    Compute cosine similarity between two gradient dictionaries.

    Args:
        grad1: First gradient dictionary
        grad2: Second gradient dictionary

    Returns:
        Cosine similarity (higher is better, range [-1, 1])
    """
    # Flatten gradients
    vec1 = torch.cat([g.flatten() for g in grad1.values()])
    vec2 = torch.cat([g.flatten() for g in grad2.values()])

    similarity = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
    return similarity.item()


def gradient_l1_distance(
    grad1: Dict[str, torch.Tensor],
    grad2: Dict[str, torch.Tensor]
) -> float:
    """
    Compute L1 distance between two gradient dictionaries.

    Args:
        grad1: First gradient dictionary
        grad2: Second gradient dictionary

    Returns:
        L1 distance (lower is better)
    """
    distances = []

    for name in grad1.keys():
        if name in grad2:
            dist = torch.abs(grad1[name] - grad2[name]).sum().item()
            distances.append(dist)

    return sum(distances)


def gradient_distance(
    grad1: Dict[str, torch.Tensor],
    grad2: Dict[str, torch.Tensor],
    metric: str = 'mse'
) -> float:
    """
    Compute distance between two gradient dictionaries.

    Args:
        grad1: First gradient dictionary
        grad2: Second gradient dictionary
        metric: Distance metric ('mse', 'cosine', 'l1')

    Returns:
        Distance value
    """
    if metric == 'mse':
        return gradient_mse_distance(grad1, grad2)
    elif metric == 'cosine':
        # Convert similarity to distance
        sim = gradient_cosine_similarity(grad1, grad2)
        return 1.0 - sim
    elif metric == 'l1':
        return gradient_l1_distance(grad1, grad2)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_gradient_norms(
    gradients: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute L2 norms of all gradient tensors.

    Args:
        gradients: Gradient dictionary

    Returns:
        Dictionary of gradient norms
    """
    norms = {}

    for name, grad in gradients.items():
        norms[name] = torch.norm(grad).item()

    # Total norm
    total_norm = torch.sqrt(sum(torch.norm(grad)**2 for grad in gradients.values()))
    norms['total'] = total_norm.item()

    return norms


def compute_layer_wise_distances(
    grad1: Dict[str, torch.Tensor],
    grad2: Dict[str, torch.Tensor],
    metric: str = 'mse'
) -> Dict[str, float]:
    """
    Compute distance for each layer separately.

    Args:
        grad1: First gradient dictionary
        grad2: Second gradient dictionary
        metric: Distance metric

    Returns:
        Dictionary of layer-wise distances
    """
    distances = {}

    for name in grad1.keys():
        if name in grad2:
            if metric == 'mse':
                dist = ((grad1[name] - grad2[name]) ** 2).mean().item()
            elif metric == 'cosine':
                dist = 1 - F.cosine_similarity(
                    grad1[name].flatten().unsqueeze(0),
                    grad2[name].flatten().unsqueeze(0)
                ).item()
            elif metric == 'l1':
                dist = torch.abs(grad1[name] - grad2[name]).mean().item()
            else:
                raise ValueError(f"Unknown metric: {metric}")

            distances[name] = dist

    return distances


def track_gradient_matching(
    true_gradients: Dict[str, torch.Tensor],
    dummy_gradients: Dict[str, torch.Tensor],
    iteration: int
) -> Dict[str, float]:
    """
    Compute comprehensive gradient matching metrics.

    Args:
        true_gradients: True gradients
        dummy_gradients: Dummy/reconstructed gradients
        iteration: Current iteration number

    Returns:
        Dictionary of metrics
    """
    return {
        'iteration': iteration,
        'mse_distance': gradient_mse_distance(true_gradients, dummy_gradients),
        'cosine_similarity': gradient_cosine_similarity(true_gradients, dummy_gradients),
        'l1_distance': gradient_l1_distance(true_gradients, dummy_gradients),
        'true_norm': compute_gradient_norms(true_gradients)['total'],
        'dummy_norm': compute_gradient_norms(dummy_gradients)['total'],
    }


def compute_convergence_metrics(
    gradient_distances: List[float],
    window: int = 100
) -> Dict[str, float]:
    """
    Compute convergence metrics from optimization history.

    Args:
        gradient_distances: List of gradient distances over iterations
        window: Window size for metrics

    Returns:
        Dictionary of convergence metrics
    """
    if len(gradient_distances) == 0:
        return {}

    metrics = {
        'final_loss': gradient_distances[-1],
        'min_loss': min(gradient_distances),
        'max_loss': max(gradient_distances),
        'total_iterations': len(gradient_distances)
    }

    # Improvement
    metrics['total_improvement'] = gradient_distances[0] - gradient_distances[-1]
    metrics['improvement_ratio'] = metrics['total_improvement'] / (gradient_distances[0] + 1e-10)

    # Recent improvement
    if len(gradient_distances) >= window:
        recent_distances = gradient_distances[-window:]
        metrics[f'last_{window}_improvement'] = recent_distances[0] - recent_distances[-1]
        metrics[f'last_{window}_mean'] = np.mean(recent_distances)
        metrics[f'last_{window}_std'] = np.std(recent_distances)

    return metrics


def format_gradient_matching_report(
    true_gradients: Dict[str, torch.Tensor],
    reconstructed_gradients: Dict[str, torch.Tensor]
) -> str:
    """
    Format gradient matching report.

    Args:
        true_gradients: True gradients
        reconstructed_gradients: Reconstructed gradients

    Returns:
        Formatted string
    """
    lines = ["Gradient Matching Report:"]

    # Overall metrics
    lines.append("  Overall:")
    lines.append(f"    MSE Distance: {gradient_mse_distance(true_gradients, reconstructed_gradients):.6e}")
    lines.append(f"    Cosine Similarity: {gradient_cosine_similarity(true_gradients, reconstructed_gradients):.6f}")
    lines.append(f"    L1 Distance: {gradient_l1_distance(true_gradients, reconstructed_gradients):.6e}")

    # Norms
    true_norms = compute_gradient_norms(true_gradients)
    rec_norms = compute_gradient_norms(reconstructed_gradients)

    lines.append("\n  Gradient Norms:")
    lines.append(f"    True: {true_norms['total']:.6f}")
    lines.append(f"    Reconstructed: {rec_norms['total']:.6f}")
    lines.append(f"    Ratio: {rec_norms['total'] / (true_norms['total'] + 1e-10):.6f}")

    # Layer-wise
    lines.append("\n  Layer-wise MSE:")
    layer_distances = compute_layer_wise_distances(true_gradients, reconstructed_gradients, 'mse')
    for name, dist in layer_distances.items():
        lines.append(f"    {name}: {dist:.6e}")

    return "\n".join(lines)


class GradientMatchingTracker:
    """
    Track gradient matching metrics during optimization.
    """

    def __init__(self, true_gradients: Dict[str, torch.Tensor]):
        """
        Initialize tracker.

        Args:
            true_gradients: True gradients to match against
        """
        self.true_gradients = true_gradients
        self.history = []

    def track(self, dummy_gradients: Dict[str, torch.Tensor], iteration: int):
        """
        Track current state.

        Args:
            dummy_gradients: Current dummy gradients
            iteration: Current iteration
        """
        metrics = track_gradient_matching(
            self.true_gradients, dummy_gradients, iteration
        )
        self.history.append(metrics)

    def get_history(self) -> List[Dict[str, float]]:
        """Get full tracking history."""
        return self.history

    def get_summary(self) -> Dict[str, float]:
        """Get summary of optimization progress."""
        if len(self.history) == 0:
            return {}

        initial = self.history[0]
        final = self.history[-1]

        return {
            'iterations': len(self.history),
            'initial_mse': initial['mse_distance'],
            'final_mse': final['mse_distance'],
            'mse_improvement': initial['mse_distance'] - final['mse_distance'],
            'initial_cosine': initial['cosine_similarity'],
            'final_cosine': final['cosine_similarity'],
            'cosine_improvement': final['cosine_similarity'] - initial['cosine_similarity'],
        }

    def plot_history(self, save_path: str = None):
        """
        Plot optimization history.

        Args:
            save_path: Optional path to save plot
        """
        import matplotlib.pyplot as plt

        iterations = [m['iteration'] for m in self.history]
        mse_distances = [m['mse_distance'] for m in self.history]
        cosine_sims = [m['cosine_similarity'] for m in self.history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # MSE distance
        ax1.plot(iterations, mse_distances)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('MSE Distance')
        ax1.set_title('Gradient Matching Loss')
        ax1.set_yscale('log')
        ax1.grid(True)

        # Cosine similarity
        ax2.plot(iterations, cosine_sims)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Cosine Similarity')
        ax2.set_title('Gradient Cosine Similarity')
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.close()


if __name__ == "__main__":
    # Test gradient matching metrics
    print("Testing gradient matching metrics...")

    # Create dummy gradients
    grad1 = {
        'weight1': torch.randn(10, 10),
        'bias1': torch.randn(10),
        'weight2': torch.randn(10, 5)
    }

    grad2 = {
        'weight1': grad1['weight1'] + torch.randn_like(grad1['weight1']) * 0.1,
        'bias1': grad1['bias1'] + torch.randn_like(grad1['bias1']) * 0.1,
        'weight2': grad1['weight2'] + torch.randn_like(grad1['weight2']) * 0.1
    }

    # Test metrics
    print("\n" + format_gradient_matching_report(grad1, grad2))

    # Test tracker
    print("\nTesting GradientMatchingTracker...")
    tracker = GradientMatchingTracker(grad1)

    for i in range(10):
        dummy_grad = {k: v + torch.randn_like(v) * 0.1 * (1 - i/10) for k, v in grad1.items()}
        tracker.track(dummy_grad, i)

    print("\nTracker Summary:")
    summary = tracker.get_summary()
    for k, v in summary.items():
        print(f"  {k}: {v:.6f}")
