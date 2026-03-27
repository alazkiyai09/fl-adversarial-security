"""
Visualization utilities for gradient leakage attack results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def plot_image_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    original_label: int,
    reconstructed_label: int,
    save_path: Optional[str] = None,
    show_metrics: bool = True,
    metrics: Optional[Dict[str, float]] = None
) -> plt.Figure:
    """
    Plot side-by-side comparison of original and reconstructed images.

    Args:
        original: Original image [C, H, W] or [H, W]
        reconstructed: Reconstructed image [C, H, W] or [H, W]
        original_label: Original label
        reconstructed_label: Reconstructed label
        save_path: Optional path to save figure
        show_metrics: Whether to show metrics
        metrics: Optional metrics to display

    Returns:
        Figure object
    """
    # Convert to numpy
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

    # Squeeze batch and channel dimensions
    if original.ndim == 4:
        original = original[0]
    if reconstructed.ndim == 4:
        reconstructed = reconstructed[0]

    if original.ndim == 3 and original.shape[0] == 1:
        original = original[0]
        reconstructed = reconstructed[0]

    # For RGB, transpose
    if original.ndim == 3:
        original = np.transpose(original, (1, 2, 0))
        reconstructed = np.transpose(reconstructed, (1, 2, 0))

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Original
    axes[0].imshow(original, cmap='gray' if original.ndim == 2 else None)
    axes[0].set_title(f'Original\nLabel: {original_label}', fontsize=14)
    axes[0].axis('off')

    # Reconstructed
    axes[1].imshow(reconstructed, cmap='gray' if reconstructed.ndim == 2 else None)

    title = f'Reconstructed\nLabel: {reconstructed_label}'
    if reconstructed_label == original_label:
        title += ' ✓'
    else:
        title += ' ✗'

    axes[1].set_title(title, fontsize=14)
    axes[1].axis('off')

    # Add metrics if provided
    if show_metrics and metrics:
        metrics_text = "Metrics:\n"
        for key, value in metrics.items():
            if key in ['mse', 'psnr', 'ssim', 'label_accuracy']:
                if key == 'mse':
                    metrics_text += f"MSE: {value:.6f}\n"
                elif key == 'psnr':
                    metrics_text += f"PSNR: {value:.2f} dB\n"
                elif key == 'ssim':
                    metrics_text += f"SSIM: {value:.4f}\n"

        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_optimization_history(
    gradient_distances: List[float],
    save_path: Optional[str] = None,
    title: str = "Gradient Matching Loss"
) -> plt.Figure:
    """
    Plot optimization history (gradient distance over iterations).

    Args:
        gradient_distances: List of gradient distances
        save_path: Optional path to save figure
        title: Plot title

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    iterations = range(len(gradient_distances))
    ax.plot(iterations, gradient_distances, linewidth=2)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Gradient Distance', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Add final value
    final_value = gradient_distances[-1]
    ax.text(0.98, 0.02, f'Final: {final_value:.6e}',
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_multiple_reconstructions(
    originals: List[torch.Tensor],
    reconstructed: List[torch.Tensor],
    original_labels: List[int],
    reconstructed_labels: List[int],
    metrics_list: Optional[List[Dict[str, float]]] = None,
    save_path: Optional[str] = None,
    nrows: int = 2
) -> plt.Figure:
    """
    Plot multiple reconstruction comparisons in a grid.

    Args:
        originals: List of original images
        reconstructed: List of reconstructed images
        original_labels: List of original labels
        reconstructed_labels: List of reconstructed labels
        metrics_list: Optional list of metrics for each sample
        save_path: Optional path to save figure
        nrows: Number of rows in grid

    Returns:
        Figure object
    """
    num_samples = len(originals)
    ncols = min(4, num_samples)

    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 4, nrows * 3))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(num_samples):
        row = idx // ncols
        col = (idx % ncols) * 2

        # Convert to numpy
        orig = originals[idx].detach().cpu().numpy()
        rec = reconstructed[idx].detach().cpu().numpy()

        # Squeeze dimensions
        if orig.ndim == 4:
            orig = orig[0]
        if rec.ndim == 4:
            rec = rec[0]

        if orig.ndim == 3 and orig.shape[0] == 1:
            orig = orig[0]
            rec = rec[0]

        if orig.ndim == 3:
            orig = np.transpose(orig, (1, 2, 0))
            rec = np.transpose(rec, (1, 2, 0))

        # Original
        axes[row, col].imshow(orig, cmap='gray' if orig.ndim == 2 else None)
        axes[row, col].set_title(f'Orig: {original_labels[idx]}', fontsize=10)
        axes[row, col].axis('off')

        # Reconstructed
        match = "✓" if reconstructed_labels[idx] == original_labels[idx] else "✗"
        axes[row, col + 1].imshow(rec, cmap='gray' if rec.ndim == 2 else None)
        axes[row, col + 1].set_title(f'Rec: {reconstructed_labels[idx]} {match}', fontsize=10)
        axes[row, col + 1].axis('off')

    # Hide unused subplots
    for idx in range(num_samples, nrows * ncols):
        row = idx // ncols
        col = (idx % ncols) * 2
        axes[row, col].axis('off')
        axes[row, col + 1].axis('off')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_label_distribution(
    original_labels: torch.Tensor,
    reconstructed_labels: torch.Tensor,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot label distribution comparison.

    Args:
        original_labels: Original labels
        reconstructed_labels: Reconstructed labels
        save_path: Optional path to save

    Returns:
        Figure object
    """
    if isinstance(original_labels, torch.Tensor):
        original_labels = original_labels.cpu().numpy()
    if isinstance(reconstructed_labels, torch.Tensor):
        reconstructed_labels = reconstructed_labels.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Original labels
    unique_orig, counts_orig = np.unique(original_labels, return_counts=True)
    axes[0].bar(unique_orig, counts_orig)
    axes[0].set_xlabel('Label')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Original Label Distribution')
    axes[0].grid(True, alpha=0.3)

    # Reconstructed labels
    unique_rec, counts_rec = np.unique(reconstructed_labels, return_counts=True)
    axes[1].bar(unique_rec, counts_rec)
    axes[1].set_xlabel('Label')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Reconstructed Label Distribution')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def create_reconstruction_gallery(
    results: List[Dict],
    save_dir: str,
    max_samples: int = 20
) -> None:
    """
    Create a gallery of reconstruction results.

    Args:
        results: List of reconstruction result dictionaries
        save_dir: Directory to save gallery
        max_samples: Maximum number of samples to show
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Limit samples
    results = results[:max_samples]

    # Create grid plot
    originals = [r['original'] for r in results]
    reconstructed = [r['reconstructed'] for r in results]
    orig_labels = [r['original_label'] for r in results]
    rec_labels = [r['reconstructed_label'] for r in results]
    metrics = [r.get('metrics', {}) for r in results]

    fig = plot_multiple_reconstructions(
        originals, reconstructed, orig_labels, rec_labels,
        metrics, nrows=(len(results) + 3) // 4
    )

    fig.savefig(save_dir / 'gallery.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Gallery saved to {save_dir / 'gallery.png'}")


def plot_defense_effectiveness(
    defense_strengths: List[float],
    attack_success_rates: List[float],
    defense_name: str = "Defense",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot defense effectiveness (attack success vs defense strength).

    Args:
        defense_strengths: List of defense strength values
        attack_success_rates: Corresponding attack success rates
        defense_name: Name of defense mechanism
        save_path: Optional path to save

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(defense_strengths, attack_success_rates, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel(f'{defense_name} Strength', fontsize=12)
    ax.set_ylabel('Attack Success Rate', fontsize=12)
    ax.set_title(f'{defense_name} Effectiveness Against Gradient Leakage', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    # Add threshold line
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% Success')
    ax.legend()

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


if __name__ == "__main__":
    # Test visualization
    print("Testing visualization utilities...")

    # Create dummy data
    original = torch.rand(1, 28, 28)
    reconstructed = original + torch.randn_like(original) * 0.1

    # Plot reconstruction
    fig = plot_image_reconstruction(
        original, reconstructed,
        original_label=5, reconstructed_label=5,
        save_path="data/comparison/test_reconstruction.png"
    )
    plt.close(fig)

    # Plot optimization history
    distances = [10 ** (-i/100) for i in range(1000)]
    fig = plot_optimization_history(
        distances,
        save_path="data/comparison/test_optimization.png"
    )
    plt.close(fig)

    print("Visualization test complete!")
