"""
Analyze defense sensitivity - find breaking points for different defenses.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import yaml
from pathlib import Path

from models.simple_cnn import SimpleCNN
from data.data_loaders import get_mnist_loader
from data.preparation import compute_gradients
from attacks.dlg import dlg_lbfgs
from defenses.dp_noise import DPDefense
from defenses.gradient_compression import SparsifiedGradientDefense
from metrics.reconstruction_quality import compute_reconstruction_metrics
from utils.experiment_logger import ExperimentLogger


def analyze_dp_sensitivity(
    model: torch.nn.Module,
    test_samples: list,
    sigma_values: list,
    num_iterations: int = 500,
    device: torch.device = torch.device('cpu')
) -> dict:
    """
    Analyze DP noise sensitivity.

    Args:
        model: Target model
        test_samples: List of (x, y) samples
        sigma_values: Sigma values to test
        num_iterations: Attack iterations
        device: Device

    Returns:
        Results dictionary
    """
    print("\n" + "="*60)
    print("DP NOISE SENSITIVITY ANALYSIS")
    print("="*60)

    results = {
        'sigma_values': [],
        'label_accuracies': [],
        'mse_values': [],
        'ssim_values': [],
        'psnr_values': []
    }

    for sigma in sigma_values:
        print(f"\nTesting sigma={sigma:.4f}")

        defense = DPDefense(noise_type='gaussian', sigma=sigma)

        label_correct = 0
        mse_list = []
        ssim_list = []
        psnr_list = []

        for x, y in test_samples:
            x = x.to(device)
            y = y.to(device)

            # Compute gradients
            true_grads = compute_gradients(model, x, y)

            # Add noise
            noisy_grads = defense.add_noise(true_grads)

            # Attack
            result = dlg_lbfgs(
                true_gradients=noisy_grads,
                model=model,
                input_shape=(1, 28, 28),
                num_classes=10,
                num_iterations=num_iterations,
                device=device,
                verbose=False
            )

            # Evaluate
            metrics = compute_reconstruction_metrics(
                x, result.reconstructed_x.to(device),
                y, result.reconstructed_y.to(device)
            )

            if metrics['label_match']:
                label_correct += 1

            mse_list.append(metrics['mse'])
            ssim_list.append(metrics['ssim'])
            psnr_list.append(metrics['psnr'])

        # Average
        label_acc = label_correct / len(test_samples)
        avg_mse = np.mean(mse_list)
        avg_ssim = np.mean(ssim_list)
        avg_psnr = np.mean(psnr_list)

        results['sigma_values'].append(sigma)
        results['label_accuracies'].append(label_acc)
        results['mse_values'].append(avg_mse)
        results['ssim_values'].append(avg_ssim)
        results['psnr_values'].append(avg_psnr)

        print(f"  Label Accuracy: {label_acc:.2%}")
        print(f"  MSE: {avg_mse:.6f}")
        print(f"  SSIM: {avg_ssim:.4f}")

    return results


def analyze_compression_sensitivity(
    model: torch.nn.Module,
    test_samples: list,
    sparsity_values: list,
    num_iterations: int = 500,
    device: torch.device = torch.device('cpu')
) -> dict:
    """
    Analyze gradient compression sensitivity.

    Args:
        model: Target model
        test_samples: List of samples
        sparsity_values: Sparsity values to test
        num_iterations: Attack iterations
        device: Device

    Returns:
        Results dictionary
    """
    print("\n" + "="*60)
    print("GRADIENT COMPRESSION SENSITIVITY ANALYSIS")
    print("="*60)

    results = {
        'sparsity_values': [],
        'label_accuracies': [],
        'mse_values': [],
        'ssim_values': [],
        'psnr_values': []
    }

    for sparsity in sparsity_values:
        print(f"\nTesting sparsity={sparsity:.2f}")

        defense = SparsifiedGradientDefense(method='topk', sparsity=sparsity)

        label_correct = 0
        mse_list = []
        ssim_list = []
        psnr_list = []

        for x, y in test_samples:
            x = x.to(device)
            y = y.to(device)

            # Compute gradients
            true_grads = compute_gradients(model, x, y)

            # Compress
            compressed_grads = defense.apply(true_grads)

            # Attack
            result = dlg_lbfgs(
                true_gradients=compressed_grads,
                model=model,
                input_shape=(1, 28, 28),
                num_classes=10,
                num_iterations=num_iterations,
                device=device,
                verbose=False
            )

            # Evaluate
            metrics = compute_reconstruction_metrics(
                x, result.reconstructed_x.to(device),
                y, result.reconstructed_y.to(device)
            )

            if metrics['label_match']:
                label_correct += 1

            mse_list.append(metrics['mse'])
            ssim_list.append(metrics['ssim'])
            psnr_list.append(metrics['psnr'])

        # Average
        label_acc = label_correct / len(test_samples)
        avg_mse = np.mean(mse_list)
        avg_ssim = np.mean(ssim_list)
        avg_psnr = np.mean(psnr_list)

        results['sparsity_values'].append(sparsity)
        results['label_accuracies'].append(label_acc)
        results['mse_values'].append(avg_mse)
        results['ssim_values'].append(avg_ssim)
        results['psnr_values'].append(avg_psnr)

        print(f"  Label Accuracy: {label_acc:.2%}")
        print(f"  MSE: {avg_mse:.6f}")
        print(f"  SSIM: {avg_ssim:.4f}")

    return results


def plot_sensitivity_analysis(
    dp_results: dict,
    compression_results: dict,
    save_dir: str
):
    """
    Plot sensitivity analysis results.

    Args:
        dp_results: DP analysis results
        compression_results: Compression analysis results
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # DP results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Label accuracy vs sigma
    axes[0, 0].plot(dp_results['sigma_values'], dp_results['label_accuracies'],
                    marker='o', linewidth=2)
    axes[0, 0].set_xlabel('Sigma (Noise Level)', fontsize=12)
    axes[0, 0].set_ylabel('Label Accuracy', fontsize=12)
    axes[0, 0].set_title('DP Defense: Label Recovery vs Noise', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.1])

    # MSE vs sigma
    axes[0, 1].plot(dp_results['sigma_values'], dp_results['mse_values'],
                    marker='s', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Sigma (Noise Level)', fontsize=12)
    axes[0, 1].set_ylabel('MSE', fontsize=12)
    axes[0, 1].set_title('DP Defense: Reconstruction Quality vs Noise', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')

    # Label accuracy vs sparsity
    axes[1, 0].plot(compression_results['sparsity_values'],
                    compression_results['label_accuracies'],
                    marker='o', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Sparsity (Fraction Kept)', fontsize=12)
    axes[1, 0].set_ylabel('Label Accuracy', fontsize=12)
    axes[1, 0].set_title('Compression Defense: Label Recovery vs Sparsity', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.1])

    # MSE vs sparsity
    axes[1, 1].plot(compression_results['sparsity_values'],
                    compression_results['mse_values'],
                    marker='s', linewidth=2, color='red')
    axes[1, 1].set_xlabel('Sparsity (Fraction Kept)', fontsize=12)
    axes[1, 1].set_ylabel('MSE', fontsize=12)
    axes[1, 1].set_title('Compression Defense: Reconstruction Quality vs Sparsity', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_dir / 'defense_sensitivity.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_dir / 'defense_sensitivity.png'}")
    plt.close()

    # Find breaking points
    print("\n" + "="*60)
    print("BREAKING POINT ANALYSIS")
    print("="*60)

    # DP breaking point (where accuracy drops below 50%)
    for i, acc in enumerate(dp_results['label_accuracies']):
        if acc < 0.5:
            print(f"\nDP Defense breaking point:")
            print(f"  Sigma >= {dp_results['sigma_values'][i]:.4f} reduces success to <50%")
            print(f"  At this level: Accuracy={acc:.2%}, MSE={dp_results['mse_values'][i]:.6f}")
            break

    # Compression breaking point
    for i, acc in enumerate(compression_results['label_accuracies']):
        if acc < 0.5:
            print(f"\nCompression Defense breaking point:")
            print(f"  Sparsity <= {compression_results['sparsity_values'][i]:.2f} reduces success to <50%")
            print(f"  At this level: Accuracy={acc:.2%}, MSE={compression_results['mse_values'][i]:.6f}")
            break


def main():
    parser = argparse.ArgumentParser(description='Analyze defense sensitivity')
    parser.add_argument('--config', type=str, default='config/attack_config.yaml')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--num-iterations', type=int, default=500)
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device or config['experiment']['device'])
    print(f"Using device: {device}")

    # Setup logger
    logger = ExperimentLogger(
        log_dir=config['experiment']['log_dir'],
        experiment_name='defense_sensitivity'
    )

    # Load model
    print("\nLoading model...")
    model_config = config['model']['cnn']
    model = SimpleCNN(
        input_channels=1,
        num_classes=10,
        **model_config
    ).to(device)
    model.eval()

    # Load test samples
    print(f"\nLoading {args.num_samples} test samples...")
    data_dir = config['data']['data_dir']
    dataloader = get_mnist_loader(
        data_dir=data_dir,
        batch_size=1,
        num_samples=args.num_samples,
        shuffle=False
    )
    test_samples = [(x, y) for x, y in dataloader]

    # Test DP sensitivity
    sigma_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    dp_results = analyze_dp_sensitivity(
        model, test_samples, sigma_values,
        args.num_iterations, device
    )

    # Test compression sensitivity
    sparsity_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    compression_results = analyze_compression_sensitivity(
        model, test_samples, sparsity_values,
        args.num_iterations, device
    )

    # Plot results
    results_dir = Path(config['experiment']['results_dir'])
    plot_sensitivity_analysis(dp_results, compression_results, results_dir)

    # Save results
    dp_df = pd.DataFrame(dp_results)
    compression_df = pd.DataFrame(compression_results)

    dp_df.to_csv(results_dir / 'dp_sensitivity.csv', index=False)
    compression_df.to_csv(results_dir / 'compression_sensitivity.csv', index=False)

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
