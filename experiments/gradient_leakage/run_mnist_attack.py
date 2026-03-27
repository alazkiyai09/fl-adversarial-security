"""
Run gradient leakage attack on MNIST dataset.
Complete end-to-end experiment.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import yaml
import argparse
from pathlib import Path

from models.simple_cnn import SimpleCNN
from data.data_loaders import get_mnist_loader
from data.preparation import prepare_ground_truth_gradients
from attacks.dlg import dlg_with_multiple_restarts
from attacks.dlg_adam import dlg_adam
from attacks.dlg_cosine import dlg_cosine
from metrics.reconstruction_quality import compute_reconstruction_metrics, format_metrics_report
from metrics.gradient_matching import format_gradient_matching_report
from utils.visualization import (
    plot_image_reconstruction,
    plot_optimization_history,
    create_reconstruction_gallery
)
from utils.experiment_logger import ExperimentLogger, create_results_table


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_attack_experiment(
    config: dict,
    num_samples: int = 10,
    attack_type: str = 'dlg',
    num_restarts: int = 5,
    save_results: bool = True
) -> dict:
    """
    Run gradient leakage attack on MNIST.

    Args:
        config: Configuration dictionary
        num_samples: Number of samples to attack
        attack_type: Type of attack ('dlg', 'adam', 'cosine')
        num_restarts: Number of random restarts for DLG
        save_results: Whether to save results

    Returns:
        Experiment results dictionary
    """
    # Setup device
    device = torch.device(config['experiment']['device'])
    print(f"Using device: {device}")

    # Initialize logger
    logger = ExperimentLogger(
        log_dir=config['experiment']['log_dir'],
        experiment_name=f"mnist_{attack_type}_attack"
    )

    # Log config
    attack_config = {
        'dataset': 'MNIST',
        'attack_type': attack_type,
        'num_samples': num_samples,
        'num_restarts': num_restarts,
        'num_iterations': config['attack']['num_iterations']
    }
    logger.log_config(attack_config)

    # Load model
    print("\nLoading model...")
    model_config = config['model']['cnn']
    model = SimpleCNN(
        input_channels=1,
        num_classes=10,
        **model_config
    ).to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())} parameters")

    # Load data
    print("\nLoading data...")
    data_dir = config['data']['data_dir']
    dataloader = get_mnist_loader(
        data_dir=data_dir,
        batch_size=1,
        num_samples=num_samples,
        shuffle=False
    )

    # Run attacks
    print(f"\nRunning {attack_type.upper()} attack on {num_samples} samples...")
    print("="*60)

    all_results = []

    for sample_idx, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        print(f"\n--- Sample {sample_idx + 1}/{num_samples} ---")
        print(f"True label: {y.item()}")

        # Compute ground-truth gradients
        true_gradients, true_loss = prepare_ground_truth_gradients(
            model, x, y, device=device
        )
        print(f"True loss: {true_loss:.4f}")

        # Run attack
        if attack_type == 'dlg':
            result = dlg_with_multiple_restarts(
                true_gradients=true_gradients,
                model=model,
                input_shape=(1, 28, 28),
                num_classes=10,
                num_restarts=num_restarts,
                num_iterations=config['attack']['num_iterations'],
                device=device,
                verbose=False
            )
        elif attack_type == 'adam':
            result = dlg_adam(
                true_gradients=true_gradients,
                model=model,
                input_shape=(1, 28, 28),
                num_classes=10,
                num_iterations=config['attack']['num_iterations'],
                lr=config['attack']['adam']['lr'],
                device=device,
                verbose=False
            )
        elif attack_type == 'cosine':
            result = dlg_cosine(
                true_gradients=true_gradients,
                model=model,
                input_shape=(1, 28, 28),
                num_classes=10,
                num_iterations=config['attack']['num_iterations'],
                lr=config['attack']['cosine']['lr'],
                device=device,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # Evaluate reconstruction
        metrics = compute_reconstruction_metrics(
            x, result.reconstructed_x.to(device),
            y, result.reconstructed_y.to(device)
        )

        # Print results
        print(f"Reconstructed label: {result.reconstructed_y.item()}")
        print(f"Label match: {'✓' if metrics['label_match'] else '✗'}")
        print(f"\nMetrics:")
        print(f"  Final matching loss: {result.final_matching_loss:.6e}")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")

        # Save individual results
        logger.log_reconstruction_result(
            sample_idx=sample_idx,
            original_label=y.item(),
            reconstructed_label=result.reconstructed_y.item(),
            label_match=metrics['label_match'],
            metrics=metrics,
            final_matching_loss=result.final_matching_loss,
            convergence_iterations=result.convergence_iterations,
            metadata=result.metadata
        )

        # Visualize if saving
        if save_results:
            output_dir = Path(config['data']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save comparison plot
            fig = plot_image_reconstruction(
                x.cpu(),
                result.reconstructed_x.cpu(),
                y.item(),
                result.reconstructed_y.item(),
                metrics=metrics,
                save_path=str(output_dir / f"sample_{sample_idx}_comparison.png")
            )

            # Save optimization history
            if len(result.gradient_distances) > 0:
                fig = plot_optimization_history(
                    result.gradient_distances,
                    save_path=str(output_dir / f"sample_{sample_idx}_optimization.png")
                )

        # Store for gallery
        all_results.append({
            'original': x.cpu(),
            'reconstructed': result.reconstructed_x.cpu(),
            'original_label': y.item(),
            'reconstructed_label': result.reconstructed_y.item(),
            'metrics': metrics
        })

    # Create gallery
    if save_results and all_results:
        comparison_dir = Path(config['data']['comparison_dir'])
        create_reconstruction_gallery(
            all_results,
            save_dir=comparison_dir,
            max_samples=num_samples
        )

    # Print summary
    logger.save_summary()

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print("\nResults:")
    print(create_results_table(logger.results))

    return {
        'results': logger.results,
        'summary': logger.get_summary() if hasattr(logger, 'get_summary') else {}
    }


def main():
    parser = argparse.ArgumentParser(description='Run gradient leakage attack on MNIST')
    parser.add_argument('--config', type=str, default='config/attack_config.yaml',
                        help='Path to config file')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to attack')
    parser.add_argument('--attack', type=str, default='dlg',
                        choices=['dlg', 'adam', 'cosine'],
                        help='Attack type')
    parser.add_argument('--restarts', type=int, default=5,
                        help='Number of random restarts for DLG')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu/cuda)')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override device if specified
    if args.device:
        config['experiment']['device'] = args.device

    # Run experiment
    results = run_attack_experiment(
        config=config,
        num_samples=args.num_samples,
        attack_type=args.attack,
        num_restarts=args.restarts
    )

    return results


if __name__ == "__main__":
    results = main()
