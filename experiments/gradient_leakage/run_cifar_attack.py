"""
Run gradient leakage attack on CIFAR-10 dataset.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import yaml
import argparse
from pathlib import Path

from models.simple_cnn import SimpleCNN
from data.data_loaders import get_cifar10_loader
from data.preparation import prepare_ground_truth_gradients
from attacks.dlg import dlg_with_multiple_restarts
from attacks.dlg_adam import dlg_adam
from metrics.reconstruction_quality import compute_reconstruction_metrics
from utils.visualization import plot_image_reconstruction, create_reconstruction_gallery
from utils.experiment_logger import ExperimentLogger, create_results_table


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_cifar_attack(
    config: dict,
    num_samples: int = 10,
    attack_type: str = 'dlg',
    num_restarts: int = 5,
    save_results: bool = True
):
    """
    Run gradient leakage attack on CIFAR-10.

    Args:
        config: Configuration dictionary
        num_samples: Number of samples to attack
        attack_type: Type of attack
        num_restarts: Number of random restarts
        save_results: Whether to save results
    """
    # Setup
    device = torch.device(config['experiment']['device'])
    print(f"Using device: {device}")

    logger = ExperimentLogger(
        log_dir=config['experiment']['log_dir'],
        experiment_name=f"cifar10_{attack_type}_attack"
    )

    # Log config
    logger.log_config({
        'dataset': 'CIFAR-10',
        'attack_type': attack_type,
        'num_samples': num_samples,
        'num_restarts': num_restarts
    })

    # Load model (3 channels for CIFAR-10)
    print("\nLoading model...")
    model_config = config['model']['cnn']
    model = SimpleCNN(
        input_channels=3,  # CIFAR-10 has 3 channels
        num_classes=10,
        **model_config
    ).to(device)
    model.eval()

    # Load data
    print("\nLoading CIFAR-10 data...")
    data_dir = config['data']['data_dir']
    dataloader = get_cifar10_loader(
        data_dir=data_dir,
        batch_size=1,
        num_samples=num_samples,
        shuffle=False
    )

    # Run attacks
    print(f"\nRunning {attack_type.upper()} attack on CIFAR-10...")
    print("="*60)

    all_results = []

    for sample_idx, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        print(f"\n--- Sample {sample_idx + 1}/{num_samples} ---")
        print(f"True label: {y.item()}")

        # Compute gradients
        true_gradients, _ = prepare_ground_truth_gradients(
            model, x, y, device=device
        )

        # Run attack
        if attack_type == 'dlg':
            result = dlg_with_multiple_restarts(
                true_gradients=true_gradients,
                model=model,
                input_shape=(3, 32, 32),  # CIFAR-10 shape
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
                input_shape=(3, 32, 32),
                num_classes=10,
                num_iterations=config['attack']['num_iterations'],
                lr=config['attack']['adam']['lr'],
                device=device,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # Evaluate
        metrics = compute_reconstruction_metrics(
            x, result.reconstructed_x.to(device),
            y, result.reconstructed_y.to(device)
        )

        print(f"Reconstructed label: {result.reconstructed_y.item()}")
        print(f"Label match: {'✓' if metrics['label_match'] else '✗'}")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")

        # Log
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

        # Visualize
        if save_results:
            output_dir = Path(config['data']['output_dir']) / 'cifar10'
            output_dir.mkdir(parents=True, exist_ok=True)

            fig = plot_image_reconstruction(
                x.cpu(),
                result.reconstructed_x.cpu(),
                y.item(),
                result.reconstructed_y.item(),
                metrics=metrics,
                save_path=str(output_dir / f"sample_{sample_idx}.png")
            )

        all_results.append({
            'original': x.cpu(),
            'reconstructed': result.reconstructed_x.cpu(),
            'original_label': y.item(),
            'reconstructed_label': result.reconstructed_y.item(),
            'metrics': metrics
        })

    # Gallery
    if save_results and all_results:
        comparison_dir = Path(config['data']['comparison_dir']) / 'cifar10'
        create_reconstruction_gallery(all_results, save_dir=comparison_dir)

    # Summary
    logger.save_summary()

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print("\nResults:")
    print(create_results_table(logger.results))

    return logger.results


def main():
    parser = argparse.ArgumentParser(description='Run gradient leakage attack on CIFAR-10')
    parser.add_argument('--config', type=str, default='config/attack_config.yaml')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--attack', type=str, default='dlg', choices=['dlg', 'adam'])
    parser.add_argument('--restarts', type=int, default=5)
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config['experiment']['device'] = args.device

    run_cifar_attack(
        config=config,
        num_samples=args.num_samples,
        attack_type=args.attack,
        num_restarts=args.restarts
    )


if __name__ == "__main__":
    main()
