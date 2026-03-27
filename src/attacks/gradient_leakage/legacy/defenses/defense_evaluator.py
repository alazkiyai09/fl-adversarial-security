"""
Defense evaluation framework.
Test effectiveness of various defenses against gradient leakage attacks.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Optional
from pathlib import Path

from ..data.preparation import compute_gradients
from ..metrics.reconstruction_quality import compute_reconstruction_metrics
from ..attacks.dlg import dlg_lbfgs
from ..attacks.dlg_adam import dlg_adam
from ..dp_noise import DPDefense
from ..gradient_compression import SparsifiedGradientDefense


class DefenseEvaluator:
    """
    Evaluate effectiveness of defenses against gradient leakage.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize evaluator.

        Args:
            model: Target model
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device
        self.results = []

    def evaluate_defense(
        self,
        defense: Callable,
        defense_name: str,
        test_samples: List[Tuple[torch.Tensor, torch.Tensor]],
        defense_params: Dict[str, float],
        attack_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single defense configuration.

        Args:
            defense: Defense function that takes gradients and returns defended gradients
            defense_name: Name of defense
            test_samples: List of (x, y) test samples
            defense_params: Parameters for defense
            attack_fn: Attack function to use (default: dlg_lbfgs)

        Returns:
            Dictionary of evaluation metrics
        """
        if attack_fn is None:
            attack_fn = dlg_lbfgs

        label_matches = 0
        mse_values = []
        ssim_values = []
        psnr_values = []
        gradient_distances = []

        for x, y in test_samples:
            x = x.to(self.device)
            y = y.to(self.device)

            # Compute true gradients
            true_gradients = compute_gradients(self.model, x, y)

            # Apply defense
            defended_gradients = defense(true_gradients, **defense_params)

            # Run attack
            result = attack_fn(
                defended_gradients,
                self.model,
                input_shape=x.shape[1:],
                num_classes=10,
                verbose=False
            )

            # Evaluate reconstruction
            metrics = compute_reconstruction_metrics(
                x, result.reconstructed_x,
                y, result.reconstructed_y
            )

            label_matches += metrics['label_match']
            mse_values.append(metrics['mse'])
            ssim_values.append(metrics['ssim'])
            psnr_values.append(metrics['psnr'])
            gradient_distances.append(result.final_matching_loss)

        # Compute aggregate metrics
        num_samples = len(test_samples)

        summary = {
            'defense': defense_name,
            'defense_params': defense_params,
            'label_accuracy': label_matches / num_samples,
            'mse_mean': np.mean(mse_values),
            'mse_std': np.std(mse_values),
            'ssim_mean': np.mean(ssim_values),
            'ssim_std': np.std(ssim_values),
            'psnr_mean': np.mean(psnr_values),
            'psnr_std': np.std(psnr_values),
            'gradient_distance_mean': np.mean(gradient_distances),
            'num_samples': num_samples
        }

        self.results.append(summary)

        return summary

    def evaluate_dp_defense(
        self,
        test_samples: List[Tuple[torch.Tensor, torch.Tensor]],
        sigma_values: List[float],
        noise_type: str = 'gaussian'
    ) -> pd.DataFrame:
        """
        Evaluate DP defense across multiple sigma values.

        Args:
            test_samples: Test samples
            sigma_values: List of sigma values
            noise_type: Type of noise ('gaussian', 'laplace')

        Returns:
            DataFrame with results
        """
        print(f"\nEvaluating DP defense ({noise_type})...")

        defense_obj = DPDefense(noise_type=noise_type)

        def defense_fn(grads, sigma):
            defense_obj.sigma = sigma
            return defense_obj.add_noise(grads)

        for sigma in sigma_values:
            print(f"  Testing sigma={sigma:.3f}...")
            summary = self.evaluate_defense(
                defense=defense_fn,
                defense_name=f'dp_{noise_type}',
                test_samples=test_samples,
                defense_params={'sigma': sigma}
            )
            print(f"    Label Accuracy: {summary['label_accuracy']:.2%}")
            print(f"    MSE: {summary['mse_mean']:.6f}")

        return pd.DataFrame(self.results)

    def evaluate_compression_defense(
        self,
        test_samples: List[Tuple[torch.Tensor, torch.Tensor]],
        sparsity_values: List[float],
        method: str = 'topk'
    ) -> pd.DataFrame:
        """
        Evaluate gradient compression defense.

        Args:
            test_samples: Test samples
            sparsity_values: List of sparsity values (fraction to keep)
            method: Compression method

        Returns:
            DataFrame with results
        """
        print(f"\nEvaluating compression defense ({method})...")

        defense_obj = SparsifiedGradientDefense(method=method)

        for sparsity in sparsity_values:
            print(f"  Testing sparsity={sparsity:.2f}...")
            summary = self.evaluate_defense(
                defense=lambda g, **kw: defense_obj.apply(g, **kw),
                defense_name=f'compression_{method}',
                test_samples=test_samples,
                defense_params={'sparsity': sparsity}
            )
            print(f"    Label Accuracy: {summary['label_accuracy']:.2%}")
            print(f"    MSE: {summary['mse_mean']:.6f}")

        return pd.DataFrame(self.results)

    def find_defense_threshold(
        self,
        test_samples: List[Tuple[torch.Tensor, torch.Tensor]],
        defense_type: str = 'dp',
        target_success_rate: float = 0.1
    ) -> float:
        """
        Find minimum defense strength to achieve target success rate.

        Args:
            test_samples: Test samples
            defense_type: Type of defense ('dp' or 'compression')
            target_success_rate: Target maximum attack success rate

        Returns:
            Minimum defense parameter value
        """
        if defense_type == 'dp':
            # Binary search for sigma
            low, high = 0.01, 10.0

            for _ in range(20):
                mid = (low + high) / 2

                defense_obj = DPDefense(sigma=mid)
                summary = self.evaluate_defense(
                    defense=lambda g, **kw: defense_obj.add_noise(g),
                    defense_name='dp_gaussian',
                    test_samples=test_samples[:3],  # Use fewer samples for speed
                    defense_params={}
                )

                if summary['label_accuracy'] <= target_success_rate:
                    high = mid
                else:
                    low = mid

            return high

        elif defense_type == 'compression':
            # Binary search for sparsity
            low, high = 0.0, 1.0

            for _ in range(20):
                mid = (low + high) / 2

                defense_obj = SparsifiedGradientDefense(sparsity=mid)
                summary = self.evaluate_defense(
                    defense=lambda g, **kw: defense_obj.apply(g, **kw),
                    defense_name='compression_topk',
                    test_samples=test_samples[:3],
                    defense_params={}
                )

                if summary['label_accuracy'] <= target_success_rate:
                    low = mid
                else:
                    high = mid

            return low

        else:
            raise ValueError(f"Unknown defense type: {defense_type}")

    def plot_defense_effectiveness(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot defense effectiveness results.

        Args:
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        if not self.results:
            print("No results to plot")
            return None

        df = pd.DataFrame(self.results)

        # Group by defense type
        defense_types = df['defense'].unique()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot label accuracy
        for defense_type in defense_types:
            subset = df[df['defense'] == defense_type]
            axes[0].plot(
                subset.index,
                subset['label_accuracy'],
                marker='o',
                label=defense_type
            )

        axes[0].set_xlabel('Experiment Index', fontsize=12)
        axes[0].set_ylabel('Label Accuracy', fontsize=12)
        axes[0].set_title('Defense Effectiveness (Label Recovery)', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.1])

        # Plot MSE
        for defense_type in defense_types:
            subset = df[df['defense'] == defense_type]
            axes[1].plot(
                subset.index,
                subset['mse_mean'],
                marker='s',
                label=defense_type
            )

        axes[1].set_xlabel('Experiment Index', fontsize=12)
        axes[1].set_ylabel('MSE', fontsize=12)
        axes[1].set_title('Defense Effectiveness (Data Quality)', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        return fig

    def save_results(self, save_path: str):
        """
        Save results to CSV.

        Args:
            save_path: Path to save CSV
        """
        df = pd.DataFrame(self.results)
        df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")


def run_comprehensive_defense_evaluation(
    model: torch.nn.Module,
    test_samples: List[Tuple[torch.Tensor, torch.Tensor]],
    save_dir: str = "results/defenses",
    device: torch.device = torch.device('cpu')
) -> DefenseEvaluator:
    """
    Run comprehensive defense evaluation.

    Args:
        model: Target model
        test_samples: Test samples
        save_dir: Directory to save results
        device: Device

    Returns:
        DefenseEvaluator instance
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    evaluator = DefenseEvaluator(model, device)

    # Evaluate DP defenses
    print("="*60)
    print("Evaluating DP Defenses")
    print("="*60)

    sigma_values = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
    evaluator.evaluate_dp_defense(test_samples, sigma_values, 'gaussian')

    # Save DP results
    evaluator.save_results(save_dir / "dp_defense_results.csv")

    # Clear results for next defense type
    evaluator.results = []

    # Evaluate compression defenses
    print("\n" + "="*60)
    print("Evaluating Compression Defenses")
    print("="*60)

    sparsity_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    evaluator.evaluate_compression_defense(test_samples, sparsity_values, 'topk')

    # Save compression results
    evaluator.save_results(save_dir / "compression_defense_results.csv")

    # Plot results
    fig = evaluator.plot_defense_effectiveness(save_dir / "defense_effectiveness.png")
    if fig is not None:
        plt.close(fig)

    return evaluator


if __name__ == "__main__":
    # Test defense evaluator
    print("Testing DefenseEvaluator...")

    from models.simple_cnn import SimpleCNN
    from data.data_loaders import get_mnist_loader

    # Setup
    device = torch.device('cpu')
    model = SimpleCNN(input_channels=1, num_classes=10).to(device)

    # Get test samples
    loader = get_mnist_loader("./data/raw", batch_size=1, num_samples=5)
    test_samples = [(x, y) for x, y in loader]

    print(f"Test samples: {len(test_samples)}")

    # Run quick evaluation
    evaluator = DefenseEvaluator(model, device)

    # Test DP with a few sigma values
    evaluator.evaluate_dp_defense(
        test_samples[:3],
        sigma_values=[0.1, 0.5, 1.0],
        noise_type='gaussian'
    )

    print("\nEvaluation complete!")
    print(evaluator.results)
