"""
Experiment logging utilities.
Track and save gradient leakage attack experiments.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import torch


class ExperimentLogger:
    """Log and save experiment results."""

    def __init__(self, log_dir: str, experiment_name: str):
        """
        Initialize logger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.start_time = datetime.now()

        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)

        # Log file
        self.log_file = self.experiment_dir / "log.txt"

        # Results file
        self.results_file = self.experiment_dir / "results.json"

        # CSV file for structured data
        self.csv_file = self.experiment_dir / "results.csv"

        # Initialize
        self._write_log(f"Experiment: {experiment_name}")
        self._write_log(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Results storage
        self.results = []

    def _write_log(self, message: str):
        """Write message to log file."""
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
        print(message)

    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self._write_log("\n" + "="*50)
        self._write_log("Configuration:")
        for key, value in config.items():
            self._write_log(f"  {key}: {value}")
        self._write_log("="*50 + "\n")

        # Save config as JSON
        config_file = self.experiment_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def log_result(self, result: Dict[str, Any], iteration: Optional[int] = None):
        """
        Log a single result.

        Args:
            result: Result dictionary
            iteration: Optional iteration number
        """
        if iteration is not None:
            result['iteration'] = iteration

        self.results.append(result)

        # Write to log
        self._write_log(f"Result {len(self.results)}:")
        for key, value in result.items():
            if isinstance(value, float):
                self._write_log(f"  {key}: {value:.6f}")
            else:
                self._write_log(f"  {key}: {value}")

    def log_reconstruction_result(
        self,
        sample_idx: int,
        original_label: int,
        reconstructed_label: int,
        label_match: bool,
        metrics: Dict[str, float],
        final_matching_loss: float,
        convergence_iterations: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a reconstruction result with standard fields.

        Args:
            sample_idx: Sample index
            original_label: Original label
            reconstructed_label: Reconstructed label
            label_match: Whether labels match
            metrics: Reconstruction quality metrics
            final_matching_loss: Final gradient matching loss
            convergence_iterations: Number of iterations to converge
            metadata: Additional metadata
        """
        result = {
            'sample_idx': sample_idx,
            'original_label': original_label,
            'reconstructed_label': reconstructed_label,
            'label_match': label_match,
            'final_matching_loss': final_matching_loss,
            'convergence_iterations': convergence_iterations,
            'metadata': metadata or {}
        }

        # Add metrics
        result.update(metrics)

        self.log_result(result)

    def save_summary(self):
        """Save summary of all results."""
        if len(self.results) == 0:
            self._write_log("No results to save.")
            return

        # Compute summary statistics
        label_matches = [r.get('label_match', False) for r in self.results]
        label_accuracy = sum(label_matches) / len(label_matches)

        mse_values = [r.get('mse', 0) for r in self.results if 'mse' in r]
        ssim_values = [r.get('ssim', 0) for r in self.results if 'ssim' in r]
        psnr_values = [r.get('psnr', 0) for r in self.results if 'psnr' in r]

        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_samples': len(self.results),
            'label_accuracy': label_accuracy,
            'num_correct_labels': sum(label_matches),
            'mse_mean': float(np.mean(mse_values)) if mse_values else 0,
            'mse_std': float(np.std(mse_values)) if mse_values else 0,
            'ssim_mean': float(np.mean(ssim_values)) if ssim_values else 0,
            'ssim_std': float(np.std(ssim_values)) if ssim_values else 0,
            'psnr_mean': float(np.mean(psnr_values)) if psnr_values else 0,
            'psnr_std': float(np.std(psnr_values)) if psnr_values else 0,
            'results': self.results
        }

        # Save as JSON
        with open(self.results_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save as CSV
        self._save_csv()

        # Log summary
        self._write_log("\n" + "="*50)
        self._write_log("Summary:")
        self._write_log(f"  Total samples: {summary['total_samples']}")
        self._write_log(f"  Label accuracy: {label_accuracy:.2%}")
        self._write_log(f"  MSE: {summary['mse_mean']:.6f} ± {summary['mse_std']:.6f}")
        self._write_log(f"  SSIM: {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")
        self._write_log(f"  PSNR: {summary['psnr_mean']:.2f} ± {summary['psnr_std']:.2f} dB")
        self._write_log("="*50)

    def _save_csv(self):
        """Save results as CSV."""
        if len(self.results) == 0:
            return

        # Get all unique keys
        all_keys = set()
        for result in self.results:
            all_keys.update(result.keys())
            if 'metadata' in result and isinstance(result['metadata'], dict):
                all_keys.update([f"metadata_{k}" for k in result['metadata'].keys()])

        all_keys = sorted(all_keys)

        # Write CSV
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()

            for result in self.results:
                # Flatten metadata
                row = result.copy()
                if 'metadata' in result and isinstance(result['metadata'], dict):
                    for key, value in result['metadata'].items():
                        row[f'metadata_{key}'] = value
                    del row['metadata']

                # Convert tensors to strings
                for key, value in row.items():
                    if isinstance(value, torch.Tensor):
                        row[key] = f"Tensor{list(value.shape)}"
                    elif not isinstance(value, (str, int, float, bool, type(None))):
                        row[key] = str(value)

                writer.writerow(row)

    def save_checkpoint(self, data: Dict[str, Any], name: str):
        """
        Save checkpoint data.

        Args:
            data: Data to save
            name: Checkpoint name
        """
        checkpoint_file = self.experiment_dir / f"{name}.pt"
        torch.save(data, checkpoint_file)
        self._write_log(f"Checkpoint saved: {checkpoint_file}")

    def load_checkpoint(self, name: str) -> Dict[str, Any]:
        """
        Load checkpoint data.

        Args:
            name: Checkpoint name

        Returns:
            Checkpoint data
        """
        checkpoint_file = self.experiment_dir / f"{name}.pt"
        data = torch.load(checkpoint_file)
        self._write_log(f"Checkpoint loaded: {checkpoint_file}")
        return data


def create_results_table(
    results: List[Dict[str, Any]],
    metrics: List[str] = ['mse', 'ssim', 'psnr', 'label_accuracy']
) -> str:
    """
    Create a formatted results table.

    Args:
        results: List of result dictionaries
        metrics: Metrics to include in table

    Returns:
        Formatted table string
    """
    if not results:
        return "No results"

    lines = []
    lines.append("=" * 100)
    lines.append(f"{'Sample':<10} {'Orig':<6} {'Rec':<6} {'Match':<8}", end="")

    for metric in metrics:
        lines[-1] += f" {metric[:8]:<12}"

    lines.append("=" * 100)

    for result in results:
        sample_idx = result.get('sample_idx', result.get('iteration', '?'))
        orig_label = result.get('original_label', '?')
        rec_label = result.get('reconstructed_label', '?')
        match = '✓' if result.get('label_match', False) else '✗'

        line = f"{sample_idx:<10} {orig_label:<6} {rec_label:<6} {match:<8}"

        for metric in metrics:
            value = result.get(metric, 0)
            if isinstance(value, float):
                if metric == 'mse':
                    line += f" {value:.6f}   "
                elif metric == 'ssim':
                    line += f" {value:.4f}   "
                elif metric == 'psnr':
                    line += f" {value:.2f} dB "
                else:
                    line += f" {value:.4f}   "
            else:
                line += f" {str(value):<12}"

        lines.append(line)

    lines.append("=" * 100)

    return "\n".join(lines)


# Import numpy for summary stats
import numpy as np


if __name__ == "__main__":
    # Test logger
    print("Testing ExperimentLogger...")

    logger = ExperimentLogger(
        log_dir="logs",
        experiment_name="test_experiment"
    )

    # Log config
    config = {
        'model': 'simple_cnn',
        'dataset': 'mnist',
        'num_samples': 10,
        'num_iterations': 1000
    }
    logger.log_config(config)

    # Log some results
    for i in range(5):
        result = {
            'sample_idx': i,
            'original_label': i,
            'reconstructed_label': i,
            'label_match': True,
            'mse': 0.01 + i * 0.001,
            'ssim': 0.95 - i * 0.01,
            'psnr': 30.0 - i * 0.5,
            'final_matching_loss': 1e-6 * (i + 1),
            'convergence_iterations': 500 + i * 10
        }
        logger.log_reconstruction_result(**result)

    # Save summary
    logger.save_summary()

    print("\nResults table:")
    print(create_results_table(logger.results))

    print(f"\nLog files saved to: {logger.experiment_dir}")
