"""
LaTeX table generation for publication-ready results.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path


def generate_latex_table(
    data: Dict[str, List[float]],
    metric_name: str,
    caption: str,
    label: str,
    save_path: str,
    highlight_best: bool = True,
    highlight_worst: bool = False,
    precision: int = 3,
    use_std: bool = True,
) -> None:
    """
    Generate LaTeX table with mean ± std values.

    Args:
        data: Dictionary mapping method names to lists of values
        metric_name: Name of the metric (for column header)
        caption: Table caption
        label: LaTeX label for cross-referencing
        save_path: Path to save .tex file
        highlight_best: Whether to bold the best value
        highlight_worst: Whether to underline the worst value
        precision: Number of decimal places
        use_std: Whether to show standard deviation
    """
    methods = list(data.keys())

    # Compute statistics
    stats = {}
    for method, values in data.items():
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        stats[method] = {"mean": mean, "std": std}

    # Determine best and worst values
    means = {m: s["mean"] for m, s in stats.items()}

    if highlight_best:
        # For accuracy, higher is better; for ASR, lower is better
        if "accuracy" in metric_name.lower() or "auprc" in metric_name.lower():
            best_method = max(means, key=means.get)
        else:
            best_method = min(means, key=means.get)
    else:
        best_method = None

    if highlight_worst:
        if "accuracy" in metric_name.lower() or "auprc" in metric_name.lower():
            worst_method = min(means, key=means.get)
        else:
            worst_method = max(means, key=means.get)
    else:
        worst_method = None

    # Generate LaTeX code
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("    \\centering")
    latex_lines.append("    \\caption{" + caption + "}")
    latex_lines.append("    \\label{" + label + "}")
    latex_lines.append("    \\begin{tabular}{l" + "c" * len(methods) + "}")
    latex_lines.append("        \\toprule")
    latex_lines.append("        Method & " + " & ".join([m.replace("_", " ").title() for m in methods]) + " \\\\")
    latex_lines.append("        \\midrule")

    # Data row
    row_cells = []
    for method in methods:
        mean = stats[method]["mean"]
        std = stats[method]["std"]

        if use_std:
            value_str = f"{mean:.{precision}f} \\pm {std:.{precision}f}"
        else:
            value_str = f"{mean:.{precision}f}"

        # Apply highlighting
        if method == best_method and highlight_best:
            value_str = "\\textbf{" + value_str + "}"
        if method == worst_method and highlight_worst:
            value_str = "\\underline{" + value_str + "}"

        row_cells.append(value_str)

    latex_lines.append("        " + metric_name.replace("_", " ").title() + " & " + " & ".join(row_cells) + " \\\\")

    latex_lines.append("        \\bottomrule")
    latex_lines.append("    \\end{tabular}")
    latex_lines.append("\\end{table}")

    # Write to file
    with open(save_path, "w") as f:
        f.write("\n".join(latex_lines))


def generate_comparison_table(
    results: Dict[str, Dict[str, List[float]]],
    caption: str,
    label: str,
    save_path: str,
    metrics: List[str] = ["clean_accuracy", "asr", "auprc"],
    precision: int = 3,
) -> None:
    """
    Generate comparison table with multiple metrics.

    Args:
        results: Nested dict: {method: {metric: [values]}}
        caption: Table caption
        label: LaTeX label
        save_path: Path to save .tex file
        metrics: List of metrics to include
        precision: Number of decimal places
    """
    methods = list(results.keys())

    # Generate LaTeX code
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("    \\centering")
    latex_lines.append("    \\caption{" + caption + "}")
    latex_lines.append("    \\label{" + label + "}")
    latex_lines.append("    \\begin{tabular}{l" + "c" * len(metrics) + "}")
    latex_lines.append("        \\toprule")
    latex_lines.append("        Method & " + " & ".join([m.replace("_", " ").title() for m in metrics]) + " \\\\")
    latex_lines.append("        \\midrule")

    # Data rows for each method
    for method in methods:
        row_cells = [method.replace("_", " ").title()]

        for metric in metrics:
            if metric in results[method]:
                values = results[method][metric]
                mean = np.mean(values)
                std = np.std(values, ddof=1)
                cell_str = f"{mean:.{precision}f} \\pm {std:.{precision}f}"
            else:
                cell_str = "N/A"

            row_cells.append(cell_str)

        latex_lines.append("        " + " & ".join(row_cells) + " \\\\")

    latex_lines.append("        \\bottomrule")
    latex_lines.append("    \\end{tabular}")
    latex_lines.append("\\end{table}")

    # Write to file
    with open(save_path, "w") as f:
        f.write("\n".join(latex_lines))


def generate_heatmap_table(
    data: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    caption: str,
    label: str,
    save_path: str,
    precision: int = 3,
) -> None:
    """
    Generate LaTeX table for heatmap-style data.

    Args:
        data: 2D array of values
        row_labels: Labels for rows
        col_labels: Labels for columns
        caption: Table caption
        label: LaTeX label
        save_path: Path to save .tex file
        precision: Number of decimal places
    """
    # Generate LaTeX code
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("    \\centering")
    latex_lines.append("    \\caption{" + caption + "}")
    latex_lines.append("    \\label{" + label + "}")
    latex_lines.append("    \\begin{tabular}{l" + "c" * len(col_labels) + "}")
    latex_lines.append("        \\toprule")
    latex_lines.append("         & " + " & ".join(col_labels) + " \\\\")
    latex_lines.append("        \\midrule")

    # Data rows
    for i, row_label in enumerate(row_labels):
        row_cells = [row_label]
        for j in range(len(col_labels)):
            cell_str = f"{data[i, j]:.{precision}f}"
            row_cells.append(cell_str)

        latex_lines.append("        " + " & ".join(row_cells) + " \\\\")

    latex_lines.append("        \\bottomrule")
    latex_lines.append("    \\end{tabular}")
    latex_lines.append("\\end{table}")

    # Write to file
    with open(save_path, "w") as f:
        f.write("\n".join(latex_lines))


def generate_statistical_table(
    comparisons: List[Dict[str, Any]],
    caption: str,
    label: str,
    save_path: str,
) -> None:
    """
    Generate LaTeX table for statistical comparisons.

    Args:
        comparisons: List of comparison dictionaries with t-test results
        caption: Table caption
        label: LaTeX label
        save_path: Path to save .tex file
    """
    # Generate LaTeX code
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("    \\centering")
    latex_lines.append("    \\caption{" + caption + "}")
    latex_lines.append("    \\label{" + label + "}")
    latex_lines.append("    \\begin{tabular}{lccc}")
    latex_lines.append("        \\toprule")
    latex_lines.append("        Comparison & Statistic & p-value & Significant \\\\")
    latex_lines.append("        \\midrule")

    # Data rows
    for comp in comparisons:
        comparison = comp.get("comparison", "N/A")
        statistic = f"{comp.get('statistic', 0):.3f}"
        p_value = comp.get("p_value", 1.0)

        # Format p-value
        if p_value < 0.001:
            p_str = "< 0.001"
        else:
            p_str = f"{p_value:.3f}"

        # Significance indicator
        significant = comp.get("significant", False)
        sig_str = "$^*$" if significant else ""

        latex_lines.append(
            f"        {comparison} & {statistic} & {p_str} & {sig_str} \\\\"
        )

    latex_lines.append("        \\bottomrule")
    latex_lines.append("    \\end{tabular}")
    latex_lines.append("    \\vspace{0.5em}")
    latex_lines.append("    {\\small $^*$ indicates significance at $\\alpha = 0.05$}")
    latex_lines.append("\\end{table}")

    # Write to file
    with open(save_path, "w") as f:
        f.write("\n".join(latex_lines))


def generate_all_tables(
    results: Dict[str, Any],
    output_dir: str,
) -> None:
    """
    Generate all LaTeX tables from results dictionary.

    Args:
        results: Complete results dictionary
        output_dir: Directory to save tables
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Main comparison table
    if "final_metrics" in results:
        comparison_data = {}
        for defense, metrics in results["final_metrics"].items():
            comparison_data[defense] = {
                "clean_accuracy": [metrics.get("clean_accuracy", 0.0)],
                "asr": [metrics.get("asr", 0.0)],
                "auprc": [metrics.get("auprc", 0.0)],
            }

        generate_comparison_table(
            results=comparison_data,
            caption="Comparison of Defense Methods",
            label="tab:comparison",
            save_path=str(output_path / "comparison_table.tex"),
        )

    # Attacker fraction tables
    if "accuracy_vs_fraction" in results:
        for metric_name, data in results["accuracy_vs_fraction"].items():
            generate_latex_table(
                data=data,
                metric_name="Clean Accuracy",
                caption=f"Clean Accuracy vs Attacker Fraction ({metric_name})",
                label=f"tab:acc_vs_fraction_{metric_name}",
                save_path=str(output_path / f"acc_vs_fraction_{metric_name}.tex"),
            )

    # Statistical significance table
    if "statistical_tests" in results:
        generate_statistical_table(
            comparisons=results["statistical_tests"],
            caption="Statistical Significance of Defense Comparisons",
            label="tab:statistical",
            save_path=str(output_path / "statistical_table.tex"),
        )
