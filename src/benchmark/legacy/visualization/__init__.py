"""
Visualization and reporting for benchmark results.
"""

from .plots import (
    plot_heatmap,
    plot_convergence,
    plot_bar_comparison,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_attacker_fraction_vs_metric,
    plot_non_iid_impact,
    create_summary_figure,
)

from .tables import (
    generate_latex_table,
    generate_comparison_table,
    generate_heatmap_table,
    generate_statistical_table,
    generate_all_tables,
)

from .reports import (
    generate_markdown_report,
    generate_results_summary,
    save_results_json,
    load_results_json,
    generate_experiment_log,
    combine_multiple_runs,
)

__all__ = [
    "plot_heatmap",
    "plot_convergence",
    "plot_bar_comparison",
    "plot_roc_curves",
    "plot_precision_recall_curves",
    "plot_attacker_fraction_vs_metric",
    "plot_non_iid_impact",
    "create_summary_figure",
    "generate_latex_table",
    "generate_comparison_table",
    "generate_heatmap_table",
    "generate_statistical_table",
    "generate_all_tables",
    "generate_markdown_report",
    "generate_results_summary",
    "save_results_json",
    "load_results_json",
    "generate_experiment_log",
    "combine_multiple_runs",
]
