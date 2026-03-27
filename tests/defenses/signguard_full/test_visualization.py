"""Tests for visualization utilities."""

import pytest
import sys

# Make matplotlib optional for tests
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    np = None

from signguard.utils.visualization import (
    create_table_from_results,
)

pytestmark = pytest.mark.skipif(
    not HAS_MATPLOTLIB,
    reason="Matplotlib not installed (optional dependency)"
)


class TestTableGeneration:
    """Tests for LaTeX table generation."""

    def test_create_table_from_results(self):
        """Test creating LaTeX table."""
        results = {
            "FedAvg": {"No Attack": 0.85, "Label Flip": 0.65},
            "SignGuard": {"No Attack": 0.82, "Label Flip": 0.81},
        }
        
        table_str = create_table_from_results(
            results=results,
            caption="Defense Comparison",
            label="tab:comparison",
        )
        
        assert "\\begin{table}" in table_str
        assert "\\caption{Defense Comparison}" in table_str
        assert "FedAvg" in table_str
        assert "SignGuard" in table_str
