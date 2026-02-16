"""Unit tests for the Data Generating Process (DGP) module.

Tests verify that the synthetic DGP produces data with the expected
statistical properties: correct dimensions, treatment assignment,
selection mechanism, and exclusion restriction.

Updated for v4.0 interface (Board Review, Feb 2026).
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.dgp import SyntheticDGP


class TestSyntheticDGP:
    """Tests for SyntheticDGP data generation."""

    def setup_method(self):
        """Initialise DGP with known parameters."""
        np.random.seed(42)
        self.dgp = SyntheticDGP(selection_mode="structural")
        self.panel = self.dgp.generate_panel_data(
            n_individuals=200, n_periods=5
        )
        self.params = self.dgp.get_true_parameters()

    def test_panel_shape(self):
        """Generated panel should have n_individuals * n_periods rows."""
        expected = 200 * 5
        assert len(self.panel) == expected, (
            f"Expected {expected} rows, got {len(self.panel)}"
        )

    def test_required_columns(self):
        """Panel should contain all required columns."""
        required = ["individual_id", "period", "treatment", "outcome",
                     "career_sequence_history"]
        for col in required:
            assert col in self.panel.columns, f"Missing column: {col}"

    def test_treatment_is_binary(self):
        """Treatment should be binary (0 or 1)."""
        unique_vals = self.panel["treatment"].unique()
        assert set(unique_vals).issubset({0, 1}), (
            f"Treatment contains non-binary values: {unique_vals}"
        )

    def test_treatment_has_both_groups(self):
        """Both treatment and control groups should be non-empty."""
        t_sum = self.panel["treatment"].sum()
        assert t_sum > 0, "No treated individuals"
        assert t_sum < len(self.panel), "No control individuals"

    def test_outcome_is_finite(self):
        """All outcomes should be finite (no NaN or Inf)."""
        assert self.panel["outcome"].notna().all(), "Outcome contains NaN"
        assert np.all(np.isfinite(self.panel["outcome"].values)), (
            "Outcome contains Inf values"
        )

    def test_true_parameters(self):
        """get_true_parameters should return a dict with true_ate."""
        assert "true_ate" in self.params, "Missing true_ate in parameters"
        assert isinstance(self.params["true_ate"], (int, float)), (
            "true_ate should be numeric"
        )

    def test_career_sequences_are_lists(self):
        """Career sequence history should be lists of integers."""
        sample = self.panel["career_sequence_history"].iloc[0]
        assert isinstance(sample, list), (
            f"Expected list, got {type(sample)}"
        )

    def test_individual_ids_correct(self):
        """Should have exactly n_individuals unique IDs."""
        n_unique = self.panel["individual_id"].nunique()
        assert n_unique == 200, (
            f"Expected 200 unique individuals, got {n_unique}"
        )

    def test_periods_correct(self):
        """Each individual should have exactly n_periods observations."""
        counts = self.panel.groupby("individual_id").size()
        assert (counts == 5).all(), (
            f"Not all individuals have 5 periods: {counts.value_counts().to_dict()}"
        )

    def test_selection_modes(self):
        """Both selection modes should produce valid data."""
        for mode in ["structural", "mechanical"]:
            dgp = SyntheticDGP(selection_mode=mode)
            panel = dgp.generate_panel_data(n_individuals=50, n_periods=5)
            assert len(panel) == 250, f"Mode {mode}: wrong panel size"
            assert panel["treatment"].sum() > 0, f"Mode {mode}: no treated"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
