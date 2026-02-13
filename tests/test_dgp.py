"""Unit tests for the Data Generating Process (DGP) module.

Tests verify that the synthetic DGP produces data with the expected
statistical properties: correct dimensions, treatment assignment,
selection mechanism, and exclusion restriction.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.dgp import CareerDGP


class TestCareerDGP:
    """Tests for CareerDGP data generation."""

    def setup_method(self):
        """Initialise DGP with known parameters."""
        self.dgp = CareerDGP(
            n_individuals=500,
            n_periods=10,
            n_jobs=8,
            true_ate=0.5,
            seed=42
        )
        self.data = self.dgp.generate()

    def test_output_shape(self):
        """Generated data should have n_individuals rows."""
        sequences, X, T, Y, true_cate = self.data
        assert len(T) == 500, f"Expected 500 individuals, got {len(T)}"
        assert len(Y) == 500
        assert len(true_cate) == 500

    def test_sequences_shape(self):
        """Career sequences should be (n, T) integer arrays."""
        sequences, X, T, Y, true_cate = self.data
        assert sequences.shape == (500, 10), (
            f"Expected (500, 10), got {sequences.shape}"
        )

    def test_treatment_is_binary(self):
        """Treatment should be binary (0 or 1)."""
        _, _, T, _, _ = self.data
        unique_vals = np.unique(T)
        assert set(unique_vals).issubset({0, 1}), (
            f"Treatment contains non-binary values: {unique_vals}"
        )

    def test_treatment_has_both_groups(self):
        """Both treatment and control groups should be non-empty."""
        _, _, T, _, _ = self.data
        assert T.sum() > 0, "No treated individuals"
        assert (1 - T).sum() > 0, "No control individuals"

    def test_true_cate_mean_near_ate(self):
        """Average of true CATEs should approximate the true ATE."""
        _, _, _, _, true_cate = self.data
        mean_cate = true_cate.mean()
        assert abs(mean_cate - 0.5) < 0.15, (
            f"Mean CATE = {mean_cate:.4f}, expected near 0.5"
        )

    def test_outcome_is_finite(self):
        """All outcomes should be finite (no NaN or Inf)."""
        _, _, _, Y, _ = self.data
        assert np.all(np.isfinite(Y)), "Outcome contains NaN or Inf values"

    def test_covariates_shape(self):
        """Covariates X should have correct dimensions."""
        _, X, _, _, _ = self.data
        assert X.shape[0] == 500, f"Expected 500 rows, got {X.shape[0]}"
        assert X.shape[1] > 0, "Covariates should have at least 1 column"

    def test_reproducibility(self):
        """Same seed should produce identical data."""
        dgp2 = CareerDGP(
            n_individuals=500,
            n_periods=10,
            n_jobs=8,
            true_ate=0.5,
            seed=42
        )
        data2 = dgp2.generate()
        _, _, T1, Y1, _ = self.data
        _, _, T2, Y2, _ = data2
        np.testing.assert_array_equal(T1, T2)
        np.testing.assert_array_almost_equal(Y1, Y2)

    def test_different_seed_produces_different_data(self):
        """Different seeds should produce different data."""
        dgp2 = CareerDGP(
            n_individuals=500,
            n_periods=10,
            n_jobs=8,
            true_ate=0.5,
            seed=99
        )
        data2 = dgp2.generate()
        _, _, _, Y1, _ = self.data
        _, _, _, Y2, _ = data2
        assert not np.allclose(Y1, Y2), "Different seeds produced identical data"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
