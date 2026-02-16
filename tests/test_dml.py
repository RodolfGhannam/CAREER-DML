"""Unit tests for the DML estimation pipeline.

Tests verify that the CausalDMLPipeline wrapper produces valid
ATE estimates with proper inference objects.

Updated for v4.0 interface (Board Review, Feb 2026).
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.dml import CausalDMLPipeline


class TestCausalDMLPipeline:
    """Tests for the CausalDMLPipeline estimation wrapper."""

    def setup_method(self):
        """Create simple synthetic data for testing."""
        np.random.seed(42)
        n = 500
        self.X = np.random.randn(n, 5)
        self.T = (self.X[:, 0] + np.random.randn(n) > 0).astype(int)
        self.Y = 0.5 * self.T + self.X[:, 0] + np.random.randn(n) * 0.5
        self.pipeline = CausalDMLPipeline()

    def test_fit_predict_returns_tuple(self):
        """fit_predict should return (ate, cates, keep_idx)."""
        result = self.pipeline.fit_predict(self.Y, self.T, self.X)
        assert isinstance(result, tuple), "fit_predict() should return a tuple"
        assert len(result) == 3, f"Expected 3 values, got {len(result)}"

    def test_ate_is_finite(self):
        """ATE estimate should be a finite number."""
        ate, _, _ = self.pipeline.fit_predict(self.Y, self.T, self.X)
        assert np.isfinite(ate), f"ATE is not finite: {ate}"

    def test_ate_is_reasonable(self):
        """ATE estimate should be in a reasonable range for true ATE = 0.5."""
        ate, _, _ = self.pipeline.fit_predict(self.Y, self.T, self.X)
        assert -1.0 < ate < 2.0, f"ATE = {ate:.4f} is outside reasonable range"

    def test_cates_shape(self):
        """CATEs should have the same length as the kept observations."""
        ate, cates, keep_idx = self.pipeline.fit_predict(self.Y, self.T, self.X)
        assert len(cates) == len(keep_idx), (
            f"CATEs length ({len(cates)}) != keep_idx length ({len(keep_idx)})"
        )

    def test_inference_attributes(self):
        """After fit_predict, pipeline should have SE, CI, and p-value."""
        self.pipeline.fit_predict(self.Y, self.T, self.X)
        assert hasattr(self.pipeline, 'ate_se'), "Missing ate_se attribute"
        assert hasattr(self.pipeline, 'ate_ci'), "Missing ate_ci attribute"
        assert hasattr(self.pipeline, 'ate_pvalue'), "Missing ate_pvalue attribute"

    def test_se_is_positive(self):
        """Standard error should be positive."""
        self.pipeline.fit_predict(self.Y, self.T, self.X)
        assert self.pipeline.ate_se > 0, f"SE = {self.pipeline.ate_se:.4f} is not positive"

    def test_ci_contains_ate(self):
        """Confidence interval should contain the point estimate."""
        ate, _, _ = self.pipeline.fit_predict(self.Y, self.T, self.X)
        ci = self.pipeline.ate_ci
        assert ci[0] <= ate <= ci[1], (
            f"CI [{ci[0]:.4f}, {ci[1]:.4f}] does not contain ATE = {ate:.4f}"
        )

    def test_pvalue_is_valid(self):
        """p-value should be between 0 and 1."""
        self.pipeline.fit_predict(self.Y, self.T, self.X)
        p = self.pipeline.ate_pvalue
        assert 0 <= p <= 1, f"p-value = {p} is outside [0, 1]"

    def test_estimate_gates(self):
        """estimate_gates should return a DataFrame with n_groups rows."""
        self.pipeline.fit_predict(self.Y, self.T, self.X)
        X_df = pd.DataFrame(self.X)
        gates_df = self.pipeline.estimate_gates(X_df, n_groups=3)
        assert isinstance(gates_df, pd.DataFrame), "GATES should return a DataFrame"
        assert len(gates_df) == 3, f"Expected 3 groups, got {len(gates_df)}"

    def test_gates_heterogeneity_test(self):
        """test_gates_heterogeneity should return a dict with required keys."""
        self.pipeline.fit_predict(self.Y, self.T, self.X)
        X_df = pd.DataFrame(self.X)
        gates_df = self.pipeline.estimate_gates(X_df, n_groups=3)
        het_test = self.pipeline.test_gates_heterogeneity(gates_df)
        required_keys = ['q1_mean', 'q5_mean', 'difference', 't_statistic', 'p_value', 'significant']
        for key in required_keys:
            assert key in het_test, f"Missing key '{key}' in heterogeneity test"

    def test_propensity_trimming(self):
        """Propensity trimming should remove extreme observations."""
        ate, cates, keep_idx = self.pipeline.fit_predict(self.Y, self.T, self.X)
        # keep_idx should be a subset of all indices
        assert len(keep_idx) <= len(self.Y), "keep_idx is larger than input"
        assert len(keep_idx) > 0, "keep_idx is empty (all observations trimmed)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
