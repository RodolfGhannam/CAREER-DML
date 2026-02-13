"""Unit tests for the DML estimation pipeline.

Tests verify that the CausalForestDML wrapper produces valid
ATE estimates with proper inference objects.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.dml import CausalDML


class TestCausalDML:
    """Tests for the CausalDML estimation wrapper."""

    def setup_method(self):
        """Create simple synthetic data for testing."""
        np.random.seed(42)
        n = 200
        self.X = np.random.randn(n, 5)
        self.T = (self.X[:, 0] + np.random.randn(n) > 0).astype(float)
        self.Y = 0.5 * self.T + self.X[:, 0] + np.random.randn(n) * 0.5
        self.dml = CausalDML()

    def test_fit_returns_result(self):
        """Fitting should return a result dictionary."""
        result = self.dml.fit(self.Y, self.T, self.X)
        assert isinstance(result, dict), "fit() should return a dict"

    def test_result_contains_ate(self):
        """Result should contain an ATE estimate."""
        result = self.dml.fit(self.Y, self.T, self.X)
        assert 'ate' in result, "Result missing 'ate' key"
        assert np.isfinite(result['ate']), "ATE is not finite"

    def test_result_contains_inference(self):
        """Result should contain SE, CI, and p-value."""
        result = self.dml.fit(self.Y, self.T, self.X)
        for key in ['se', 'ci_lower', 'ci_upper', 'p_value']:
            assert key in result, f"Result missing '{key}' key"

    def test_ate_is_reasonable(self):
        """ATE estimate should be in a reasonable range for true ATE = 0.5."""
        result = self.dml.fit(self.Y, self.T, self.X)
        ate = result['ate']
        assert -1.0 < ate < 2.0, f"ATE = {ate:.4f} is outside reasonable range"

    def test_se_is_positive(self):
        """Standard error should be positive."""
        result = self.dml.fit(self.Y, self.T, self.X)
        assert result['se'] > 0, f"SE = {result['se']:.4f} is not positive"

    def test_ci_contains_ate(self):
        """Confidence interval should contain the point estimate."""
        result = self.dml.fit(self.Y, self.T, self.X)
        assert result['ci_lower'] <= result['ate'] <= result['ci_upper'], (
            f"CI [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}] "
            f"does not contain ATE = {result['ate']:.4f}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
