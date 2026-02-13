"""End-to-end integration test for the CAREER-DML pipeline.

This test runs a minimal version of the full pipeline (small n, few epochs)
to verify that all components work together without errors.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.dgp import CareerDGP
from src.dml import CausalDML


class TestIntegration:
    """End-to-end pipeline integration tests."""

    def test_dgp_to_dml_pipeline(self):
        """Full pipeline from DGP to DML estimation should run without error."""
        # Generate data
        dgp = CareerDGP(
            n_individuals=100,
            n_periods=5,
            n_jobs=8,
            true_ate=0.5,
            seed=42
        )
        sequences, X, T, Y, true_cate = dgp.generate()

        # Use raw covariates (skip embedding for speed)
        dml = CausalDML()
        result = dml.fit(Y, T, X)

        # Verify output structure
        assert 'ate' in result
        assert 'se' in result
        assert np.isfinite(result['ate'])
        assert result['se'] > 0

    def test_pipeline_with_known_ate(self):
        """Pipeline should recover a reasonable ATE from well-specified DGP."""
        dgp = CareerDGP(
            n_individuals=500,
            n_periods=5,
            n_jobs=8,
            true_ate=0.5,
            seed=42
        )
        sequences, X, T, Y, true_cate = dgp.generate()

        dml = CausalDML()
        result = dml.fit(Y, T, X)

        # ATE should be within 0.5 of true value (generous bound for raw X)
        bias = abs(result['ate'] - 0.5)
        assert bias < 0.5, (
            f"ATE = {result['ate']:.4f}, bias = {bias:.4f} exceeds 0.5"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
