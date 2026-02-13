"""Unit tests for the GRU embedding variants.

Tests verify that each of the three embedding architectures
(Predictive, Causal VIB, Debiased Adversarial) produces
well-formed embeddings with expected dimensionality.
"""

import sys
import os
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.embeddings import PredictiveGRU, CausalGRU, DebiasedGRU


class TestPredictiveGRU:
    """Tests for the Predictive GRU baseline."""

    def setup_method(self):
        """Initialise model with small dimensions for testing."""
        self.model = PredictiveGRU(
            n_jobs=8,
            embedding_dim=16,
            hidden_dim=32,
            output_dim=16
        )

    def test_output_shape(self):
        """Forward pass should return embeddings of correct shape."""
        x = torch.randint(0, 8, (10, 5))  # batch=10, seq_len=5
        embeddings = self.model(x)
        assert embeddings.shape == (10, 16), (
            f"Expected (10, 16), got {embeddings.shape}"
        )

    def test_output_is_finite(self):
        """Embeddings should not contain NaN or Inf."""
        x = torch.randint(0, 8, (10, 5))
        embeddings = self.model(x)
        assert torch.all(torch.isfinite(embeddings)), (
            "Embeddings contain NaN or Inf"
        )

    def test_deterministic_in_eval_mode(self):
        """Model in eval mode should produce deterministic output."""
        self.model.eval()
        x = torch.randint(0, 8, (5, 5))
        e1 = self.model(x)
        e2 = self.model(x)
        torch.testing.assert_close(e1, e2)


class TestCausalGRU:
    """Tests for the Causal GRU with VIB regularisation."""

    def setup_method(self):
        """Initialise model."""
        self.model = CausalGRU(
            n_jobs=8,
            embedding_dim=16,
            hidden_dim=32,
            output_dim=16,
            beta=0.1
        )

    def test_output_shape(self):
        """Forward pass should return embeddings of correct shape."""
        x = torch.randint(0, 8, (10, 5))
        embeddings = self.model(x)
        assert embeddings.shape == (10, 16)

    def test_kl_loss_is_nonnegative(self):
        """KL divergence loss should be non-negative."""
        x = torch.randint(0, 8, (10, 5))
        self.model.train()
        _ = self.model(x)
        kl = self.model.kl_loss()
        assert kl.item() >= 0, f"KL loss is negative: {kl.item()}"


class TestDebiasedGRU:
    """Tests for the Debiased GRU with adversarial training."""

    def setup_method(self):
        """Initialise model."""
        self.model = DebiasedGRU(
            n_jobs=8,
            embedding_dim=16,
            hidden_dim=32,
            output_dim=16,
            lambda_adv=1.0
        )

    def test_output_shape(self):
        """Forward pass should return embeddings of correct shape."""
        x = torch.randint(0, 8, (10, 5))
        embeddings = self.model(x)
        assert embeddings.shape == (10, 16)

    def test_output_is_finite(self):
        """Embeddings should not contain NaN or Inf."""
        x = torch.randint(0, 8, (10, 5))
        embeddings = self.model(x)
        assert torch.all(torch.isfinite(embeddings))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
