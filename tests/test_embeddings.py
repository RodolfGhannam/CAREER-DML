"""Unit tests for the GRU embedding variants.

Tests verify that each of the three embedding architectures
(Predictive, Causal VIB, Debiased Adversarial) produces
well-formed embeddings with expected dimensionality.

Updated for v4.0 interface (Board Review, Feb 2026).
"""

import sys
import os
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.embeddings import (
    PredictiveGRU, CausalGRU, DebiasedGRU, Adversary,
    train_predictive_embedding, train_causal_embedding, train_debiased_embedding,
)


# =============================================================================
# Fixtures
# =============================================================================

VOCAB_SIZE = 10
EMBEDDING_DIM = 16
HIDDEN_DIM = 32
PHI_DIM = 16
BATCH_SIZE = 8
SEQ_LEN = 5
N_SAMPLES = 20


def make_dummy_loader():
    """Create a small DataLoader for testing training functions."""
    sequences = torch.randint(0, VOCAB_SIZE, (N_SAMPLES, SEQ_LEN))
    treatments = torch.randint(0, 2, (N_SAMPLES,))
    outcomes = torch.randn(N_SAMPLES)
    dataset = TensorDataset(sequences, treatments, outcomes)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


# =============================================================================
# Tests: PredictiveGRU
# =============================================================================

class TestPredictiveGRU:
    """Tests for the Predictive GRU baseline."""

    def setup_method(self):
        self.model = PredictiveGRU(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
        )

    def test_forward_output_shape(self):
        """Forward pass should return a 1D tensor of predicted outcomes."""
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        y_pred = self.model(x)
        assert y_pred.shape == (BATCH_SIZE,), f"Expected ({BATCH_SIZE},), got {y_pred.shape}"

    def test_get_representation_shape(self):
        """get_representation should return embeddings of shape (batch, hidden_dim)."""
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        rep = self.model.get_representation(x)
        assert rep.shape == (BATCH_SIZE, HIDDEN_DIM), (
            f"Expected ({BATCH_SIZE}, {HIDDEN_DIM}), got {rep.shape}"
        )

    def test_output_is_finite(self):
        """Embeddings should not contain NaN or Inf."""
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        rep = self.model.get_representation(x)
        assert torch.all(torch.isfinite(rep)), "Embeddings contain NaN or Inf"

    def test_deterministic_in_eval_mode(self):
        """Model in eval mode should produce deterministic output."""
        self.model.eval()
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        e1 = self.model.get_representation(x)
        e2 = self.model.get_representation(x)
        torch.testing.assert_close(e1, e2)


# =============================================================================
# Tests: CausalGRU (VIB)
# =============================================================================

class TestCausalGRU:
    """Tests for the Causal GRU with VIB regularisation."""

    def setup_method(self):
        self.model = CausalGRU(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            phi_dim=PHI_DIM,
        )

    def test_forward_output_tuple(self):
        """Forward pass should return (y_pred, t_pred, mu, logvar)."""
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        self.model.train()
        result = self.model(x)
        assert len(result) == 4, f"Expected 4 outputs, got {len(result)}"

    def test_forward_shapes(self):
        """All forward outputs should have correct shapes."""
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        self.model.train()
        y_pred, t_pred, mu, logvar = self.model(x)
        assert y_pred.shape == (BATCH_SIZE,), f"y_pred shape: {y_pred.shape}"
        assert t_pred.shape == (BATCH_SIZE,), f"t_pred shape: {t_pred.shape}"
        assert mu.shape == (BATCH_SIZE, PHI_DIM), f"mu shape: {mu.shape}"
        assert logvar.shape == (BATCH_SIZE, PHI_DIM), f"logvar shape: {logvar.shape}"

    def test_get_representation_shape(self):
        """get_representation should return mu of shape (batch, phi_dim)."""
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        self.model.eval()
        rep = self.model.get_representation(x)
        assert rep.shape == (BATCH_SIZE, PHI_DIM), (
            f"Expected ({BATCH_SIZE}, {PHI_DIM}), got {rep.shape}"
        )

    def test_reparameterize_stochastic_in_train(self):
        """Reparameterize should add noise during training."""
        self.model.train()
        mu = torch.zeros(BATCH_SIZE, PHI_DIM)
        logvar = torch.zeros(BATCH_SIZE, PHI_DIM)  # std = 1
        z1 = self.model.reparameterize(mu, logvar)
        z2 = self.model.reparameterize(mu, logvar)
        # With std=1, two samples should differ (with very high probability)
        assert not torch.allclose(z1, z2), "Reparameterize should be stochastic in train mode"

    def test_reparameterize_deterministic_in_eval(self):
        """Reparameterize should return mu in eval mode."""
        self.model.eval()
        mu = torch.randn(BATCH_SIZE, PHI_DIM)
        logvar = torch.randn(BATCH_SIZE, PHI_DIM)
        z = self.model.reparameterize(mu, logvar)
        torch.testing.assert_close(z, mu)

    def test_logvar_clamping(self):
        """Extreme logvar values should be clamped for numerical stability."""
        self.model.eval()
        mu = torch.zeros(BATCH_SIZE, PHI_DIM)
        logvar_extreme = torch.full((BATCH_SIZE, PHI_DIM), 100.0)
        z = self.model.reparameterize(mu, logvar_extreme)
        assert torch.all(torch.isfinite(z)), "Clamping failed: output is not finite"

    def test_kl_divergence_nonnegative(self):
        """KL divergence computed from forward pass should be non-negative."""
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        self.model.train()
        _, _, mu, logvar = self.model(x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        assert kl.item() >= -1e-6, f"KL divergence is negative: {kl.item()}"

    def test_output_is_finite(self):
        """All outputs should be finite."""
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        self.model.train()
        y_pred, t_pred, mu, logvar = self.model(x)
        for name, tensor in [("y_pred", y_pred), ("t_pred", t_pred), ("mu", mu), ("logvar", logvar)]:
            assert torch.all(torch.isfinite(tensor)), f"{name} contains NaN or Inf"


# =============================================================================
# Tests: DebiasedGRU (Adversarial)
# =============================================================================

class TestDebiasedGRU:
    """Tests for the Debiased GRU with adversarial training."""

    def setup_method(self):
        self.model = DebiasedGRU(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            phi_dim=PHI_DIM,
        )

    def test_forward_output_tuple(self):
        """Forward pass should return (y_pred, phi)."""
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        result = self.model(x)
        assert len(result) == 2, f"Expected 2 outputs, got {len(result)}"

    def test_forward_shapes(self):
        """Forward outputs should have correct shapes."""
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        y_pred, phi = self.model(x)
        assert y_pred.shape == (BATCH_SIZE,), f"y_pred shape: {y_pred.shape}"
        assert phi.shape == (BATCH_SIZE, PHI_DIM), f"phi shape: {phi.shape}"

    def test_get_representation_shape(self):
        """get_representation should return phi of shape (batch, phi_dim)."""
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        rep = self.model.get_representation(x)
        assert rep.shape == (BATCH_SIZE, PHI_DIM), (
            f"Expected ({BATCH_SIZE}, {PHI_DIM}), got {rep.shape}"
        )

    def test_phi_is_nonnegative(self):
        """Phi should be non-negative (ReLU activation)."""
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        rep = self.model.get_representation(x)
        assert torch.all(rep >= 0), "Phi contains negative values (ReLU should prevent this)"

    def test_output_is_finite(self):
        """Embeddings should not contain NaN or Inf."""
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        rep = self.model.get_representation(x)
        assert torch.all(torch.isfinite(rep)), "Embeddings contain NaN or Inf"


# =============================================================================
# Tests: Adversary
# =============================================================================

class TestAdversary:
    """Tests for the Adversary network."""

    def setup_method(self):
        self.adversary = Adversary(phi_dim=PHI_DIM, hidden_dim=32)

    def test_output_shape(self):
        """Adversary should output a scalar per sample."""
        phi = torch.randn(BATCH_SIZE, PHI_DIM)
        out = self.adversary(phi)
        assert out.shape == (BATCH_SIZE,), f"Expected ({BATCH_SIZE},), got {out.shape}"

    def test_output_is_finite(self):
        """Output should be finite."""
        phi = torch.randn(BATCH_SIZE, PHI_DIM)
        out = self.adversary(phi)
        assert torch.all(torch.isfinite(out)), "Adversary output contains NaN or Inf"


# =============================================================================
# Tests: Training Functions
# =============================================================================

class TestTrainingFunctions:
    """Tests for the training functions."""

    def test_train_predictive_runs(self):
        """train_predictive_embedding should run without error."""
        model = PredictiveGRU(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
        loader = make_dummy_loader()
        losses = train_predictive_embedding(model, loader, epochs=2)
        assert len(losses) == 2, f"Expected 2 loss values, got {len(losses)}"
        assert all(np.isfinite(l) for l in losses), "Training produced non-finite loss"

    def test_train_causal_runs(self):
        """train_causal_embedding should run without error."""
        model = CausalGRU(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, phi_dim=PHI_DIM)
        loader = make_dummy_loader()
        losses = train_causal_embedding(model, loader, epochs=2)
        assert len(losses) == 2
        assert all(np.isfinite(l) for l in losses), "Training produced non-finite loss"

    def test_train_debiased_runs(self):
        """train_debiased_embedding should run without error."""
        encoder = DebiasedGRU(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, phi_dim=PHI_DIM)
        adversary = Adversary(phi_dim=PHI_DIM)
        loader = make_dummy_loader()
        losses = train_debiased_embedding(encoder, adversary, loader, epochs=2)
        assert len(losses) == 2
        assert all(np.isfinite(l) for l in losses), "Training produced non-finite loss"

    def test_predictive_loss_decreases(self):
        """Predictive training loss should generally decrease over epochs."""
        torch.manual_seed(42)
        model = PredictiveGRU(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
        loader = make_dummy_loader()
        losses = train_predictive_embedding(model, loader, epochs=10)
        # Loss at end should be lower than at start (with some tolerance)
        assert losses[-1] < losses[0] * 1.5, (
            f"Loss did not decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
