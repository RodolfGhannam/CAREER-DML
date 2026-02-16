"""End-to-end integration test for the CAREER-DML pipeline.

This test runs a minimal version of the full pipeline (small n, few epochs)
to verify that all components work together without errors.

Updated for v4.0 interface.
"""

import sys
import os
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.dgp import SyntheticDGP
from src.embeddings import (
    PredictiveGRU, CausalGRU, DebiasedGRU, Adversary,
    train_predictive_embedding, train_causal_embedding, train_debiased_embedding,
)
from src.dml import CausalDMLPipeline


# =============================================================================
# Helper functions (same as main.py)
# =============================================================================

def pad_sequences(sequences, max_len=5):
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padded.append(seq + [0] * (max_len - len(seq)))
        else:
            padded.append(seq[:max_len])
    return torch.tensor(padded, dtype=torch.long)


def extract_embeddings(model, sequences_tensor):
    model.eval()
    with torch.no_grad():
        return model.get_representation(sequences_tensor).numpy()


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end pipeline integration tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Generate synthetic data once for all tests."""
        np.random.seed(42)
        torch.manual_seed(42)

        self.N = 100
        self.N_PERIODS = 5
        self.N_OCC = 50
        self.EMBED_DIM = 8
        self.HIDDEN_DIM = 16
        self.PHI_DIM = 8
        self.EPOCHS = 3
        self.BATCH_SIZE = 32

        dgp = SyntheticDGP(selection_mode="structural")
        self.panel = dgp.generate_panel_data(
            n_individuals=self.N, n_periods=self.N_PERIODS
        )
        self.params = dgp.get_true_parameters()
        self.true_ate = self.params["true_ate"]

        # Prepare cross-sectional data
        final = self.panel.groupby("individual_id").last().reset_index()
        self.sequences_tensor = pad_sequences(
            final["career_sequence_history"].tolist(), max_len=self.N_PERIODS
        )
        self.Y = final["outcome"].values
        self.T = final["treatment"].values

        treatments_t = torch.tensor(self.T, dtype=torch.long)
        outcomes_t = torch.tensor(self.Y, dtype=torch.float32)
        dataset = TensorDataset(self.sequences_tensor, treatments_t, outcomes_t)
        self.loader = DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=True)

    def test_dgp_generates_valid_panel(self):
        """DGP should generate a panel with expected columns."""
        required_cols = ["individual_id", "period", "treatment", "outcome",
                         "career_sequence_history"]
        for col in required_cols:
            assert col in self.panel.columns, f"Missing column: {col}"
        assert len(self.panel) == self.N * self.N_PERIODS, (
            f"Expected {self.N * self.N_PERIODS} rows, got {len(self.panel)}"
        )

    def test_predictive_gru_pipeline(self):
        """Full pipeline with Predictive GRU should run without error."""
        model = PredictiveGRU(
            vocab_size=self.N_OCC, embedding_dim=self.EMBED_DIM,
            hidden_dim=self.HIDDEN_DIM
        )
        train_predictive_embedding(model, self.loader, epochs=self.EPOCHS)
        X_embed = extract_embeddings(model, self.sequences_tensor)

        pipeline = CausalDMLPipeline()
        ate, cates, keep_idx = pipeline.fit_predict(self.Y, self.T, X_embed)

        assert np.isfinite(ate), f"ATE is not finite: {ate}"
        assert pipeline.ate_se > 0, "SE is not positive"
        assert len(cates) > 0, "No CATEs returned"

    def test_causal_gru_pipeline(self):
        """Full pipeline with Causal GRU (VIB) should run without error."""
        model = CausalGRU(
            vocab_size=self.N_OCC, embedding_dim=self.EMBED_DIM,
            hidden_dim=self.HIDDEN_DIM, phi_dim=self.PHI_DIM
        )
        train_causal_embedding(model, self.loader, epochs=self.EPOCHS)
        X_embed = extract_embeddings(model, self.sequences_tensor)

        pipeline = CausalDMLPipeline()
        ate, cates, keep_idx = pipeline.fit_predict(self.Y, self.T, X_embed)

        assert np.isfinite(ate), f"ATE is not finite: {ate}"
        assert len(cates) > 0, "No CATEs returned"

    def test_debiased_gru_pipeline(self):
        """Full pipeline with Debiased GRU (Adversarial) should run without error."""
        encoder = DebiasedGRU(
            vocab_size=self.N_OCC, embedding_dim=self.EMBED_DIM,
            hidden_dim=self.HIDDEN_DIM, phi_dim=self.PHI_DIM
        )
        adversary = Adversary(phi_dim=self.PHI_DIM)
        train_debiased_embedding(encoder, adversary, self.loader, epochs=self.EPOCHS)
        X_embed = extract_embeddings(encoder, self.sequences_tensor)

        pipeline = CausalDMLPipeline()
        ate, cates, keep_idx = pipeline.fit_predict(self.Y, self.T, X_embed)

        assert np.isfinite(ate), f"ATE is not finite: {ate}"
        assert len(cates) > 0, "No CATEs returned"

    def test_ate_is_reasonable(self):
        """ATE from Predictive GRU should be within a generous range of true ATE."""
        model = PredictiveGRU(
            vocab_size=self.N_OCC, embedding_dim=self.EMBED_DIM,
            hidden_dim=self.HIDDEN_DIM
        )
        train_predictive_embedding(model, self.loader, epochs=self.EPOCHS)
        X_embed = extract_embeddings(model, self.sequences_tensor)

        pipeline = CausalDMLPipeline()
        ate, _, _ = pipeline.fit_predict(self.Y, self.T, X_embed)

        # With small N and few epochs, generous bound
        bias = abs(ate - self.true_ate)
        assert bias < 1.0, (
            f"ATE = {ate:.4f}, true = {self.true_ate:.4f}, bias = {bias:.4f} exceeds 1.0"
        )

    def test_embedding_dimensions_consistent(self):
        """All embedding variants should produce arrays compatible with DML."""
        models = {
            "Predictive": PredictiveGRU(
                vocab_size=self.N_OCC, embedding_dim=self.EMBED_DIM,
                hidden_dim=self.HIDDEN_DIM
            ),
            "Causal": CausalGRU(
                vocab_size=self.N_OCC, embedding_dim=self.EMBED_DIM,
                hidden_dim=self.HIDDEN_DIM, phi_dim=self.PHI_DIM
            ),
            "Debiased": DebiasedGRU(
                vocab_size=self.N_OCC, embedding_dim=self.EMBED_DIM,
                hidden_dim=self.HIDDEN_DIM, phi_dim=self.PHI_DIM
            ),
        }

        for name, model in models.items():
            model.eval()
            rep = extract_embeddings(model, self.sequences_tensor)
            assert rep.ndim == 2, f"{name}: Expected 2D array, got {rep.ndim}D"
            assert rep.shape[0] == self.N, (
                f"{name}: Expected {self.N} rows, got {rep.shape[0]}"
            )
            assert np.all(np.isfinite(rep)), f"{name}: Contains NaN or Inf"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
